from __future__ import annotations

import cv2
import numpy as np

from ricevision_qi.app.core.config import merge_config
from ricevision_qi.app.core.postprocessing import postprocess_instances
from ricevision_qi.app.core.preprocessing import preprocess_image
from ricevision_qi.app.core.segmentation import connected_components_segment, watershed_segment


class SegmentationBackend:
    name = "base"

    def segment(self, image, config):
        raise NotImplementedError


class OpenCVThresholdBackend(SegmentationBackend):
    name = "opencv_threshold"

    def segment(self, image, config):
        cfg = merge_config(config)
        mask, corrected, debug = preprocess_image(image, cfg)
        instances = connected_components_segment(mask, cfg)
        instances = postprocess_instances(instances, image.shape[:2], cfg)
        debug_results = _debug_results(image, corrected, debug, mask, np.zeros(mask.shape, dtype=np.int32))
        return {"instances": instances, "debug_results": debug_results, "stats": _stats(mask, instances, image)}


class OpenCVWatershedBackend(SegmentationBackend):
    name = "opencv_watershed"

    def segment(self, image, config):
        cfg = merge_config(config)
        mask, corrected, debug = preprocess_image(image, cfg)
        if cfg.get("watershed_enabled", True):
            instances, markers = watershed_segment(image, mask, cfg)
            if not instances:
                instances = connected_components_segment(mask, cfg)
        else:
            markers = np.zeros(mask.shape, dtype=np.int32)
            instances = connected_components_segment(mask, cfg)
        instances = postprocess_instances(instances, image.shape[:2], cfg)
        debug_results = _debug_results(image, corrected, debug, mask, markers)
        return {"instances": instances, "debug_results": debug_results, "stats": _stats(mask, instances, image)}


class EdgeEnhancedWatershedBackend(OpenCVWatershedBackend):
    name = "edge_enhanced_watershed"

    def segment(self, image, config):
        enhanced = _edge_enhance(image)
        output = super().segment(enhanced, config)
        output["debug_results"]["edge_enhanced"] = enhanced
        return output


def create_segmentation_backend(name: str | None) -> SegmentationBackend:
    backend_name = (name or "opencv_watershed").strip()
    if backend_name == "opencv_threshold":
        return OpenCVThresholdBackend()
    if backend_name == "edge_enhanced_watershed":
        return EdgeEnhancedWatershedBackend()
    if backend_name == "yolo_seg":
        from ricevision_qi.app.core.yolo_seg_backend import YoloSegBackend

        return YoloSegBackend()
    return OpenCVWatershedBackend()


def _debug_results(image, corrected, preprocess_debug, mask, markers):
    return {
        "original": image,
        "illumination_corrected": preprocess_debug.get("illumination_corrected", corrected),
        "initial_mask": preprocess_debug.get("initial_mask", mask),
        "cleaned_mask": preprocess_debug.get("cleaned_mask", mask),
        "watershed_markers": markers,
        "overlay": image,
    }


def _stats(mask, instances, image):
    image_area = float(image.shape[0] * image.shape[1]) if image is not None and image.size else 0.0
    foreground = float(np.count_nonzero(mask)) if mask is not None and mask.size else 0.0
    return {
        "num_instances": len(instances),
        "foreground_area_ratio": foreground / image_area if image_area else 0.0,
        "possible_merge_count": sum(1 for item in instances if item.status == "possible_merged"),
        "possible_split_count": sum(1 for item in instances if item.status == "possible_split"),
        "border_artifact_count": sum(1 for item in instances if item.status == "border_artifact"),
    }


def _edge_enhance(image):
    if image is None or image.size == 0:
        return image
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l_channel)
    blurred = cv2.GaussianBlur(clahe, (0, 0), 1.2)
    sharpened = cv2.addWeighted(clahe, 1.35, blurred, -0.35, 0)
    return cv2.cvtColor(cv2.merge((sharpened, a_channel, b_channel)), cv2.COLOR_LAB2RGB)

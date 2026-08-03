from dataclasses import dataclass

import numpy as np
import cv2

from ricevision_qi.app.core.classification_rules import classify_grain
from ricevision_qi.app.core.config import merge_config
from ricevision_qi.app.core.measurement import measure_grain
from ricevision_qi.app.core.segmentation_backends import create_segmentation_backend
from ricevision_qi.app.core.visualization import draw_detection_overlay
from ricevision_qi.app.data.models import normalize_roi


@dataclass
class DetectionResult:
    original_image: np.ndarray
    overlay_image: np.ndarray
    grains: list
    summary: dict
    debug_results: dict
    roi: tuple[int, int, int, int] | None = None
    backend_stats: dict | None = None
    profile_name: str = "default"


def run_detection(image_rgb, config=None):
    cfg = merge_config(config)
    if image_rgb is None or image_rgb.size == 0:
        empty = np.zeros((0, 0, 3), dtype=np.uint8)
        return DetectionResult(
            original_image=empty,
            overlay_image=empty,
            grains=[],
            summary=_summary([]),
            debug_results={
                "original": empty,
                "illumination_corrected": empty,
                "initial_mask": np.zeros((0, 0), dtype=np.uint8),
                "cleaned_mask": np.zeros((0, 0), dtype=np.uint8),
                "watershed_markers": np.zeros((0, 0), dtype=np.int32),
                "overlay": empty,
            },
            roi=None,
            backend_stats={},
            profile_name=cfg.get("profile_name", "default"),
        )

    roi = normalize_roi(cfg.get("roi"), image_rgb.shape[:2])
    roi_image = _crop_image(image_rgb, roi)
    processing_image, scale = _processing_image(roi_image, cfg)

    backend = create_segmentation_backend(cfg.get("segmentation_backend", cfg.get("backend")))
    backend_output = backend.segment(processing_image, cfg)
    grains = backend_output.get("instances", [])
    if scale != 1.0:
        grains = _scale_grains_to_original(grains, scale)
    if roi is not None:
        grains = _offset_grains_to_original(grains, roi[0], roi[1])
    pixel_per_mm = cfg.get("pixel_per_mm")

    for grain in grains:
        grain.features = measure_grain(image_rgb, grain, pixel_per_mm=pixel_per_mm)

    reference_stats = _reference_stats(grains)
    for grain in grains:
        grain.classification, grain.confidence = classify_grain(
            grain.features,
            reference_stats=reference_stats,
            config=cfg,
        )
        grain.predicted_class = grain.classification
        grain.final_class = grain.manual_class or grain.predicted_class

    overlay = draw_detection_overlay(image_rgb, grains)
    debug_results = _debug_results_to_original(
        image_rgb,
        backend_output.get("debug_results", {}),
        overlay,
        roi,
        scale,
    )
    return DetectionResult(
        original_image=image_rgb,
        overlay_image=overlay,
        grains=grains,
        summary=_summary(grains),
        debug_results=debug_results,
        roi=roi,
        backend_stats=backend_output.get("stats", {}),
        profile_name=cfg.get("profile_name", "default"),
    )


class DetectionPipeline:
    def run(self, image_rgb, config=None):
        return run_detection(image_rgb, config=config)


def _reference_stats(grains):
    if not grains:
        return {"mean_length_px": 0.0, "mean_area_px": 0.0}
    return {
        "mean_length_px": float(np.mean([grain.features["length_px"] for grain in grains])),
        "mean_area_px": float(np.mean([grain.features["area_px"] for grain in grains])),
    }


def _summary(grains):
    counts = {
        "total_grains": len(grains),
        "normal_count": 0,
        "broken_count": 0,
        "defective_count": 0,
        "impurity_count": 0,
        "unknown_count": 0,
    }
    for grain in grains:
        if getattr(grain, "status", "valid") == "false_positive":
            continue
        key = f"{getattr(grain, 'final_class', grain.classification)}_count"
        if key in counts:
            counts[key] += 1
        else:
            counts["unknown_count"] += 1
    counts["total_grains"] = sum(counts[key] for key in counts if key.endswith("_count"))
    return counts


def _processing_image(image_rgb, cfg):
    max_dimension = int(cfg.get("max_processing_dimension", 1800) or 0)
    height, width = image_rgb.shape[:2]
    largest = max(height, width)
    if max_dimension <= 0 or largest <= max_dimension:
        return image_rgb, 1.0

    scale = max_dimension / float(largest)
    resized = cv2.resize(
        image_rgb,
        (int(round(width * scale)), int(round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def _scale_grains_to_original(grains, scale):
    inverse = 1.0 / scale
    for grain in grains:
        grain.contour = np.round(grain.contour.astype(np.float32) * inverse).astype(np.int32)
        x, y, w, h = cv2.boundingRect(grain.contour)
        grain.bbox = (int(x), int(y), int(w), int(h))
        grain.area_px = float(grain.area_px * inverse * inverse)
        grain.center_x = float(grain.center_x * inverse)
        grain.center_y = float(grain.center_y * inverse)
        grain.mask = np.zeros((1, 1), dtype=np.uint8)
    return grains


def _offset_grains_to_original(grains, offset_x, offset_y):
    offset = np.array([[[int(offset_x), int(offset_y)]]], dtype=np.int32)
    for grain in grains:
        grain.contour = grain.contour.astype(np.int32) + offset
        x, y, w, h = cv2.boundingRect(grain.contour)
        grain.bbox = (int(x), int(y), int(w), int(h))
        grain.center_x = float(grain.center_x + offset_x)
        grain.center_y = float(grain.center_y + offset_y)
        grain.metadata["roi_offset"] = (int(offset_x), int(offset_y))
    return grains


def _crop_image(image, roi):
    if roi is None:
        return image
    x, y, w, h = roi
    return image[y : y + h, x : x + w].copy()


def _debug_results_to_original(original, backend_debug, overlay, roi, scale):
    corrected = backend_debug.get("illumination_corrected", original)
    initial_mask = backend_debug.get("initial_mask", np.zeros(original.shape[:2], dtype=np.uint8))
    cleaned_mask = backend_debug.get("cleaned_mask", np.zeros(original.shape[:2], dtype=np.uint8))
    markers = backend_debug.get("watershed_markers", np.zeros(initial_mask.shape, dtype=np.int32))
    if scale != 1.0:
        target_shape = _crop_image(original, roi).shape
        size = (target_shape[1], target_shape[0])
        corrected = cv2.resize(corrected, size, interpolation=cv2.INTER_LINEAR)
        initial_mask = cv2.resize(initial_mask, size, interpolation=cv2.INTER_NEAREST)
        cleaned_mask = cv2.resize(cleaned_mask, size, interpolation=cv2.INTER_NEAREST)
        markers = cv2.resize(markers.astype(np.int32), size, interpolation=cv2.INTER_NEAREST)
    if roi is not None:
        corrected = _paste_roi(original, corrected, roi, is_mask=False)
        initial_mask = _paste_roi(np.zeros(original.shape[:2], dtype=np.uint8), initial_mask, roi, is_mask=True)
        cleaned_mask = _paste_roi(np.zeros(original.shape[:2], dtype=np.uint8), cleaned_mask, roi, is_mask=True)
        markers = _paste_roi(np.zeros(original.shape[:2], dtype=np.int32), markers, roi, is_mask=True)
    return {
        "original": original,
        "illumination_corrected": corrected,
        "initial_mask": initial_mask,
        "cleaned_mask": cleaned_mask,
        "watershed_markers": markers,
        "overlay": overlay,
    }


def _paste_roi(canvas, roi_image, roi, is_mask):
    x, y, w, h = roi
    result = canvas.copy()
    if roi_image.shape[0] != h or roi_image.shape[1] != w:
        interpolation = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
        roi_image = cv2.resize(roi_image, (w, h), interpolation=interpolation)
    result[y : y + h, x : x + w] = roi_image
    return result

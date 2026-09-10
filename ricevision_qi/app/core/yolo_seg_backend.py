from pathlib import Path

import cv2
import numpy as np

from ricevision_qi.app.core.segmentation import GrainInstance


class YoloSegBackend:
    name = "yolo_seg"

    def __init__(self, weights: str | None = None) -> None:
        self.weights = str(weights) if weights else None
        self.model = None
        self.message = "YOLO-seg backend is not configured."
        if self.weights:
            self._load_model(self.weights)

    def segment(self, image, config):
        height, width = image.shape[:2] if image is not None and image.size else (1, 1)
        empty_mask = np.zeros((height, width), dtype=np.uint8)
        weights = config.get("weights") or config.get("yolo_weights") or self.weights
        if self.model is None:
            if not weights:
                self.message = "YOLO-seg weights not configured."
            elif not Path(str(weights)).exists():
                self.message = f"YOLO-seg weights not found: {weights}"
            else:
                self._load_model(str(weights))
        if self.model is None:
            return _empty_output(image, empty_mask, self.message)

        try:
            results = self.model.predict(
                image,
                imgsz=int(config.get("imgsz", 640) or 640),
                conf=float(config.get("conf", 0.25) or 0.25),
                iou=float(config.get("iou", 0.5) or 0.5),
                max_det=int(config.get("max_det", 1000) or 1000),
                verbose=False,
            )
        except Exception as exc:
            return _empty_output(image, empty_mask, f"YOLO-seg inference failed: {exc}")

        instances = _instances_from_results(results, (height, width))
        foreground = _merge_instance_masks(instances, (height, width))
        return {
            "instances": instances,
            "debug_results": {
                "original": image,
                "illumination_corrected": image,
                "initial_mask": foreground,
                "cleaned_mask": foreground,
                "watershed_markers": np.zeros((height, width), dtype=np.int32),
                "overlay": image,
            },
            "stats": {
                "num_instances": len(instances),
                "foreground_area_ratio": float(np.count_nonzero(foreground)) / float(height * width) if height * width else 0.0,
                "message": "",
            },
        }

    def _load_model(self, weights: str) -> None:
        if not Path(weights).exists():
            self.message = f"YOLO-seg weights not found: {weights}"
            return
        try:
            from ultralytics import YOLO  # type: ignore
        except Exception:
            self.message = "ultralytics is not installed; YOLO-seg backend is unavailable."
            return
        try:
            self.model = YOLO(weights)
            self.message = ""
        except Exception as exc:
            self.message = f"Failed to load YOLO-seg weights: {exc}"


def _empty_output(image, empty_mask, message: str) -> dict:
    height, width = empty_mask.shape[:2]
    return {
        "instances": [],
        "debug_results": {
            "original": image,
            "illumination_corrected": image,
            "initial_mask": empty_mask,
            "cleaned_mask": empty_mask,
            "watershed_markers": np.zeros((height, width), dtype=np.int32),
            "overlay": image,
        },
        "stats": {"num_instances": 0, "foreground_area_ratio": 0.0, "message": message},
    }


def _instances_from_results(results, image_shape: tuple[int, int]) -> list[GrainInstance]:
    if not results:
        return []
    first = results[0]
    masks = getattr(first, "masks", None)
    boxes = getattr(first, "boxes", None)
    if masks is None or getattr(masks, "data", None) is None:
        return []
    mask_data = masks.data
    try:
        mask_array = mask_data.detach().cpu().numpy()
    except AttributeError:
        mask_array = np.asarray(mask_data)
    confidences = _confidences_from_boxes(boxes, len(mask_array))
    instances = []
    height, width = image_shape
    for index, mask in enumerate(mask_array, start=1):
        binary = np.where(mask > 0.5, 255, 0).astype(np.uint8)
        if binary.shape != (height, width):
            binary = cv2.resize(binary, (width, height), interpolation=cv2.INTER_NEAREST)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        area = float(np.count_nonzero(binary))
        if area <= 0:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        moments = cv2.moments(contour)
        if moments["m00"]:
            center_x = float(moments["m10"] / moments["m00"])
            center_y = float(moments["m01"] / moments["m00"])
        else:
            center_x = float(x + w / 2)
            center_y = float(y + h / 2)
        instances.append(
            GrainInstance(
                id=len(instances) + 1,
                contour=contour,
                bbox=(int(x), int(y), int(w), int(h)),
                mask=binary,
                area_px=area,
                center_x=center_x,
                center_y=center_y,
                classification="rice_grain",
                predicted_class="rice_grain",
                final_class="rice_grain",
                status="valid",
                confidence=float(confidences[index - 1]),
            )
        )
    return instances


def _confidences_from_boxes(boxes, count: int) -> list[float]:
    if boxes is None or getattr(boxes, "conf", None) is None:
        return [0.0] * count
    try:
        values = boxes.conf.detach().cpu().numpy().tolist()
    except AttributeError:
        values = list(np.asarray(boxes.conf).reshape(-1))
    values = [float(value) for value in values[:count]]
    return values + [0.0] * max(0, count - len(values))


def _merge_instance_masks(instances: list[GrainInstance], image_shape: tuple[int, int]) -> np.ndarray:
    merged = np.zeros(image_shape, dtype=np.uint8)
    for instance in instances:
        if instance.mask.shape == image_shape:
            merged[instance.mask > 0] = 255
    return merged

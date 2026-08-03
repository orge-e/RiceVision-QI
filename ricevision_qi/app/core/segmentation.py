from dataclasses import dataclass, field

import cv2
import numpy as np

from ricevision_qi.app.core.config import merge_config


@dataclass
class GrainInstance:
    id: int
    contour: np.ndarray
    bbox: tuple[int, int, int, int]
    mask: np.ndarray
    area_px: float
    center_x: float
    center_y: float
    features: dict = field(default_factory=dict)
    classification: str = "unknown"
    predicted_class: str = "unknown"
    manual_class: str | None = None
    final_class: str = "unknown"
    status: str = "valid"
    confidence: float = 0.0
    metadata: dict = field(default_factory=dict)


def segment_grains(image_rgb, binary_mask, config=None):
    cfg = merge_config(config)
    if binary_mask is None or binary_mask.size == 0:
        return []

    mask = (binary_mask > 0).astype(np.uint8) * 255
    if np.count_nonzero(mask) == 0:
        return []

    if cfg.get("watershed_enabled", True):
        instances, _ = watershed_segment(image_rgb, mask, cfg)
        if instances:
            return _renumber(filter_grain_instances(instances, cfg))

    return _renumber(filter_grain_instances(connected_components_segment(mask, cfg), cfg))


def connected_components_segment(mask, config=None):
    cfg = merge_config(config)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    instances = []
    for label in range(1, num_labels):
        area = float(stats[label, cv2.CC_STAT_AREA])
        component_mask = (labels == label).astype(np.uint8) * 255
        contour = _largest_contour(component_mask)
        if contour is None:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        center_x, center_y = centroids[label]
        instances.append(
            GrainInstance(
                id=len(instances) + 1,
                contour=contour,
                bbox=(int(x), int(y), int(w), int(h)),
                mask=component_mask,
                area_px=area,
                center_x=float(center_x),
                center_y=float(center_y),
            )
        )
    return instances


def watershed_segment(image_rgb, mask, config=None):
    cfg = merge_config(config)
    empty_markers = np.zeros(mask.shape, dtype=np.int32)
    try:
        kernel = np.ones((3, 3), np.uint8)
        sure_bg = cv2.dilate(mask, kernel, iterations=2)
        distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        if distance.max() <= 0:
            return [], empty_markers
        ratio = float(cfg.get("watershed_dist_ratio", 0.35))
        local_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        local_max = distance == cv2.dilate(distance, local_kernel)
        sure_fg = np.where(local_max & (distance >= ratio * distance.max()), 255, 0).astype(np.uint8)
        sure_fg = cv2.dilate(sure_fg, np.ones((3, 3), np.uint8), iterations=1)
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        unknown = cv2.subtract(sure_bg, sure_fg)
        markers[unknown == 255] = 0
        markers = cv2.watershed(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR), markers)

        instances = []
        for marker in sorted(set(np.unique(markers))):
            if marker <= 1:
                continue
            component_mask = (markers == marker).astype(np.uint8) * 255
            area = float(np.count_nonzero(component_mask))
            contour = _largest_contour(component_mask)
            if contour is None:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            moments = cv2.moments(contour)
            if moments["m00"]:
                center_x = moments["m10"] / moments["m00"]
                center_y = moments["m01"] / moments["m00"]
            else:
                center_x = x + w / 2
                center_y = y + h / 2
            instances.append(
                GrainInstance(
                    id=len(instances) + 1,
                    contour=contour,
                    bbox=(int(x), int(y), int(w), int(h)),
                    mask=component_mask,
                    area_px=area,
                    center_x=float(center_x),
                    center_y=float(center_y),
                )
            )
        return instances, markers.astype(np.int32)
    except cv2.error:
        return [], empty_markers


def filter_grain_instances(instances, config=None):
    cfg = merge_config(config)
    filtered = []
    max_area = float(cfg.get("max_area", cfg.get("max_grain_area", 100000)))
    min_area = float(cfg.get("min_area", cfg.get("min_grain_area", 100)))
    for instance in instances:
        area = float(instance.area_px)
        if area < min_area or area > max_area:
            continue
        rect = cv2.minAreaRect(instance.contour)
        width, height = rect[1]
        short = max(1.0, min(width, height))
        long = max(width, height)
        aspect_ratio = long / short
        if aspect_ratio < 1.05 or aspect_ratio > 12.0:
            continue
        filtered.append(instance)
    return _renumber(filtered)


def _largest_contour(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def _area_allowed(area, cfg):
    return float(cfg.get("min_area", cfg["min_grain_area"])) <= area <= float(
        cfg.get("max_area", cfg["max_grain_area"])
    )


def _renumber(instances):
    instances = sorted(instances, key=lambda grain: (grain.center_y, grain.center_x))
    for index, instance in enumerate(instances, start=1):
        instance.id = index
    return instances

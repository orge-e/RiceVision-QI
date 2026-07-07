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
    confidence: float = 0.0


def segment_grains(image_rgb, binary_mask, config=None):
    cfg = merge_config(config)
    if binary_mask is None or binary_mask.size == 0:
        return []

    mask = (binary_mask > 0).astype(np.uint8) * 255
    if np.count_nonzero(mask) == 0:
        return []

    if cfg.get("watershed_enabled", True):
        instances = _watershed_instances(image_rgb, mask, cfg)
        if instances:
            return _renumber(instances)

    return _renumber(_connected_component_instances(mask, cfg))


def _connected_component_instances(mask, cfg):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    instances = []
    for label in range(1, num_labels):
        area = float(stats[label, cv2.CC_STAT_AREA])
        if not _area_allowed(area, cfg):
            continue
        component_mask = np.where(labels == label, 255, 0).astype(np.uint8)
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


def _watershed_instances(image_rgb, mask, cfg):
    try:
        kernel = np.ones((3, 3), np.uint8)
        sure_bg = cv2.dilate(mask, kernel, iterations=2)
        distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        if distance.max() <= 0:
            return []
        _, sure_fg = cv2.threshold(distance, 0.42 * distance.max(), 255, 0)
        sure_fg = sure_fg.astype(np.uint8)
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
            if not _area_allowed(area, cfg):
                continue
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
        return instances
    except cv2.error:
        return []


def _largest_contour(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def _area_allowed(area, cfg):
    return float(cfg["min_grain_area"]) <= area <= float(cfg["max_grain_area"])


def _renumber(instances):
    instances = sorted(instances, key=lambda grain: (grain.center_y, grain.center_x))
    for index, instance in enumerate(instances, start=1):
        instance.id = index
    return instances

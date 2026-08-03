import cv2
import numpy as np


def remove_border_artifacts(instances, image_shape, config):
    height, width = image_shape[:2]
    margin = int(config.get("border_margin_px", 2) if config else 2)
    for instance in instances:
        x, y, w, h = instance.bbox
        if x <= margin or y <= margin or x + w >= width - margin or y + h >= height - margin:
            instance.status = "border_artifact"
    return instances


def remove_abnormal_shapes(instances, config):
    cfg = config or {}
    min_area = float(cfg.get("min_area", cfg.get("min_grain_area", 100)))
    max_area = float(cfg.get("max_area", cfg.get("max_grain_area", 100000)))
    min_aspect = float(cfg.get("min_aspect_ratio", 1.1))
    max_aspect = float(cfg.get("max_aspect_ratio", 12.0))
    for instance in instances:
        area = float(instance.area_px)
        if area < min_area:
            instance.status = "filtered_small"
            continue
        if area > max_area:
            instance.status = "filtered_large"
            continue
        rect = cv2.minAreaRect(instance.contour)
        side_a, side_b = rect[1]
        short = max(1.0, min(side_a, side_b))
        aspect_ratio = max(side_a, side_b) / short
        if aspect_ratio < min_aspect and instance.status == "valid":
            instance.status = "possible_split"
        elif aspect_ratio > max_aspect and instance.status == "valid":
            instance.status = "possible_merged"
    return instances


def merge_tiny_fragments(instances, config):
    cfg = config or {}
    min_area = float(cfg.get("min_area", cfg.get("min_grain_area", 100)))
    for instance in instances:
        if instance.status == "valid" and instance.area_px < min_area * 0.65:
            instance.status = "possible_split"
    return instances


def flag_merge_suspects(instances, config):
    cfg = config or {}
    valid_areas = np.array([item.area_px for item in instances if item.area_px > 0], dtype=float)
    if valid_areas.size == 0:
        return instances
    median_area = float(np.median(valid_areas))
    ratio = float(cfg.get("merge_area_ratio", 3.0))
    for instance in instances:
        if instance.status == "valid" and instance.area_px > median_area * ratio:
            instance.status = "possible_merged"
    return instances


def flag_split_suspects(instances, config):
    cfg = config or {}
    valid_areas = np.array([item.area_px for item in instances if item.area_px > 0], dtype=float)
    if valid_areas.size == 0:
        return instances
    median_area = float(np.median(valid_areas))
    ratio = float(cfg.get("split_area_ratio", 0.35))
    for instance in instances:
        if instance.status == "valid" and instance.area_px < median_area * ratio:
            instance.status = "possible_split"
    return instances


def postprocess_instances(instances, image_shape, config):
    remove_border_artifacts(instances, image_shape, config)
    remove_abnormal_shapes(instances, config)
    merge_tiny_fragments(instances, config)
    flag_merge_suspects(instances, config)
    flag_split_suspects(instances, config)
    return instances

from dataclasses import dataclass

import numpy as np

from ricevision_qi.app.core.classification_rules import classify_grain
from ricevision_qi.app.core.config import merge_config
from ricevision_qi.app.core.measurement import measure_grain
from ricevision_qi.app.core.preprocessing import preprocess_image
from ricevision_qi.app.core.segmentation import segment_grains
from ricevision_qi.app.core.visualization import draw_detection_overlay


@dataclass
class DetectionResult:
    original_image: np.ndarray
    overlay_image: np.ndarray
    grains: list
    summary: dict


def run_detection(image_rgb, config=None):
    cfg = merge_config(config)
    if image_rgb is None or image_rgb.size == 0:
        empty = np.zeros((0, 0, 3), dtype=np.uint8)
        return DetectionResult(
            original_image=empty,
            overlay_image=empty,
            grains=[],
            summary=_summary([]),
        )

    binary_mask, _, _ = preprocess_image(image_rgb, cfg)
    grains = segment_grains(image_rgb, binary_mask, cfg)
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

    overlay = draw_detection_overlay(image_rgb, grains)
    return DetectionResult(
        original_image=image_rgb,
        overlay_image=overlay,
        grains=grains,
        summary=_summary(grains),
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
        key = f"{grain.classification}_count"
        if key in counts:
            counts[key] += 1
        else:
            counts["unknown_count"] += 1
    return counts

from ricevision_qi.app.core.config import merge_config


def classify_grain(features, reference_stats=None, config=None):
    cfg = merge_config(config)
    reference_stats = reference_stats or {}
    mean_length = float(reference_stats.get("mean_length_px") or features.get("length_px") or 0)
    mean_area = float(reference_stats.get("mean_area_px") or features.get("area_px") or 0)
    length = float(features.get("length_px", 0))
    area = float(features.get("area_px", 0))
    aspect_ratio = float(features.get("aspect_ratio", 0))
    mean_lab = features.get("mean_lab", (0, 128, 128))

    if area <= 0 or length <= 0:
        return "unknown", 0.2
    if mean_area and area < mean_area * float(cfg["impurity_area_ratio"]):
        return "impurity", 0.72
    if aspect_ratio < 1.25:
        return "impurity", 0.62
    if mean_length and length < mean_length * float(cfg["broken_length_ratio"]):
        return "broken", 0.76

    lightness, lab_a, lab_b = [float(v) for v in mean_lab]
    color_score = abs(lab_a - 128) + max(0.0, lab_b - 150) + max(0.0, 105 - lightness)
    if color_score > float(cfg["defective_color_threshold"]):
        return "defective", 0.66

    return "normal", 0.84

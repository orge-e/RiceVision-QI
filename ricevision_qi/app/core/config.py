from pathlib import Path


DEFAULT_CONFIG = {
    "profile_name": "default",
    "segmentation_backend": "opencv_watershed",
    "min_grain_area": 120,
    "max_grain_area": 20000,
    "morph_kernel_size": 5,
    "segmentation_method": "lab_threshold",
    "kernel_size": 3,
    "open_iter": 1,
    "close_iter": 0,
    "min_area": 120,
    "max_area": 100000,
    "watershed_dist_ratio": 0.20,
    "fill_holes": False,
    "max_hole_area": 3000,
    "hsv_s_min": 0,
    "hsv_s_max": 70,
    "hsv_v_min": 120,
    "lab_l_min": 120,
    "lab_b_min": 140,
    "lab_a_min": 120,
    "lab_a_max": 145,
    "watershed_enabled": True,
    "broken_length_ratio": 0.75,
    "impurity_area_ratio": 0.35,
    "defective_color_threshold": 34,
    "pixel_per_mm": None,
}


PROFILE_DISPLAY_NAMES = {
    "clean_background": "Clean Background",
    "plastic_bag_rice": "Plastic Bag Rice",
    "dense_rice_cluster": "Dense Rice Cluster",
    "debug_fast": "Debug Fast",
    "yolo_seg_rice": "YOLO Seg Rice",
    "yolo_seg_rice_s": "YOLO Seg Rice S",
    "yolo_seg_rice_v8s": "YOLOv8s Seg Rice (Recommended)",
    "yolo_seg_rice_v8m": "YOLOv8m Seg Rice",
}


def load_default_config() -> dict:
    config = DEFAULT_CONFIG.copy()
    config_path = Path(__file__).resolve().parents[1] / "config" / "default.yaml"
    if not config_path.exists():
        return config

    _merge_yaml_sections(config, _parse_simple_yaml(config_path))
    return config


def load_profile_config(profile_name: str | None) -> dict:
    base = load_default_config()
    requested = (profile_name or "default").strip()
    if not requested or requested.lower() in {"default", "custom"}:
        base["profile_name"] = "default"
        return base

    profile_path = Path(__file__).resolve().parents[1] / "config" / "profiles" / f"{requested}.yaml"
    if not profile_path.exists():
        base["profile_name"] = "default"
        base["profile_warning"] = f"Profile not found: {requested}. Falling back to default."
        return base

    _merge_yaml_sections(base, _parse_simple_yaml(profile_path))
    base["profile_name"] = requested
    return base


def merge_config(config: dict | None = None) -> dict:
    profile_name = config.get("profile") if config else None
    merged = load_profile_config(profile_name) if profile_name else load_default_config()
    if config:
        overrides = config.copy()
        overrides.pop("profile", None)
        merged.update(overrides)
    return merged


def _merge_yaml_sections(config: dict, parsed: dict) -> None:
    for section, values in parsed.items():
        if not isinstance(values, dict):
            continue
        for key, value in values.items():
            if section == "segmentation" and key == "backend":
                config["segmentation_backend"] = value
            elif section == "segmentation" and key == "method":
                config["method"] = value
                config["segmentation_method"] = value
            else:
                config[key] = value


def _parse_simple_yaml(path: Path) -> dict:
    parsed: dict[str, dict] = {}
    current_section = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.endswith(":") and not line.startswith("-"):
            current_section = line[:-1]
            parsed.setdefault(current_section, {})
            continue
        if current_section is None or ":" not in line:
            continue
        key, value = line.split(":", 1)
        parsed[current_section][key.strip()] = _coerce_value(value.strip().strip('"'))
    return parsed


def _coerce_value(value: str):
    if value.lower() in {"none", "null", ""}:
        return None
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        return float(value) if "." in value else int(value)
    except ValueError:
        return value

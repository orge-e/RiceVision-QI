from pathlib import Path


DEFAULT_CONFIG = {
    "min_grain_area": 120,
    "max_grain_area": 20000,
    "morph_kernel_size": 5,
    "watershed_enabled": True,
    "broken_length_ratio": 0.75,
    "impurity_area_ratio": 0.35,
    "defective_color_threshold": 34,
    "pixel_per_mm": None,
}


def load_default_config() -> dict:
    config = DEFAULT_CONFIG.copy()
    config_path = Path(__file__).resolve().parents[1] / "config" / "default.yaml"
    if not config_path.exists():
        return config

    current_section = None
    for raw_line in config_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.endswith(":") and not line.startswith("-"):
            current_section = line[:-1]
            continue
        if current_section != "detection" or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip().strip('"')
        if value.lower() in {"none", "null", ""}:
            config[key] = None
        elif value.lower() in {"true", "false"}:
            config[key] = value.lower() == "true"
        else:
            try:
                config[key] = float(value) if "." in value else int(value)
            except ValueError:
                config[key] = value
    return config


def merge_config(config: dict | None = None) -> dict:
    merged = load_default_config()
    if config:
        merged.update(config)
    return merged

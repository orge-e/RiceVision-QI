import cv2
import numpy as np

from ricevision_qi.app.core.config import merge_config


def normalize_illumination(image):
    if image is None or image.size == 0:
        return image

    image_rgb = np.ascontiguousarray(image.astype(np.uint8))
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    background = cv2.GaussianBlur(l_channel, (0, 0), sigmaX=35, sigmaY=35)
    corrected_l = cv2.divide(l_channel, background, scale=180)
    corrected_l = cv2.normalize(corrected_l, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    corrected_l = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8)).apply(corrected_l)
    corrected_lab = cv2.merge((corrected_l, a_channel, b_channel))
    return cv2.cvtColor(corrected_lab, cv2.COLOR_LAB2RGB)


def convert_color_spaces(image):
    image_rgb = np.ascontiguousarray(image.astype(np.uint8))
    return {
        "rgb": image_rgb,
        "gray": cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY),
        "hsv": cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV),
        "lab": cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB),
    }


def build_rice_mask(image, config):
    cfg = merge_config(config)
    if image is None or image.size == 0:
        return np.zeros((0, 0), dtype=np.uint8)

    spaces = convert_color_spaces(image)
    method = cfg.get("segmentation_method", cfg.get("method", "lab_threshold"))
    if method == "gray_otsu":
        return _gray_otsu_mask(spaces["gray"])
    if method == "adaptive_threshold":
        return _guard_foreground_ratio(_adaptive_threshold_mask(spaces["gray"]), spaces["gray"])
    if method == "hsv_threshold":
        return _guard_foreground_ratio(_hsv_threshold_mask(spaces["hsv"], cfg), spaces["gray"])
    if method == "lab_threshold":
        return _guard_foreground_ratio(_lab_threshold_mask(spaces["lab"], spaces["hsv"], cfg), spaces["gray"])
    return _guard_foreground_ratio(_lab_threshold_mask(spaces["lab"], spaces["hsv"], cfg), spaces["gray"])


def clean_mask(mask, config):
    cfg = merge_config(config)
    if mask is None or mask.size == 0:
        return np.zeros((0, 0), dtype=np.uint8)

    cleaned = (mask > 0).astype(np.uint8) * 255
    kernel_size = max(1, int(cfg.get("kernel_size", cfg.get("morph_kernel_size", 3))))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    open_iter = max(0, int(cfg.get("open_iter", 1)))
    close_iter = max(0, int(cfg.get("close_iter", 2)))
    if open_iter:
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=open_iter)
    if close_iter:
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=close_iter)
    if bool(cfg.get("fill_holes", True)):
        cleaned = _fill_holes(cleaned, int(cfg.get("max_hole_area", 3000)))
    cleaned = _remove_small_components(cleaned, int(cfg.get("min_area", cfg.get("min_grain_area", 100))))
    return cleaned.astype(np.uint8)


def preprocess_image(image_rgb, config=None):
    cfg = merge_config(config)
    if image_rgb is None or image_rgb.size == 0:
        empty = np.zeros((0, 0), dtype=np.uint8)
        return empty, image_rgb, {"foreground_pixels": 0, "method": "empty"}

    corrected = normalize_illumination(image_rgb)
    initial_mask = build_rice_mask(corrected, cfg)
    cleaned_mask = clean_mask(initial_mask, cfg)
    debug_info = {
        "foreground_pixels": int(np.count_nonzero(cleaned_mask)),
        "method": cfg.get("segmentation_method", "lab_threshold"),
        "kernel_size": int(cfg.get("kernel_size", 3)),
        "initial_mask": initial_mask,
        "cleaned_mask": cleaned_mask,
        "illumination_corrected": corrected,
    }
    return cleaned_mask.astype(np.uint8), corrected, debug_info


def _gray_otsu_mask(gray):
    if float(np.std(gray)) < 2.0:
        return np.zeros_like(gray, dtype=np.uint8)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, inverted = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binary_ratio = np.count_nonzero(binary) / binary.size
    inverted_ratio = np.count_nonzero(inverted) / inverted.size
    if 0.005 < inverted_ratio < binary_ratio:
        return inverted.astype(np.uint8)
    return binary.astype(np.uint8)


def _adaptive_threshold_mask(gray):
    if gray is None or gray.size == 0:
        return np.zeros((0, 0), dtype=np.uint8)
    block_size = max(15, (min(gray.shape[:2]) // 12) | 1)
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        -3,
    )
    inverted = cv2.bitwise_not(binary)
    binary_ratio = np.count_nonzero(binary) / binary.size
    inverted_ratio = np.count_nonzero(inverted) / inverted.size
    if 0.005 < inverted_ratio < binary_ratio:
        return inverted.astype(np.uint8)
    return binary.astype(np.uint8)


def _guard_foreground_ratio(mask, gray):
    ratio = np.count_nonzero(mask) / mask.size if mask.size else 0
    if ratio > 0.60 or ratio < 0.001:
        return _gray_otsu_mask(gray)
    return mask.astype(np.uint8)


def _hsv_threshold_mask(hsv, cfg):
    lower = np.array([0, int(cfg.get("hsv_s_min", 0)), int(cfg.get("hsv_v_min", 120))], dtype=np.uint8)
    upper = np.array([179, int(cfg.get("hsv_s_max", 120)), 255], dtype=np.uint8)
    return cv2.inRange(hsv, lower, upper)


def _lab_threshold_mask(lab, hsv, cfg):
    l_channel, a_channel, b_channel = cv2.split(lab)
    saturation = hsv[:, :, 1]
    l_min = int(cfg.get("lab_l_min", 115))
    b_min = int(cfg.get("lab_b_min", 126))
    a_min = int(cfg.get("lab_a_min", 120))
    a_max = int(cfg.get("lab_a_max", 145))
    saturation_max = int(cfg.get("hsv_s_max", 120))
    color_gate = (
        (a_channel >= a_min)
        & (a_channel <= a_max)
        & (b_channel >= b_min)
        & (saturation <= saturation_max)
    )
    base = (l_channel >= l_min) & color_gate
    bright_binary, _ = _bright_grain_mask(l_channel, b_channel, saturation, cfg)
    bright_binary = cv2.bitwise_and(bright_binary, np.where(color_gate, 255, 0).astype(np.uint8))
    return np.where(base | (bright_binary > 0), 255, 0).astype(np.uint8)


def _fill_holes(mask, max_hole_area):
    inverse = cv2.bitwise_not(mask)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inverse, 8)
    filled = mask.copy()
    height, width = mask.shape
    for label in range(1, num_labels):
        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]
        w = stats[label, cv2.CC_STAT_WIDTH]
        h = stats[label, cv2.CC_STAT_HEIGHT]
        area = stats[label, cv2.CC_STAT_AREA]
        touches_border = x == 0 or y == 0 or x + w >= width or y + h >= height
        if not touches_border and area <= max_hole_area:
            filled[labels == label] = 255
    return filled


def _remove_small_components(mask, min_area):
    if min_area <= 0:
        return mask
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    filtered = np.zeros_like(mask)
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            filtered[labels == label] = 255
    return filtered


def _bright_grain_mask(l_channel, b_channel, saturation, cfg):
    otsu_l, _ = cv2.threshold(l_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bright_threshold = max(float(otsu_l) + 35.0, float(np.percentile(l_channel, 72)))
    warm_threshold = max(130.0, float(np.percentile(b_channel, 58)))
    low_saturation = np.percentile(saturation, 92) + 20

    local_kernel_size = int(cfg.get("local_bright_kernel_size", 41))
    local_kernel_size = max(15, local_kernel_size)
    if local_kernel_size % 2 == 0:
        local_kernel_size += 1
    local_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (local_kernel_size, local_kernel_size),
    )
    top_hat = cv2.morphologyEx(l_channel, cv2.MORPH_TOPHAT, local_kernel)
    local_threshold = max(18.0, float(np.percentile(top_hat, 88)))

    bright = (
        (l_channel >= bright_threshold)
        & (b_channel >= warm_threshold)
        & (saturation <= low_saturation)
    )
    local_bright = (
        (top_hat >= local_threshold)
        & (l_channel >= bright_threshold - 35.0)
        & (b_channel >= warm_threshold - 3.0)
    )
    mask = np.where(bright | local_bright, 255, 0).astype(np.uint8)

    small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, small_kernel)
    return mask, {
        "bright_threshold": float(bright_threshold),
        "warm_threshold": float(warm_threshold),
        "local_threshold": float(local_threshold),
    }

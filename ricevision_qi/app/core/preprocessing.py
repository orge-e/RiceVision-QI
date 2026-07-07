import cv2
import numpy as np

from ricevision_qi.app.core.config import merge_config


def preprocess_image(image_rgb, config=None):
    cfg = merge_config(config)
    if image_rgb is None or image_rgb.size == 0:
        empty = np.zeros((0, 0), dtype=np.uint8)
        return empty, image_rgb, {"foreground_pixels": 0, "method": "empty"}

    image_rgb = np.ascontiguousarray(image_rgb.astype(np.uint8))
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    normalized_l = clahe.apply(l_channel)
    corrected_lab = cv2.merge((normalized_l, a_channel, b_channel))
    preprocessed = cv2.cvtColor(corrected_lab, cv2.COLOR_LAB2RGB)

    background_lab = np.median(lab.reshape(-1, 3), axis=0).astype(np.float32)
    delta = np.linalg.norm(lab.astype(np.float32) - background_lab, axis=2)
    delta_u8 = cv2.normalize(delta, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    if int(delta_u8.max()) == 0:
        binary = np.zeros(delta_u8.shape, dtype=np.uint8)
        threshold_value = 0
    else:
        threshold_value, binary = cv2.threshold(
            delta_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        if np.count_nonzero(binary) > binary.size * 0.65:
            binary = cv2.adaptiveThreshold(
                delta_u8,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                35,
                2,
            )

    bright_binary, bright_debug = _bright_grain_mask(l_channel, b_channel, saturation, cfg)
    if np.count_nonzero(binary) > binary.size * 0.20:
        binary = bright_binary
    else:
        binary = cv2.bitwise_or(binary, bright_binary)

    kernel_size = max(3, int(cfg["morph_kernel_size"]))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    debug_info = {
        "foreground_pixels": int(np.count_nonzero(binary)),
        "method": "lab_delta_otsu",
        "threshold": float(threshold_value),
        "kernel_size": kernel_size,
        **bright_debug,
    }
    return binary.astype(np.uint8), preprocessed, debug_info


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

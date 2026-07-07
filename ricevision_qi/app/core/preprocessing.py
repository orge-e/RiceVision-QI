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
    }
    return binary.astype(np.uint8), preprocessed, debug_info

import cv2
import numpy as np


def measure_grain(image_rgb, grain_instance, pixel_per_mm=None):
    contour = grain_instance.contour
    rect = cv2.minAreaRect(contour)
    side_a, side_b = rect[1]
    length_px = float(max(side_a, side_b))
    width_px = float(min(side_a, side_b))
    area_px = float(max(grain_instance.area_px, cv2.contourArea(contour)))
    perimeter_px = float(cv2.arcLength(contour, True))
    aspect_ratio = float(length_px / width_px) if width_px > 0 else 0.0

    x, y, w, h = cv2.boundingRect(contour)
    x = max(0, x)
    y = max(0, y)
    w = min(w, image_rgb.shape[1] - x)
    h = min(h, image_rgb.shape[0] - y)
    if w > 0 and h > 0:
        roi = image_rgb[y : y + h, x : x + w]
        local_contour = contour - np.array([[[x, y]]], dtype=contour.dtype)
        local_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(local_mask, [local_contour], -1, 255, -1)
        mask_bool = local_mask > 0
    else:
        roi = image_rgb[:0, :0]
        mask_bool = np.zeros((0, 0), dtype=bool)

    if np.any(mask_bool):
        pixels_rgb = roi[mask_bool]
        mean_rgb = tuple(float(v) for v in pixels_rgb.mean(axis=0))
        lab_roi = cv2.cvtColor(roi, cv2.COLOR_RGB2LAB)
        mean_lab = tuple(float(v) for v in lab_roi[mask_bool].mean(axis=0))
    else:
        mean_rgb = (0.0, 0.0, 0.0)
        mean_lab = (0.0, 0.0, 0.0)

    features = {
        "length_px": length_px,
        "width_px": width_px,
        "area_px": area_px,
        "aspect_ratio": aspect_ratio,
        "perimeter_px": perimeter_px,
        "mean_rgb": mean_rgb,
        "mean_lab": mean_lab,
    }
    if pixel_per_mm:
        ppm = float(pixel_per_mm)
        features["length_mm"] = length_px / ppm
        features["width_mm"] = width_px / ppm
        features["area_mm2"] = area_px / (ppm * ppm)
    return features

import cv2
import numpy as np


CLASS_COLORS_RGB = {
    "normal": (34, 197, 94),
    "broken": (245, 158, 11),
    "defective": (239, 68, 68),
    "impurity": (59, 130, 246),
    "unknown": (107, 114, 128),
}


def draw_detection_overlay(image_rgb, grains):
    overlay = np.ascontiguousarray(image_rgb.copy())
    for grain in grains:
        color_rgb = CLASS_COLORS_RGB.get(grain.classification, CLASS_COLORS_RGB["unknown"])
        color_bgr = tuple(reversed(color_rgb))
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        cv2.drawContours(overlay_bgr, [grain.contour], -1, color_bgr, 2)
        cv2.putText(
            overlay_bgr,
            str(grain.id),
            (int(grain.center_x) - 8, int(grain.center_y) + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color_bgr,
            1,
            cv2.LINE_AA,
        )
        overlay = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
    return overlay

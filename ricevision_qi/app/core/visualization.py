import cv2
import numpy as np


CLASS_COLORS_RGB = {
    "normal": (34, 197, 94),
    "broken": (245, 158, 11),
    "defective": (239, 68, 68),
    "impurity": (59, 130, 246),
    "unknown": (107, 114, 128),
}


def draw_detection_overlay(image_rgb, grains, selected_grain_id=None):
    overlay = np.ascontiguousarray(image_rgb.copy())
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    for grain in grains:
        color_rgb = CLASS_COLORS_RGB.get(grain.classification, CLASS_COLORS_RGB["unknown"])
        color_bgr = tuple(reversed(color_rgb))
        is_selected = grain.id == selected_grain_id
        if is_selected:
            cv2.drawContours(overlay_bgr, [grain.contour], -1, (255, 255, 255), 7)
            cv2.drawMarker(
                overlay_bgr,
                (int(grain.center_x), int(grain.center_y)),
                (255, 255, 255),
                markerType=cv2.MARKER_CROSS,
                markerSize=28,
                thickness=3,
                line_type=cv2.LINE_AA,
            )
        cv2.drawContours(overlay_bgr, [grain.contour], -1, color_bgr, 4 if is_selected else 2)
        cv2.putText(
            overlay_bgr,
            str(grain.id),
            (int(grain.center_x) - 8, int(grain.center_y) + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68 if is_selected else 0.48,
            color_bgr,
            2 if is_selected else 1,
            cv2.LINE_AA,
        )
    return cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

import cv2
import numpy as np


def make_blank_image(width: int = 240, height: int = 160) -> np.ndarray:
    return np.full((height, width, 3), 255, dtype=np.uint8)


def make_single_grain_image(width: int = 240, height: int = 160) -> np.ndarray:
    image = make_blank_image(width, height)
    cv2.ellipse(image, (120, 80), (42, 14), 18, 0, 360, (210, 190, 145), -1)
    return image


def make_multiple_grain_image(width: int = 320, height: int = 220) -> np.ndarray:
    image = make_blank_image(width, height)
    cv2.ellipse(image, (82, 70), (38, 13), -12, 0, 360, (214, 196, 150), -1)
    cv2.ellipse(image, (188, 84), (40, 14), 20, 0, 360, (206, 185, 135), -1)
    cv2.ellipse(image, (130, 156), (24, 10), 8, 0, 360, (208, 191, 150), -1)
    cv2.circle(image, (252, 156), 8, (70, 70, 65), -1)
    return image


def make_bagged_grain_image(width: int = 520, height: int = 420) -> np.ndarray:
    image = np.full((height, width, 3), 28, dtype=np.uint8)
    cv2.rectangle(image, (70, 60), (450, 390), (95, 104, 110), -1)
    cv2.rectangle(image, (70, 60), (450, 390), (170, 178, 184), 2)
    cv2.line(image, (82, 96), (438, 90), (185, 190, 194), 2)
    cv2.line(image, (88, 120), (432, 112), (145, 150, 154), 1)
    cv2.circle(image, (420, 350), 18, (235, 238, 240), -1)

    centers = [
        (150, 210), (188, 206), (226, 214), (264, 208), (302, 218),
        (168, 250), (206, 248), (244, 256), (282, 250), (320, 262),
        (148, 298), (186, 292), (224, 304), (262, 296), (300, 306),
    ]
    angles = [-18, 12, 28, -35, 8, 35, -8, 16, -24, 30, 5, -30, 24, -12, 18]
    for center, angle in zip(centers, angles):
        cv2.ellipse(image, center, (24, 8), angle, 0, 360, (224, 213, 178), -1)
        cv2.ellipse(image, center, (24, 8), angle, 0, 360, (242, 236, 208), 1)
    return image

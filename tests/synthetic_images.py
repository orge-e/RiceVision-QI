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

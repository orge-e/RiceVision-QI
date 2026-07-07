from pathlib import Path

import cv2
import numpy as np
from PySide6.QtGui import QImage, QPixmap


SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


class ImageReadError(RuntimeError):
    pass


class UnsupportedImageFormatError(ValueError):
    pass


def is_supported_image_path(path: Path) -> bool:
    return bool(path.name) and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS


def read_image(path: Path) -> np.ndarray:
    if not path or not path.name:
        raise ImageReadError("Image path is empty.")
    if not is_supported_image_path(path):
        supported = ", ".join(sorted(SUPPORTED_IMAGE_EXTENSIONS))
        raise UnsupportedImageFormatError(f"Unsupported image format. Supported formats: {supported}")
    if not path.exists():
        raise ImageReadError(f"Image file does not exist: {path}")

    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ImageReadError(f"Failed to read image: {path}")
    return image


def load_image_pixmap(path: Path) -> QPixmap:
    image_bgr = read_image(path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    height, width, channels = image_rgb.shape
    bytes_per_line = channels * width
    qimage = QImage(
        image_rgb.data,
        width,
        height,
        bytes_per_line,
        QImage.Format.Format_RGB888,
    ).copy()
    return QPixmap.fromImage(qimage)

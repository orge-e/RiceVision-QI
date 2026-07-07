from pathlib import Path

import cv2
import numpy as np
from PySide6.QtGui import QImage, QPixmap


SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


class ImageReadError(RuntimeError):
    pass


class UnsupportedImageFormatError(ValueError):
    pass


class LoadedImage:
    def __init__(self, image_rgb: np.ndarray, path: Path) -> None:
        self.image_rgb = image_rgb
        self.path = path
        self.height, self.width = image_rgb.shape[:2]
        self.shape = image_rgb.shape


def is_supported_image_path(path: Path) -> bool:
    return bool(path.name) and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS


def read_image(path: Path) -> LoadedImage:
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
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return LoadedImage(image_rgb=image_rgb, path=path)


def load_image_pixmap(path: Path) -> QPixmap:
    loaded = read_image(path)
    return pixmap_from_rgb(loaded.image_rgb)


def pixmap_from_rgb(image_rgb: np.ndarray) -> QPixmap:
    if image_rgb is None or image_rgb.size == 0:
        raise ImageReadError("Image data is empty.")

    contiguous = np.ascontiguousarray(image_rgb)
    height, width, channels = contiguous.shape
    bytes_per_line = channels * width
    qimage = QImage(
        contiguous.data,
        width,
        height,
        bytes_per_line,
        QImage.Format.Format_RGB888,
    ).copy()
    return QPixmap.fromImage(qimage)

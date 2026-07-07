import numpy as np

from ricevision_qi.app.core.preprocessing import preprocess_image
from tests.synthetic_images import make_blank_image, make_single_grain_image


def test_preprocess_blank_image_returns_empty_mask():
    image = make_blank_image()

    binary_mask, preprocessed, debug_info = preprocess_image(image)

    assert binary_mask.shape == image.shape[:2]
    assert binary_mask.dtype == np.uint8
    assert preprocessed.shape == image.shape
    assert debug_info["foreground_pixels"] == 0


def test_preprocess_simple_ellipse_creates_foreground_mask():
    image = make_single_grain_image()

    binary_mask, _, debug_info = preprocess_image(image)

    assert binary_mask.shape == image.shape[:2]
    assert np.count_nonzero(binary_mask) > 500
    assert debug_info["foreground_pixels"] == int(np.count_nonzero(binary_mask))

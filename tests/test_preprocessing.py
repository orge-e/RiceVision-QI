import numpy as np

from ricevision_qi.app.core.preprocessing import (
    build_rice_mask,
    clean_mask,
    convert_color_spaces,
    normalize_illumination,
    preprocess_image,
)
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


def test_preprocessing_steps_return_debuggable_intermediate_images():
    image = make_single_grain_image()

    corrected = normalize_illumination(image)
    spaces = convert_color_spaces(corrected)
    initial_mask = build_rice_mask(corrected, {"segmentation_method": "lab_threshold"})
    cleaned_mask = clean_mask(initial_mask, {"min_area": 50, "kernel_size": 3})

    assert corrected.shape == image.shape
    assert {"rgb", "gray", "hsv", "lab"}.issubset(spaces)
    assert initial_mask.shape == image.shape[:2]
    assert cleaned_mask.shape == image.shape[:2]
    assert np.count_nonzero(cleaned_mask) > 500


def test_build_rice_mask_supports_multiple_methods():
    image = make_single_grain_image()

    for method in ["gray_otsu", "hsv_threshold", "lab_threshold"]:
        mask = build_rice_mask(image, {"segmentation_method": method})
        assert mask.shape == image.shape[:2]
        assert mask.dtype == np.uint8

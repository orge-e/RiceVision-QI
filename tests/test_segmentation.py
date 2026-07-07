from ricevision_qi.app.core.preprocessing import preprocess_image
from ricevision_qi.app.core.segmentation import segment_grains
from tests.synthetic_images import make_blank_image, make_single_grain_image


def test_segment_blank_image_returns_no_grains():
    image = make_blank_image()
    binary_mask, _, _ = preprocess_image(image)

    grains = segment_grains(image, binary_mask)

    assert grains == []


def test_segment_simple_ellipse_detects_grain_instance():
    image = make_single_grain_image()
    binary_mask, _, _ = preprocess_image(image)

    grains = segment_grains(image, binary_mask)

    assert len(grains) >= 1
    assert grains[0].id == 1
    assert grains[0].area_px > 500
    assert grains[0].bbox[2] > 0
    assert grains[0].bbox[3] > 0

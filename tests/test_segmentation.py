from ricevision_qi.app.core.preprocessing import preprocess_image
from ricevision_qi.app.core.segmentation import (
    connected_components_segment,
    filter_grain_instances,
    segment_grains,
    watershed_segment,
)
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


def test_segmentation_exposes_connected_components_and_watershed_debug_markers():
    image = make_single_grain_image()
    binary_mask, _, _ = preprocess_image(image)

    component_grains = connected_components_segment(binary_mask, {"min_area": 50})
    watershed_grains, markers = watershed_segment(
        image,
        binary_mask,
        {"min_area": 50, "watershed_dist_ratio": 0.35},
    )
    filtered = filter_grain_instances(component_grains, {"min_area": 50, "max_area": 5000})

    assert len(component_grains) >= 1
    assert len(watershed_grains) >= 1
    assert markers.shape == binary_mask.shape
    assert len(filtered) >= 1

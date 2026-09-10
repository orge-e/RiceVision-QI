from ricevision_qi.app.core.segmentation_backends import (
    EdgeEnhancedWatershedBackend,
    OpenCVThresholdBackend,
    OpenCVWatershedBackend,
    create_segmentation_backend,
)
from tests.synthetic_images import make_multiple_grain_image


def test_backends_can_be_created_by_name():
    assert isinstance(create_segmentation_backend("opencv_threshold"), OpenCVThresholdBackend)
    assert isinstance(create_segmentation_backend("opencv_watershed"), OpenCVWatershedBackend)
    assert isinstance(create_segmentation_backend("edge_enhanced_watershed"), EdgeEnhancedWatershedBackend)


def test_backend_output_has_uniform_shape():
    backend = create_segmentation_backend("opencv_threshold")

    output = backend.segment(make_multiple_grain_image(), {"min_area": 50, "max_area": 100000})

    assert {"instances", "debug_results", "stats"}.issubset(output)
    assert isinstance(output["instances"], list)
    assert {"initial_mask", "cleaned_mask", "overlay"}.issubset(output["debug_results"])
    assert "foreground_area_ratio" in output["stats"]

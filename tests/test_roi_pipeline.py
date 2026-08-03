from ricevision_qi.app.core.pipeline import run_detection
from tests.synthetic_images import make_multiple_grain_image


def test_pipeline_roi_maps_instances_back_to_original_coordinates():
    roi = (40, 30, 220, 150)
    result = run_detection(
        make_multiple_grain_image(),
        {
            "roi": roi,
            "segmentation_backend": "opencv_watershed",
            "min_area": 30,
            "max_area": 100000,
            "max_processing_dimension": 0,
        },
    )

    assert result.roi == roi
    assert result.overlay_image.shape == result.original_image.shape
    assert result.debug_results["overlay"].shape == result.original_image.shape
    for grain in result.grains:
        x, y, width, height = grain.bbox
        assert x >= roi[0]
        assert y >= roi[1]
        assert x + width <= roi[0] + roi[2]
        assert y + height <= roi[1] + roi[3]

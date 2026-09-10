from ricevision_qi.app.core.pipeline import run_detection
from tests.synthetic_images import make_bagged_grain_image, make_blank_image, make_multiple_grain_image


def test_pipeline_blank_image_returns_zero_summary():
    result = run_detection(make_blank_image())

    assert result.summary["total_grains"] == 0
    assert result.overlay_image.shape == result.original_image.shape
    assert result.grains == []
    assert {"original", "overlay"}.issubset(result.debug_results)


def test_pipeline_detects_synthetic_grains_and_summary():
    result = run_detection(make_multiple_grain_image())

    assert result.summary["total_grains"] >= 3
    assert result.overlay_image.shape == result.original_image.shape
    assert len(result.grains) == result.summary["total_grains"]
    assert {
        "normal_count",
        "broken_count",
        "defective_count",
        "impurity_count",
        "unknown_count",
    }.issubset(result.summary)
    assert all(grain.features["length_px"] >= grain.features["width_px"] for grain in result.grains)
    assert {
        "original",
        "illumination_corrected",
        "initial_mask",
        "cleaned_mask",
        "watershed_markers",
        "overlay",
    }.issubset(result.debug_results)


def test_pipeline_detects_bagged_grains_on_dark_background():
    result = run_detection(make_bagged_grain_image())

    assert result.summary["total_grains"] >= 8
    assert result.summary["normal_count"] + result.summary["broken_count"] >= 1

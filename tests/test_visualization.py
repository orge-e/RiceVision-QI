import numpy as np

from ricevision_qi.app.core.pipeline import run_detection
from ricevision_qi.app.core.visualization import draw_detection_overlay
from tests.synthetic_images import make_multiple_grain_image


def test_overlay_highlights_selected_grain_id_differently():
    result = run_detection(make_multiple_grain_image())
    assert len(result.grains) >= 2

    first_id = result.grains[0].id
    second_id = result.grains[1].id
    first_overlay = draw_detection_overlay(result.original_image, result.grains, selected_grain_id=first_id)
    second_overlay = draw_detection_overlay(result.original_image, result.grains, selected_grain_id=second_id)

    assert first_overlay.shape == result.original_image.shape
    assert second_overlay.shape == result.original_image.shape
    assert not np.array_equal(first_overlay, second_overlay)

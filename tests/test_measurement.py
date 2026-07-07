from ricevision_qi.app.core.measurement import measure_grain
from ricevision_qi.app.core.preprocessing import preprocess_image
from ricevision_qi.app.core.segmentation import segment_grains
from tests.synthetic_images import make_single_grain_image


def test_measure_grain_returns_shape_and_color_features():
    image = make_single_grain_image()
    binary_mask, _, _ = preprocess_image(image)
    grain = segment_grains(image, binary_mask)[0]

    features = measure_grain(image, grain)

    assert features["length_px"] >= features["width_px"] > 0
    assert features["area_px"] > 500
    assert features["aspect_ratio"] > 1
    assert features["perimeter_px"] > 0
    assert len(features["mean_rgb"]) == 3
    assert len(features["mean_lab"]) == 3

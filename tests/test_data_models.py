from ricevision_qi.app.data.models import normalize_roi


def test_normalize_roi_preserves_valid_bounds():
    assert normalize_roi((10, 20, 30, 40), (100, 120)) == (10, 20, 30, 40)


def test_normalize_roi_clamps_to_image_bounds():
    assert normalize_roi((-5, 80, 30, 40), (100, 120)) == (0, 80, 25, 20)


def test_normalize_roi_rejects_empty_or_invalid_regions():
    assert normalize_roi(None, (100, 120)) is None
    assert normalize_roi((10, 20, 0, 40), (100, 120)) is None
    assert normalize_roi((200, 20, 10, 10), (100, 120)) is None
    assert normalize_roi(("bad", 20, 10, 10), (100, 120)) is None

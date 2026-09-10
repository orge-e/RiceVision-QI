from __future__ import annotations

from collections.abc import Sequence


def normalize_roi(
    roi: Sequence[int | float] | None,
    image_shape: Sequence[int],
) -> tuple[int, int, int, int] | None:
    """Return an image-bounded ``(x, y, width, height)`` ROI."""

    if roi is None:
        return None
    if len(roi) != 4 or len(image_shape) < 2:
        return None

    try:
        x, y, width, height = (int(round(float(value))) for value in roi)
        image_height, image_width = (int(image_shape[0]), int(image_shape[1]))
    except (TypeError, ValueError, OverflowError):
        return None

    if width <= 0 or height <= 0 or image_width <= 0 or image_height <= 0:
        return None

    left = max(0, min(x, image_width))
    top = max(0, min(y, image_height))
    right = max(left, min(x + width, image_width))
    bottom = max(top, min(y + height, image_height))
    if right <= left or bottom <= top:
        return None
    return left, top, right - left, bottom - top

import numpy as np

from ricevision_qi.app.core.postprocessing import (
    flag_merge_suspects,
    flag_split_suspects,
    remove_border_artifacts,
)
from ricevision_qi.app.core.segmentation import GrainInstance


def _instance(identifier, bbox, area):
    x, y, w, h = bbox
    contour = np.array([[[x, y]], [[x + w, y]], [[x + w, y + h]], [[x, y + h]]], dtype=np.int32)
    return GrainInstance(
        id=identifier,
        contour=contour,
        bbox=bbox,
        mask=np.ones((h, w), dtype=np.uint8) * 255,
        area_px=float(area),
        center_x=x + w / 2,
        center_y=y + h / 2,
    )


def test_postprocessing_marks_border_and_size_suspects():
    instances = [
        _instance(1, (0, 10, 20, 20), 400),
        _instance(2, (40, 40, 100, 80), 8000),
        _instance(3, (160, 40, 8, 8), 64),
    ]

    remove_border_artifacts(instances, (200, 200), {})
    flag_merge_suspects(instances, {"merge_area_ratio": 2.0})
    flag_split_suspects(instances, {"split_area_ratio": 0.5})

    assert instances[0].status == "border_artifact"
    assert instances[1].status == "possible_merged"
    assert instances[2].status == "possible_split"

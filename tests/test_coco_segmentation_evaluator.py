import json

import cv2
import numpy as np

from ricevision_qi.app.tools.coco_segmentation_evaluator import (
    compute_iou,
    evaluate_image_masks,
    load_coco_annotations,
    rasterize_polygons,
)


def test_load_coco_annotations_reads_images_and_masks(tmp_path):
    split_dir = tmp_path / "train"
    split_dir.mkdir()
    coco = {
        "images": [{"id": 1, "file_name": "sample.jpg", "width": 20, "height": 10}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "segmentation": [[1, 1, 8, 1, 8, 6, 1, 6]]}
        ],
        "categories": [{"id": 1, "name": "rice"}],
    }
    (split_dir / "_annotations.coco.json").write_text(json.dumps(coco), encoding="utf-8")

    records = load_coco_annotations(tmp_path)

    assert len(records) == 1
    assert records[0].split == "train"
    assert records[0].file_name == "sample.jpg"
    assert records[0].gt_mask.shape == (10, 20)
    assert np.count_nonzero(records[0].gt_mask) > 0


def test_rasterize_polygons_fills_polygon_area():
    mask = rasterize_polygons([[[2, 2, 7, 2, 7, 7, 2, 7]]], width=10, height=10)

    assert mask.shape == (10, 10)
    assert mask[4, 4] == 255
    assert mask[0, 0] == 0


def test_compute_iou_handles_empty_and_overlap_cases():
    empty = np.zeros((10, 10), dtype=np.uint8)
    square = np.zeros((10, 10), dtype=np.uint8)
    square[2:6, 2:6] = 255
    other = np.zeros((10, 10), dtype=np.uint8)
    other[7:9, 7:9] = 255

    assert compute_iou(empty, empty) == 1.0
    assert compute_iou(square, square) == 1.0
    assert compute_iou(square, other) == 0.0
    assert compute_iou(empty, square) == 0.0


def test_evaluate_image_masks_empty_prediction_and_gt_cases():
    empty = np.zeros((10, 10), dtype=np.uint8)
    square = np.zeros((10, 10), dtype=np.uint8)
    square[2:6, 2:6] = 255

    no_prediction = evaluate_image_masks(empty, square, [0.3, 0.5])
    no_gt = evaluate_image_masks(square, empty, [0.3, 0.5])
    both_empty = evaluate_image_masks(empty, empty, [0.3, 0.5])

    assert no_prediction["iou"] == 0.0
    assert no_prediction["tp@0.5"] == 0
    assert no_prediction["fn@0.5"] == 1
    assert no_gt["fp@0.5"] == 1
    assert both_empty["iou"] == 1.0
    assert both_empty["tp@0.5"] == 0


def test_evaluate_image_masks_match_and_mismatch():
    gt = np.zeros((12, 12), dtype=np.uint8)
    gt[2:8, 2:8] = 255
    pred_match = gt.copy()
    pred_miss = np.zeros((12, 12), dtype=np.uint8)
    pred_miss[9:11, 9:11] = 255

    match = evaluate_image_masks(pred_match, gt, [0.3, 0.5])
    miss = evaluate_image_masks(pred_miss, gt, [0.3, 0.5])

    assert match["tp@0.5"] == 1
    assert match["precision@0.5"] == 1.0
    assert match["recall@0.5"] == 1.0
    assert miss["fp@0.5"] == 1
    assert miss["fn@0.5"] == 1

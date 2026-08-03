import csv

import cv2
import numpy as np

from ricevision_qi.app.tools.yolo_seg_evaluator import (
    compute_iou,
    evaluate_instance_masks,
    load_yolo_label_instances,
    main,
    match_instances,
    polygon_to_mask,
)


def test_load_yolo_label_instances_converts_normalized_polygons(tmp_path):
    label_path = tmp_path / "sample.txt"
    label_path.write_text("2 0.1 0.2 0.4 0.2 0.4 0.6 0.1 0.6\n", encoding="utf-8")

    instances = load_yolo_label_instances(label_path, width=100, height=50)

    assert len(instances) == 1
    assert instances[0].class_id == 2
    assert instances[0].mask.shape == (50, 100)
    assert instances[0].polygon_px[0] == (10, 10)
    assert np.count_nonzero(instances[0].mask) > 0


def test_polygon_to_mask_rasterizes_polygon_area():
    mask = polygon_to_mask([(2, 2), (7, 2), (7, 7), (2, 7)], width=10, height=10)

    assert mask.shape == (10, 10)
    assert mask[4, 4] == 255
    assert mask[0, 0] == 0


def test_compute_iou_for_overlap_empty_and_non_overlap_cases():
    empty = np.zeros((10, 10), dtype=np.uint8)
    square = np.zeros((10, 10), dtype=np.uint8)
    square[2:6, 2:6] = 255
    shifted = np.zeros((10, 10), dtype=np.uint8)
    shifted[4:8, 4:8] = 255
    far = np.zeros((10, 10), dtype=np.uint8)
    far[8:10, 8:10] = 255

    assert compute_iou(empty, empty) == 1.0
    assert compute_iou(square, square) == 1.0
    assert compute_iou(square, far) == 0.0
    assert round(compute_iou(square, shifted), 3) == 0.143


def test_match_instances_uses_best_iou_without_reusing_masks():
    gt_a = np.zeros((20, 20), dtype=np.uint8)
    gt_a[2:8, 2:8] = 255
    gt_b = np.zeros((20, 20), dtype=np.uint8)
    gt_b[12:18, 12:18] = 255
    pred_a = gt_a.copy()
    pred_b = gt_b.copy()
    pred_fp = np.zeros((20, 20), dtype=np.uint8)
    pred_fp[0:2, 18:20] = 255

    matches = match_instances([gt_a, gt_b], [pred_fp, pred_b, pred_a], iou_threshold=0.5)

    assert [(m.gt_index, m.pred_index) for m in matches] == [(0, 2), (1, 1)]
    assert all(m.iou == 1.0 for m in matches)


def test_evaluate_instance_masks_counts_fp_fn_and_segmentation_diagnostics():
    gt_a = np.zeros((30, 30), dtype=np.uint8)
    gt_a[2:12, 2:12] = 255
    gt_b = np.zeros((30, 30), dtype=np.uint8)
    gt_b[18:28, 18:28] = 255
    pred_match = gt_a.copy()
    pred_under = np.zeros((30, 30), dtype=np.uint8)
    pred_under[16:29, 16:29] = 255
    pred_under[2:3, 2:12] = 255
    pred_fp = np.zeros((30, 30), dtype=np.uint8)
    pred_fp[0:3, 25:29] = 255

    metrics = evaluate_instance_masks([gt_a, gt_b], [pred_match, pred_under, pred_fp], thresholds=[0.5, 0.75])

    assert metrics["tp_50"] == 2
    assert metrics["fp_50"] == 1
    assert metrics["fn_50"] == 0
    assert metrics["precision_50"] == 2 / 3
    assert metrics["recall_50"] == 1.0
    assert metrics["under_segmentation_count"] == 1
    assert metrics["false_positive_count"] == 1


def test_evaluator_missing_label_file_generates_csv_and_summary(tmp_path):
    image_dir = tmp_path / "images"
    label_dir = tmp_path / "labels"
    output_dir = tmp_path / "out"
    image_dir.mkdir()
    label_dir.mkdir()
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    cv2.imencode(".jpg", image)[1].tofile(str(image_dir / "sample.jpg"))

    exit_code = main(
        [
            "--images",
            str(image_dir),
            "--labels",
            str(label_dir),
            "--output",
            str(output_dir),
            "--profile",
            "clean_background",
        ]
    )

    assert exit_code == 0
    run_dir = next(output_dir.glob("run_*"))
    csv_path = run_dir / "yolo_seg_eval_results.csv"
    summary_path = run_dir / "yolo_seg_eval_summary.md"
    assert csv_path.exists()
    assert summary_path.exists()
    assert (run_dir / "overlays").is_dir()
    with csv_path.open(newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    assert rows[0]["num_gt"] == "0"
    assert rows[0]["image_name"] == "sample.jpg"


def test_evaluator_can_skip_overlays_and_contact_sheet(tmp_path):
    image_dir = tmp_path / "images"
    label_dir = tmp_path / "labels"
    output_dir = tmp_path / "out"
    image_dir.mkdir()
    label_dir.mkdir()
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    cv2.imencode(".jpg", image)[1].tofile(str(image_dir / "sample.jpg"))

    exit_code = main(
        [
            "--images",
            str(image_dir),
            "--labels",
            str(label_dir),
            "--output",
            str(output_dir),
            "--profile",
            "clean_background",
            "--no-overlays",
            "--no-contact-sheet",
        ]
    )

    assert exit_code == 0
    run_dir = next(output_dir.glob("run_*"))
    assert (run_dir / "yolo_seg_eval_results.csv").exists()
    assert not (run_dir / "contact_sheet_eval.jpg").exists()
    assert not (run_dir / "overlays").exists()

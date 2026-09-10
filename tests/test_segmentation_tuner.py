import csv

import cv2
import numpy as np

from ricevision_qi.app.tools.segmentation_tuner import (
    iter_parameter_grid,
    load_tuning_grid,
    main,
    score_segmentation_result,
)
from tests.synthetic_images import make_bagged_grain_image


def test_score_rejects_empty_detection():
    score, reasons = score_segmentation_result({"num_instances": 0}, {})

    assert score == 0.0
    assert "no instances detected" in reasons


def test_load_tuning_grid_and_iterate_small_grid(tmp_path):
    grid_path = tmp_path / "grid.yaml"
    grid_path.write_text(
        "\n".join(
            [
                "grid:",
                "  segmentation_method: [lab_threshold]",
                "  kernel_size: [3]",
                "  open_iter: [0, 1]",
                "  close_iter: [1]",
                "  min_area: [50]",
                "  watershed_dist_ratio: [0.25]",
                "  fill_holes: [true, false]",
            ]
        ),
        encoding="utf-8",
    )

    combinations = list(iter_parameter_grid(load_tuning_grid(grid_path)))

    assert len(combinations) == 4
    assert combinations[0]["segmentation_method"] == "lab_threshold"
    assert combinations[0]["method"] == "lab_threshold"


def test_segmentation_tuner_writes_outputs(tmp_path):
    image_path = tmp_path / "sample.jpg"
    image_rgb = make_bagged_grain_image()
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    cv2.imencode(".jpg", image_bgr)[1].tofile(str(image_path))

    grid_path = tmp_path / "grid.yaml"
    grid_path.write_text(
        "\n".join(
            [
                "grid:",
                "  segmentation_method: [lab_threshold]",
                "  kernel_size: [3]",
                "  open_iter: [0]",
                "  close_iter: [1]",
                "  min_area: [50]",
                "  watershed_dist_ratio: [0.25]",
                "  fill_holes: [false]",
            ]
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "tuning"

    exit_code = main(
        [
            "--image",
            str(image_path),
            "--output",
            str(output_dir),
            "--grid",
            str(grid_path),
            "--roi",
            "50",
            "50",
            "420",
            "340",
        ]
    )

    assert exit_code == 0
    run_dirs = list(output_dir.glob("run_*"))
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    assert (run_dir / "tuning_results.csv").exists()
    assert (run_dir / "instance_stats.csv").exists()
    assert (run_dir / "best_params.yaml").exists()
    assert (run_dir / "tuning_report.md").exists()
    assert (run_dir / "contact_sheet_overlay.jpg").exists()
    assert (run_dir / "contact_sheet_mask.jpg").exists()
    assert list((run_dir / "image_001").glob("params_0001_overlay.jpg"))

    with (run_dir / "tuning_results.csv").open(encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    assert rows
    assert "num_instances" in rows[0]
    assert "score" in rows[0]

import csv

import cv2

from ricevision_qi.app.tools.real_image_validator import main
from tests.synthetic_images import make_multiple_grain_image


def test_real_image_validator_writes_review_outputs(tmp_path):
    input_dir = tmp_path / "real_images"
    output_dir = tmp_path / "validation"
    input_dir.mkdir()
    image = make_multiple_grain_image()
    cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))[1].tofile(str(input_dir / "sample.jpg"))

    exit_code = main(
        [
            "--input",
            str(input_dir),
            "--output",
            str(output_dir),
            "--profiles",
            "clean_background",
            "plastic_bag_rice",
            "--save-masks",
            "--save-debug-limit",
            "1",
        ]
    )

    assert exit_code == 0
    run_dirs = list(output_dir.glob("run_*"))
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    assert (run_dir / "real_validation_results.csv").exists()
    assert (run_dir / "real_validation_report.md").exists()
    assert (run_dir / "contact_sheet_real_validation.jpg").exists()
    assert list((run_dir / "overlays" / "clean_background").glob("*_overlay.jpg"))
    assert list((run_dir / "masks" / "clean_background").glob("*.png"))
    assert list((run_dir / "debug_views").rglob("overlay.png"))
    with (run_dir / "real_validation_results.csv").open(encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    assert {row["profile"] for row in rows} == {"clean_background", "plastic_bag_rice"}


def test_real_image_validator_handles_empty_input(tmp_path):
    input_dir = tmp_path / "empty"
    output_dir = tmp_path / "validation"
    input_dir.mkdir()

    exit_code = main(["--input", str(input_dir), "--output", str(output_dir)])

    assert exit_code == 0
    run_dir = next(output_dir.glob("run_*"))
    assert (run_dir / "real_validation_results.csv").exists()
    report = (run_dir / "real_validation_report.md").read_text(encoding="utf-8")
    assert "No supported images were found" in report


def test_real_image_validator_respects_max_images(tmp_path):
    input_dir = tmp_path / "real_images"
    output_dir = tmp_path / "validation"
    input_dir.mkdir()
    image = make_multiple_grain_image()
    for index in range(2):
        cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))[1].tofile(
            str(input_dir / f"sample_{index}.jpg")
        )

    exit_code = main(
        [
            "--input",
            str(input_dir),
            "--output",
            str(output_dir),
            "--profiles",
            "clean_background",
            "--max-images",
            "1",
            "--no-contact-sheet",
        ]
    )

    assert exit_code == 0
    run_dir = next(output_dir.glob("run_*"))
    with (run_dir / "real_validation_results.csv").open(encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 1

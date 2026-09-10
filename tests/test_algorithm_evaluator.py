import csv

import cv2

from ricevision_qi.app.tools.algorithm_evaluator import main
from tests.synthetic_images import make_multiple_grain_image


def test_algorithm_evaluator_writes_csv_report_and_contact_sheet(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
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
        ]
    )

    assert exit_code == 0
    result_file = output_dir / "algorithm_eval_results.csv"
    assert result_file.exists()
    assert (output_dir / "algorithm_eval_report.md").exists()
    assert (output_dir / "contact_sheet_by_profile.jpg").exists()
    with result_file.open(encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    assert {row["profile"] for row in rows} == {"clean_background", "plastic_bag_rice"}

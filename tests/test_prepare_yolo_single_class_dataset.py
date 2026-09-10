import cv2
import numpy as np

from ricevision_qi.app.tools.prepare_yolo_single_class_dataset import main


def test_prepare_yolo_single_class_dataset_rewrites_labels_and_yaml(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    data_yaml = tmp_path / "rice_grain_seg.yaml"
    for split in ("train", "valid", "test"):
        (src / split / "images").mkdir(parents=True)
        (src / split / "labels").mkdir(parents=True)
        image = np.zeros((16, 16, 3), dtype=np.uint8)
        cv2.imencode(".jpg", image)[1].tofile(str(src / split / "images" / f"{split}_sample.jpg"))
        (src / split / "labels" / f"{split}_sample.txt").write_text(
            "2 0.1 0.1 0.4 0.1 0.4 0.4 0.1 0.4\n"
            "5 0.5 0.5 0.8 0.5 0.8 0.8 0.5 0.8\n",
            encoding="utf-8",
        )

    exit_code = main(
        [
            "--src",
            str(src),
            "--dst",
            str(dst),
            "--class-name",
            "rice_grain",
            "--data-yaml",
            str(data_yaml),
        ]
    )

    assert exit_code == 0
    for split in ("train", "valid", "test"):
        converted = (dst / split / "labels" / f"{split}_sample.txt").read_text(encoding="utf-8").splitlines()
        assert all(line.startswith("0 ") for line in converted)
        assert (dst / split / "images" / f"{split}_sample.jpg").exists()
    assert "0: rice_grain" in data_yaml.read_text(encoding="utf-8")
    report = (dst / "single_class_conversion_report.md").read_text(encoding="utf-8")
    assert "All class IDs merged to 0: yes" in report


def test_prepare_yolo_single_class_dataset_reports_bad_labels(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    data_yaml = tmp_path / "data.yaml"
    (src / "train" / "images").mkdir(parents=True)
    (src / "train" / "labels").mkdir(parents=True)
    for split in ("valid", "test"):
        (src / split / "images").mkdir(parents=True)
        (src / split / "labels").mkdir(parents=True)
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    cv2.imencode(".jpg", image)[1].tofile(str(src / "train" / "images" / "bad.jpg"))
    (src / "train" / "labels" / "bad.txt").write_text("2 0.1 0.2\n", encoding="utf-8")

    exit_code = main(["--src", str(src), "--dst", str(dst), "--data-yaml", str(data_yaml)])

    assert exit_code == 0
    report = (dst / "single_class_conversion_report.md").read_text(encoding="utf-8")
    assert "Bad labels: 1" in report

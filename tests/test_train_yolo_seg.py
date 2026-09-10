from ricevision_qi.app.tools import train_yolo_seg


def test_train_yolo_seg_reports_missing_ultralytics_without_crashing(tmp_path, monkeypatch):
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text("path: .\ntrain: train/images\nval: valid/images\nnames:\n  0: rice_grain\n", encoding="utf-8")
    monkeypatch.setattr(train_yolo_seg.importlib.util, "find_spec", lambda name: None if name == "ultralytics" else object())

    exit_code = train_yolo_seg.main(
        [
            "--data",
            str(data_yaml),
            "--model",
            "yolo11n-seg.pt",
            "--epochs",
            "1",
            "--imgsz",
            "320",
            "--batch",
            "2",
            "--project",
            str(tmp_path / "runs"),
            "--name",
            "smoke",
        ]
    )

    assert exit_code == 2


def test_train_yolo_seg_dry_run_does_not_require_ultralytics(tmp_path):
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text("path: .\ntrain: train/images\nval: valid/images\nnames:\n  0: rice_grain\n", encoding="utf-8")

    exit_code = train_yolo_seg.main(["--data", str(data_yaml), "--dry-run"])

    assert exit_code == 0

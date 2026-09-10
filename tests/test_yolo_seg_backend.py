import sys

import numpy as np

from ricevision_qi.app.core.yolo_seg_backend import YoloSegBackend


def test_yolo_seg_backend_missing_weights_returns_empty_result(tmp_path):
    backend = YoloSegBackend()
    image = np.zeros((32, 32, 3), dtype=np.uint8)

    result = backend.segment(image, {"weights": str(tmp_path / "missing.pt")})

    assert result["instances"] == []
    assert result["stats"]["num_instances"] == 0
    assert "weights not found" in result["stats"]["message"]


def test_yolo_seg_backend_missing_ultralytics_returns_empty_result(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "ultralytics", None)
    weights = tmp_path / "fake.pt"
    weights.write_bytes(b"not a model")
    backend = YoloSegBackend()
    image = np.zeros((16, 16, 3), dtype=np.uint8)

    result = backend.segment(image, {"weights": str(weights)})

    assert result["instances"] == []
    assert "ultralytics is not installed" in result["stats"]["message"]

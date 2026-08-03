from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.dry_run:
        print(_command_summary(args), flush=True)
        return 0
    if importlib.util.find_spec("ultralytics") is None:
        print(
            "ultralytics is not installed. Install with: conda run -n RiceVision-QI pip install ultralytics",
            flush=True,
        )
        return 2
    if not Path(args.data).exists():
        print(f"data yaml not found: {args.data}", flush=True)
        return 2

    from ultralytics import YOLO  # type: ignore

    model_name = args.model
    try:
        model = YOLO(model_name)
    except Exception:
        if model_name == "yolo11n-seg.pt":
            model_name = "yolov8n-seg.pt"
            print("yolo11n-seg.pt is unavailable; falling back to yolov8n-seg.pt", flush=True)
            model = YOLO(model_name)
        else:
            raise
    train_kwargs = {
        "data": args.data,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "project": str(Path(args.project).resolve()),
        "name": args.name,
        "patience": args.patience,
        "workers": args.workers,
        "exist_ok": True,
    }
    if args.device:
        train_kwargs["device"] = args.device
    model.train(**train_kwargs)
    print(f"YOLO-seg training finished: {Path(args.project) / args.name}", flush=True)
    return 0


def _command_summary(args: argparse.Namespace) -> str:
    return (
        "YOLO-seg dry run: "
        f"data={args.data}, model={args.model}, epochs={args.epochs}, imgsz={args.imgsz}, "
        f"batch={args.batch}, device={args.device or 'auto'}, project={args.project}, name={args.name}"
    )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a YOLO segmentation model for rice grain masks.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", default="yolo11n-seg.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default=None)
    parser.add_argument("--project", default="samples/output/yolo_train")
    parser.add_argument("--name", default="rice_grain_yolo_seg")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())

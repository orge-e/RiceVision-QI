from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


IOU_THRESHOLDS = [0.3, 0.5]


@dataclass
class CocoImageRecord:
    split: str
    image_id: int
    file_name: str
    width: int
    height: int
    gt_mask: np.ndarray


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    profiles = args.profiles
    records = load_coco_annotations(Path(args.coco_root))
    rows = []
    for record in records:
        for profile in profiles:
            pred_mask = load_prediction_mask(Path(args.prediction_root), profile, record)
            metrics = evaluate_image_masks(pred_mask, record.gt_mask, IOU_THRESHOLDS)
            row = {
                "split": record.split,
                "image_name": record.file_name,
                "profile": profile,
                "gt_area": int(np.count_nonzero(record.gt_mask)),
                "pred_area": int(np.count_nonzero(pred_mask)),
                "intersection": metrics["intersection"],
                "union": metrics["union"],
                "iou": metrics["iou"],
            }
            for threshold in IOU_THRESHOLDS:
                suffix = _threshold_suffix(threshold)
                row[f"tp@{suffix}"] = metrics[f"tp@{threshold}"]
                row[f"fp@{suffix}"] = metrics[f"fp@{threshold}"]
                row[f"fn@{suffix}"] = metrics[f"fn@{threshold}"]
                row[f"precision@{suffix}"] = metrics[f"precision@{threshold}"]
                row[f"recall@{suffix}"] = metrics[f"recall@{threshold}"]
                row[f"f1@{suffix}"] = metrics[f"f1@{threshold}"]
            rows.append(row)

    per_image_csv = output_dir / "coco_segmentation_per_image.csv"
    summary_csv = output_dir / "coco_segmentation_profile_summary.csv"
    report_path = output_dir / "coco_segmentation_report.md"
    _write_per_image_csv(per_image_csv, rows)
    summary_rows = summarize_profile_metrics(rows, profiles)
    _write_summary_csv(summary_csv, summary_rows)
    _write_report(report_path, summary_rows, records, profiles)
    print(f"COCO segmentation evaluation finished: {output_dir}")
    return 0


def load_coco_annotations(coco_root: Path) -> list[CocoImageRecord]:
    records = []
    for annotation_path in sorted(coco_root.glob("*/_annotations.coco.json")):
        split = annotation_path.parent.name
        data = json.loads(annotation_path.read_text(encoding="utf-8"))
        images = {int(item["id"]): item for item in data.get("images", [])}
        annotations_by_image: dict[int, list] = {image_id: [] for image_id in images}
        for annotation in data.get("annotations", []):
            annotations_by_image.setdefault(int(annotation["image_id"]), []).append(annotation)
        for image_id, image in images.items():
            width = int(image["width"])
            height = int(image["height"])
            polygons = [ann.get("segmentation", []) for ann in annotations_by_image.get(image_id, [])]
            mask = rasterize_polygons(polygons, width=width, height=height)
            records.append(
                CocoImageRecord(
                    split=split,
                    image_id=image_id,
                    file_name=Path(image["file_name"]).name,
                    width=width,
                    height=height,
                    gt_mask=mask,
                )
            )
    return records


def rasterize_polygons(segmentations, width: int, height: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    for segmentation in segmentations:
        if isinstance(segmentation, dict):
            continue
        polygons = segmentation if _is_polygon_list(segmentation) else [segmentation]
        for polygon in polygons:
            if not polygon or len(polygon) < 6:
                continue
            points = np.array(polygon, dtype=np.float32).reshape(-1, 2)
            points[:, 0] = np.clip(points[:, 0], 0, width - 1)
            points[:, 1] = np.clip(points[:, 1], 0, height - 1)
            cv2.fillPoly(mask, [np.round(points).astype(np.int32)], 255)
    return mask


def compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    pred = pred_mask > 0
    gt = gt_mask > 0
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0
    intersection = np.logical_and(pred, gt).sum()
    return float(intersection / union)


def evaluate_image_masks(pred_mask: np.ndarray, gt_mask: np.ndarray, thresholds: list[float]) -> dict:
    pred = pred_mask > 0
    gt = gt_mask > 0
    intersection = int(np.logical_and(pred, gt).sum())
    union = int(np.logical_or(pred, gt).sum())
    iou = compute_iou(pred_mask, gt_mask)
    has_pred = bool(pred.any())
    has_gt = bool(gt.any())
    metrics = {"intersection": intersection, "union": union, "iou": iou}
    for threshold in thresholds:
        match = has_pred and has_gt and iou >= threshold
        tp = int(match)
        fp = int(has_pred and not match)
        fn = int(has_gt and not match)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        metrics[f"tp@{threshold}"] = tp
        metrics[f"fp@{threshold}"] = fp
        metrics[f"fn@{threshold}"] = fn
        metrics[f"precision@{threshold}"] = precision
        metrics[f"recall@{threshold}"] = recall
        metrics[f"f1@{threshold}"] = f1
    return metrics


def load_prediction_mask(prediction_root: Path, profile: str, record: CocoImageRecord) -> np.ndarray:
    profile_dir = prediction_root / "masks" / profile
    for stem in _candidate_prediction_stems(record):
        mask_path = profile_dir / f"{stem}.png"
        if mask_path.exists():
            image = cv2.imdecode(np.fromfile(str(mask_path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            if image is None:
                break
            if image.shape != (record.height, record.width):
                image = cv2.resize(image, (record.width, record.height), interpolation=cv2.INTER_NEAREST)
            return np.where(image > 0, 255, 0).astype(np.uint8)
    return np.zeros((record.height, record.width), dtype=np.uint8)


def summarize_profile_metrics(rows: list[dict], profiles: list[str]) -> list[dict]:
    summary = []
    for profile in profiles:
        profile_rows = [row for row in rows if row["profile"] == profile]
        if not profile_rows:
            continue
        for threshold in IOU_THRESHOLDS:
            suffix = _threshold_suffix(threshold)
            tp = sum(int(row[f"tp@{suffix}"]) for row in profile_rows)
            fp = sum(int(row[f"fp@{suffix}"]) for row in profile_rows)
            fn = sum(int(row[f"fn@{suffix}"]) for row in profile_rows)
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
            summary.append(
                {
                    "profile": profile,
                    "iou_threshold": threshold,
                    "images": len(profile_rows),
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "mean_iou": float(np.mean([float(row["iou"]) for row in profile_rows])),
                    "mean_gt_area": float(np.mean([float(row["gt_area"]) for row in profile_rows])),
                    "mean_pred_area": float(np.mean([float(row["pred_area"]) for row in profile_rows])),
                }
            )
    return summary


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate saved prediction masks against COCO segmentation annotations.")
    parser.add_argument("--coco-root", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profiles", nargs="+", required=True)
    return parser.parse_args(argv)


def _write_per_image_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "split",
        "image_name",
        "profile",
        "gt_area",
        "pred_area",
        "intersection",
        "union",
        "iou",
        "tp@0_3",
        "fp@0_3",
        "fn@0_3",
        "precision@0_3",
        "recall@0_3",
        "f1@0_3",
        "tp@0_5",
        "fp@0_5",
        "fn@0_5",
        "precision@0_5",
        "recall@0_5",
        "f1@0_5",
    ]
    _write_csv(path, rows, fields)


def _write_summary_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "profile",
        "iou_threshold",
        "images",
        "tp",
        "fp",
        "fn",
        "precision",
        "recall",
        "f1",
        "mean_iou",
        "mean_gt_area",
        "mean_pred_area",
    ]
    _write_csv(path, rows, fields)


def _write_report(path: Path, summary_rows: list[dict], records: list[CocoImageRecord], profiles: list[str]) -> None:
    lines = [
        "# COCO Segmentation Evaluation Report",
        "",
        f"- Images evaluated: {len(records)}",
        f"- Profiles: {', '.join(profiles)}",
        "- Matching unit: image-level binary foreground mask",
        "- IoU thresholds: 0.3, 0.5",
        "",
        "## Profile Summary",
        "",
        "| Profile | IoU Threshold | Precision | Recall | F1 | Mean IoU | TP | FP | FN |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['profile']} | {row['iou_threshold']} | {row['precision']:.4f} | "
            f"{row['recall']:.4f} | {row['f1']:.4f} | {row['mean_iou']:.4f} | "
            f"{row['tp']} | {row['fp']} | {row['fn']} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "This first quantitative evaluator compares each profile's saved binary foreground mask with the union of COCO polygon masks for the image.",
            "It does not yet perform per-instance matching or per-class AP calculation.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _candidate_prediction_stems(record: CocoImageRecord) -> list[str]:
    stem = Path(record.file_name).stem
    candidates = [
        f"{record.split}_{stem}",
        stem,
    ]
    for prefix in ("train_", "valid_", "test_"):
        if stem.startswith(prefix):
            candidates.append(stem[len(prefix) :])
    return list(dict.fromkeys(candidates))


def _threshold_suffix(threshold: float) -> str:
    return str(threshold).replace(".", "_")


def _is_polygon_list(segmentation) -> bool:
    return bool(segmentation) and isinstance(segmentation[0], list)


if __name__ == "__main__":
    raise SystemExit(main())

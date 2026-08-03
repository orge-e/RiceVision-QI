from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from ricevision_qi.app.core.config import load_profile_config
from ricevision_qi.app.core.image_io import SUPPORTED_IMAGE_EXTENSIONS, read_image
from ricevision_qi.app.core.pipeline import run_detection


IOU_THRESHOLDS = [0.5, 0.75]
OVERLAP_RATIO_THRESHOLD = 0.1


@dataclass
class YoloSegInstance:
    class_id: int
    polygon_px: list[tuple[int, int]]
    mask: np.ndarray


@dataclass(frozen=True)
class InstanceMatch:
    gt_index: int
    pred_index: int
    iou: float


@dataclass(frozen=True)
class ImageRecord:
    image_path: Path
    label_path: Path
    split: str


CSV_FIELDS = [
    "image_name",
    "split",
    "profile",
    "num_gt",
    "num_pred",
    "tp_50",
    "fp_50",
    "fn_50",
    "precision_50",
    "recall_50",
    "f1_50",
    "mean_iou_50",
    "tp_75",
    "fp_75",
    "fn_75",
    "precision_75",
    "recall_75",
    "f1_75",
    "mean_iou_75",
    "over_segmentation_count",
    "under_segmentation_count",
    "missed_grain_count",
    "false_positive_count",
    "runtime_ms",
]


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    profiles = args.profiles or [args.profile]
    run_dir = Path(args.output) / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    overlays_dir = run_dir / "overlays"
    if not args.no_overlays:
        overlays_dir.mkdir(parents=True, exist_ok=True)

    records = collect_image_records(args)
    print(
        f"Loaded {len(records)} images, profiles={', '.join(profiles)}, "
        f"max_dim={args.max_processing_dimension or 'profile'}",
        flush=True,
    )
    eval_start = time.perf_counter()
    rows = []
    overlay_paths = []
    saved_overlay_count = 0
    for image_index, record in enumerate(records, start=1):
        try:
            loaded = read_image(record.image_path)
        except Exception as exc:  # pragma: no cover - defensive CLI path
            print(f"Skipping unreadable image {record.image_path}: {exc}", flush=True)
            continue

        height, width = loaded.image_rgb.shape[:2]
        gt_instances = load_yolo_label_instances(record.label_path, width=width, height=height)
        gt_masks = [instance.mask for instance in gt_instances]
        for profile in profiles:
            cfg = load_profile_config(profile)
            if args.max_processing_dimension and args.max_processing_dimension > 0:
                cfg["max_processing_dimension"] = int(args.max_processing_dimension)
            print(
                f"[{image_index}/{len(records)}][{profile}] processing {record.image_path.name}, "
                f"size={width}x{height}, max_dim={args.max_processing_dimension or cfg.get('max_processing_dimension', 'profile')}",
                flush=True,
            )
            start = time.perf_counter()
            result = run_detection(loaded.image_rgb, cfg)
            pred_masks = prediction_masks_from_grains(result.grains, (height, width))
            gt_eval_masks, pred_eval_masks = resize_masks_for_evaluation(
                gt_masks,
                pred_masks,
                max_dimension=args.eval_mask_max_dimension,
            )
            metrics = evaluate_instance_masks(gt_eval_masks, pred_eval_masks, thresholds=IOU_THRESHOLDS)
            if _should_save_overlay(args, saved_overlay_count):
                overlay_matches = match_instances(gt_eval_masks, pred_eval_masks, 0.5)
                overlay = draw_evaluation_overlay(
                    loaded.image_rgb,
                    gt_masks,
                    pred_masks,
                    overlay_matches,
                    metrics,
                )
                overlay_path = overlays_dir / f"{profile}_{record.image_path.stem}_overlay.jpg"
                _save_rgb_image(overlay_path, overlay)
                overlay_paths.append(overlay_path)
                saved_overlay_count += 1
            runtime_ms = (time.perf_counter() - start) * 1000.0
            print(
                f"[{image_index}/{len(records)}][{profile}] done, gt={len(gt_masks)}, pred={len(pred_masks)}, "
                f"time={runtime_ms / 1000.0:.2f}s, elapsed={time.perf_counter() - eval_start:.2f}s",
                flush=True,
            )
            row = {
                "image_name": record.image_path.name,
                "split": record.split,
                "profile": profile,
                "num_gt": len(gt_masks),
                "num_pred": len(pred_masks),
                "runtime_ms": runtime_ms,
            }
            row.update(metrics)
            rows.append(row)

    _write_csv(run_dir / "yolo_seg_eval_results.csv", rows)
    summary_rows = summarize_rows(rows, profiles)
    _write_summary(run_dir / "yolo_seg_eval_summary.md", args, records, summary_rows)
    if not args.no_contact_sheet and overlay_paths:
        _write_contact_sheet(run_dir / "contact_sheet_eval.jpg", overlay_paths)
    print(f"YOLO segmentation evaluation finished: {run_dir}", flush=True)
    return 0


def load_yolo_label_instances(label_path: Path, width: int, height: int) -> list[YoloSegInstance]:
    if not label_path.exists():
        return []
    instances = []
    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 7 or (len(parts) - 1) % 2:
            continue
        try:
            class_id = int(float(parts[0]))
            coords = [float(value) for value in parts[1:]]
        except ValueError:
            continue
        points = []
        for x_norm, y_norm in zip(coords[0::2], coords[1::2]):
            x = int(round(np.clip(x_norm, 0.0, 1.0) * (width - 1)))
            y = int(round(np.clip(y_norm, 0.0, 1.0) * (height - 1)))
            points.append((x, y))
        mask = polygon_to_mask(points, width=width, height=height)
        instances.append(YoloSegInstance(class_id=class_id, polygon_px=points, mask=mask))
    return instances


def polygon_to_mask(points: list[tuple[int, int]], width: int, height: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    if len(points) < 3:
        return mask
    contour = np.array(points, dtype=np.int32).reshape(-1, 1, 2)
    contour[:, 0, 0] = np.clip(contour[:, 0, 0], 0, width - 1)
    contour[:, 0, 1] = np.clip(contour[:, 0, 1], 0, height - 1)
    cv2.fillPoly(mask, [contour], 255)
    return mask


def prediction_masks_from_grains(grains: list, shape: tuple[int, int]) -> list[np.ndarray]:
    masks = []
    height, width = shape
    for grain in grains:
        mask = getattr(grain, "mask", None)
        if isinstance(mask, np.ndarray) and mask.shape == (height, width):
            masks.append(np.where(mask > 0, 255, 0).astype(np.uint8))
            continue
        contour = getattr(grain, "contour", None)
        if contour is None or len(contour) < 3:
            continue
        full_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.drawContours(full_mask, [contour.astype(np.int32)], -1, 255, -1)
        masks.append(full_mask)
    return masks


def resize_masks_for_evaluation(
    gt_masks: list[np.ndarray],
    pred_masks: list[np.ndarray],
    max_dimension: int = 0,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    if max_dimension <= 0:
        return gt_masks, pred_masks
    shape = _evaluation_shape(gt_masks, pred_masks, max_dimension)
    if shape is None:
        return gt_masks, pred_masks
    return (
        [_resize_binary_mask(mask, shape) for mask in gt_masks],
        [_resize_binary_mask(mask, shape) for mask in pred_masks],
    )


def _evaluation_shape(
    gt_masks: list[np.ndarray],
    pred_masks: list[np.ndarray],
    max_dimension: int,
) -> tuple[int, int] | None:
    masks = gt_masks or pred_masks
    if not masks:
        return None
    height, width = masks[0].shape[:2]
    largest = max(height, width)
    if largest <= max_dimension:
        return height, width
    scale = max_dimension / float(largest)
    return max(1, int(round(height * scale))), max(1, int(round(width * scale)))


def _resize_binary_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    if mask.shape[:2] == (height, width):
        return mask
    resized = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    return np.where(resized > 0, 255, 0).astype(np.uint8)


def compute_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = mask_a > 0
    b = mask_b > 0
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(a, b).sum() / union)


def match_instances(gt_masks: list[np.ndarray], pred_masks: list[np.ndarray], iou_threshold: float) -> list[InstanceMatch]:
    candidates = []
    for gt_index, gt_mask in enumerate(gt_masks):
        for pred_index, pred_mask in enumerate(pred_masks):
            iou = compute_iou(pred_mask, gt_mask)
            if iou >= iou_threshold:
                candidates.append(InstanceMatch(gt_index=gt_index, pred_index=pred_index, iou=iou))
    candidates.sort(key=lambda item: (-item.iou, item.gt_index, item.pred_index))
    used_gt = set()
    used_pred = set()
    matches = []
    for candidate in candidates:
        if candidate.gt_index in used_gt or candidate.pred_index in used_pred:
            continue
        matches.append(candidate)
        used_gt.add(candidate.gt_index)
        used_pred.add(candidate.pred_index)
    return sorted(matches, key=lambda item: item.gt_index)


def evaluate_instance_masks(gt_masks: list[np.ndarray], pred_masks: list[np.ndarray], thresholds: list[float]) -> dict:
    metrics = {
        "over_segmentation_count": count_over_segmentation(gt_masks, pred_masks),
        "under_segmentation_count": count_under_segmentation(gt_masks, pred_masks),
    }
    for threshold in thresholds:
        suffix = _threshold_suffix(threshold)
        matches = match_instances(gt_masks, pred_masks, threshold)
        tp = len(matches)
        fp = max(0, len(pred_masks) - tp)
        fn = max(0, len(gt_masks) - tp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        metrics[f"tp_{suffix}"] = tp
        metrics[f"fp_{suffix}"] = fp
        metrics[f"fn_{suffix}"] = fn
        metrics[f"precision_{suffix}"] = precision
        metrics[f"recall_{suffix}"] = recall
        metrics[f"f1_{suffix}"] = f1
        metrics[f"mean_iou_{suffix}"] = float(np.mean([match.iou for match in matches])) if matches else 0.0
        if threshold == 0.5:
            metrics["missed_grain_count"] = fn
            metrics["false_positive_count"] = fp
    metrics.setdefault("missed_grain_count", len(gt_masks))
    metrics.setdefault("false_positive_count", len(pred_masks))
    return metrics


def count_over_segmentation(gt_masks: list[np.ndarray], pred_masks: list[np.ndarray]) -> int:
    count = 0
    for gt_mask in gt_masks:
        gt_area = np.count_nonzero(gt_mask)
        if gt_area == 0:
            continue
        covering = 0
        for pred_mask in pred_masks:
            intersection = np.logical_and(gt_mask > 0, pred_mask > 0).sum()
            if intersection / gt_area >= OVERLAP_RATIO_THRESHOLD:
                covering += 1
        if covering >= 2:
            count += 1
    return count


def count_under_segmentation(gt_masks: list[np.ndarray], pred_masks: list[np.ndarray]) -> int:
    count = 0
    for pred_mask in pred_masks:
        covered_gt = 0
        for gt_mask in gt_masks:
            gt_area = np.count_nonzero(gt_mask)
            if gt_area == 0:
                continue
            intersection = np.logical_and(gt_mask > 0, pred_mask > 0).sum()
            if intersection / gt_area >= OVERLAP_RATIO_THRESHOLD:
                covered_gt += 1
        if covered_gt >= 2:
            count += 1
    return count


def collect_image_records(args: argparse.Namespace) -> list[ImageRecord]:
    if args.dataset:
        split = args.split or "valid"
        image_dir = Path(args.dataset) / split / "images"
        label_dir = Path(args.dataset) / split / "labels"
        records = _records_from_dirs(image_dir, label_dir, split)
    else:
        records = _records_from_dirs(Path(args.images), Path(args.labels), "flat")
    if args.max_images and args.max_images > 0:
        return records[: args.max_images]
    return records


def summarize_rows(rows: list[dict], profiles: list[str]) -> list[dict]:
    summary_rows = []
    for profile in profiles:
        profile_rows = [row for row in rows if row["profile"] == profile]
        if not profile_rows:
            continue
        item = {
            "profile": profile,
            "images": len(profile_rows),
            "total_gt": sum(int(row["num_gt"]) for row in profile_rows),
            "total_pred": sum(int(row["num_pred"]) for row in profile_rows),
            "over_segmentation_count": sum(int(row["over_segmentation_count"]) for row in profile_rows),
            "under_segmentation_count": sum(int(row["under_segmentation_count"]) for row in profile_rows),
            "missed_grain_count": sum(int(row["missed_grain_count"]) for row in profile_rows),
            "false_positive_count": sum(int(row["false_positive_count"]) for row in profile_rows),
        }
        for threshold in IOU_THRESHOLDS:
            suffix = _threshold_suffix(threshold)
            tp = sum(int(row[f"tp_{suffix}"]) for row in profile_rows)
            fp = sum(int(row[f"fp_{suffix}"]) for row in profile_rows)
            fn = sum(int(row[f"fn_{suffix}"]) for row in profile_rows)
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            item[f"precision_{suffix}"] = precision
            item[f"recall_{suffix}"] = recall
            item[f"f1_{suffix}"] = f1
            item[f"mean_iou_{suffix}"] = _matched_mean(profile_rows, f"mean_iou_{suffix}", f"tp_{suffix}")
            item[f"tp_{suffix}"] = tp
            item[f"fp_{suffix}"] = fp
            item[f"fn_{suffix}"] = fn
        summary_rows.append(item)
    return summary_rows


def draw_evaluation_overlay(
    image_rgb: np.ndarray,
    gt_masks: list[np.ndarray],
    pred_masks: list[np.ndarray],
    matches: list[InstanceMatch],
    metrics: dict,
) -> np.ndarray:
    overlay = image_rgb.copy()
    matched_gt = {match.gt_index for match in matches}
    matched_pred = {match.pred_index for match in matches}
    over_gt = _over_segmented_gt_indices(gt_masks, pred_masks)
    under_pred = _under_segmented_pred_indices(gt_masks, pred_masks)
    for index, mask in enumerate(gt_masks):
        color = (255, 196, 0) if index in over_gt else (0, 220, 0)
        if index not in matched_gt:
            color = (255, 0, 0)
        _draw_mask_contours(overlay, mask, color, thickness=2)
    for index, mask in enumerate(pred_masks):
        color = (255, 128, 0) if index in under_pred else (0, 160, 255)
        if index not in matched_pred:
            color = (255, 0, 255)
        if index in matched_pred:
            color = (0, 255, 255)
        _draw_mask_contours(overlay, mask, color, thickness=2)
    label = (
        f"GT {len(gt_masks)} Pred {len(pred_masks)} "
        f"TP50 {metrics.get('tp_50', 0)} FP50 {metrics.get('fp_50', 0)} FN50 {metrics.get('fn_50', 0)}"
    )
    cv2.putText(overlay, label, (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 4, cv2.LINE_AA)
    cv2.putText(overlay, label, (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 2, cv2.LINE_AA)
    return overlay


def _records_from_dirs(image_dir: Path, label_dir: Path, split: str) -> list[ImageRecord]:
    if not image_dir.exists():
        return []
    records = []
    for image_path in sorted(image_dir.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            continue
        records.append(ImageRecord(image_path=image_path, label_path=label_dir / f"{image_path.stem}.txt", split=split))
    return records


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RiceVision-QI predictions against YOLO segmentation labels.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--images", type=Path)
    source.add_argument("--dataset", type=Path)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--split", default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", default="clean_background")
    parser.add_argument("--profiles", nargs="+")
    parser.add_argument("--max-images", type=int, default=0, help="Limit the number of images for quick evaluation.")
    parser.add_argument(
        "--max-processing-dimension",
        type=int,
        default=0,
        help="Override profile max_processing_dimension for faster high-resolution evaluation.",
    )
    parser.add_argument("--no-overlays", action="store_true", help="Do not save per-image overlay visualizations.")
    parser.add_argument("--no-contact-sheet", action="store_true", help="Do not generate contact_sheet_eval.jpg.")
    parser.add_argument(
        "--save-debug-limit",
        type=int,
        default=-1,
        help="Save at most N overlay images. Default -1 saves all overlays.",
    )
    parser.add_argument(
        "--eval-mask-max-dimension",
        type=int,
        default=0,
        help="Downscale GT and prediction masks for faster IoU matching. Default 0 keeps original size.",
    )
    args = parser.parse_args(argv)
    if args.images and not args.labels:
        parser.error("--labels is required when --images is used")
    if args.profiles is None and args.profile:
        args.profiles = [args.profile]
    return args


def _threshold_suffix(threshold: float) -> str:
    return str(int(round(threshold * 100)))


def _should_save_overlay(args: argparse.Namespace, saved_overlay_count: int) -> bool:
    if args.no_overlays:
        return False
    if args.save_debug_limit < 0:
        return True
    return saved_overlay_count < args.save_debug_limit


def _matched_mean(rows: list[dict], mean_key: str, tp_key: str) -> float:
    total_tp = sum(int(row[tp_key]) for row in rows)
    if total_tp == 0:
        return 0.0
    weighted = sum(float(row[mean_key]) * int(row[tp_key]) for row in rows)
    return weighted / total_tp


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})


def _write_summary(path: Path, args: argparse.Namespace, records: list[ImageRecord], summary_rows: list[dict]) -> None:
    dataset_path = args.dataset if args.dataset else args.images
    split = args.split or ("flat" if args.images else "valid")
    lines = [
        "# YOLO Segmentation Evaluation Summary",
        "",
        f"- Dataset path: `{dataset_path}`",
        f"- Split: `{split}`",
        f"- Profiles: {', '.join(row['profile'] for row in summary_rows)}",
        f"- Image count: {len(records)}",
        "",
        "## Profile Summary",
        "",
        "| Profile | GT | Pred | Precision@0.5 | Recall@0.5 | F1@0.5 | Precision@0.75 | Recall@0.75 | F1@0.75 | Over-seg | Under-seg | Missed | FP |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['profile']} | {row['total_gt']} | {row['total_pred']} | "
            f"{row['precision_50']:.4f} | {row['recall_50']:.4f} | {row['f1_50']:.4f} | "
            f"{row['precision_75']:.4f} | {row['recall_75']:.4f} | {row['f1_75']:.4f} | "
            f"{row['over_segmentation_count']} | {row['under_segmentation_count']} | "
            f"{row['missed_grain_count']} | {row['false_positive_count']} |"
        )
    lines.extend(["", "## Interpretation", ""])
    if summary_rows:
        best = max(summary_rows, key=lambda row: (row["f1_50"], row["recall_50"], row["precision_50"]))
        lines.append(f"- Best current profile by F1@0.5: `{best['profile']}`.")
        if best["under_segmentation_count"] > best["over_segmentation_count"]:
            lines.append("- Main failure mode: under-segmentation, where one predicted region covers multiple GT grains.")
        elif best["over_segmentation_count"] > best["under_segmentation_count"]:
            lines.append("- Main failure mode: over-segmentation, where one GT grain is split into multiple predictions.")
        else:
            lines.append("- Main failure mode: mixed or inconclusive from simple overlap diagnostics.")
        lines.append("- Recommendation: continue OpenCV tuning only if hardware/background is fixed and separable.")
        lines.append("- Recommendation: start YOLO-seg training when instance-level robustness on dense grains becomes the priority.")
    else:
        lines.append("- No images were evaluated.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_contact_sheet(path: Path, overlay_paths: list[Path], max_images: int = 24) -> None:
    selected = overlay_paths[:max_images]
    if not selected:
        blank = np.zeros((200, 300, 3), dtype=np.uint8)
        _save_rgb_image(path, blank)
        return
    thumbs = []
    thumb_w, thumb_h = 360, 360
    for overlay_path in selected:
        image = cv2.imdecode(np.fromfile(str(overlay_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        thumbs.append(_resize_with_padding(image, thumb_w, thumb_h))
    if not thumbs:
        return
    cols = min(4, len(thumbs))
    rows = int(np.ceil(len(thumbs) / cols))
    sheet = np.full((rows * thumb_h, cols * thumb_w, 3), 245, dtype=np.uint8)
    for index, thumb in enumerate(thumbs):
        row = index // cols
        col = index % cols
        sheet[row * thumb_h : (row + 1) * thumb_h, col * thumb_w : (col + 1) * thumb_w] = thumb
    _save_rgb_image(path, sheet)


def _resize_with_padding(image: np.ndarray, width: int, height: int) -> np.ndarray:
    scale = min(width / image.shape[1], height / image.shape[0])
    new_size = (max(1, int(image.shape[1] * scale)), max(1, int(image.shape[0] * scale)))
    resized = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    y = (height - resized.shape[0]) // 2
    x = (width - resized.shape[1]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def _draw_mask_contours(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], thickness: int) -> None:
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(image, contours, -1, color, thickness)


def _over_segmented_gt_indices(gt_masks: list[np.ndarray], pred_masks: list[np.ndarray]) -> set[int]:
    indices = set()
    for index, gt_mask in enumerate(gt_masks):
        gt_area = np.count_nonzero(gt_mask)
        if gt_area == 0:
            continue
        covering = sum(
            np.logical_and(gt_mask > 0, pred_mask > 0).sum() / gt_area >= OVERLAP_RATIO_THRESHOLD
            for pred_mask in pred_masks
        )
        if covering >= 2:
            indices.add(index)
    return indices


def _under_segmented_pred_indices(gt_masks: list[np.ndarray], pred_masks: list[np.ndarray]) -> set[int]:
    indices = set()
    for index, pred_mask in enumerate(pred_masks):
        covered = 0
        for gt_mask in gt_masks:
            gt_area = np.count_nonzero(gt_mask)
            if gt_area and np.logical_and(gt_mask > 0, pred_mask > 0).sum() / gt_area >= OVERLAP_RATIO_THRESHOLD:
                covered += 1
        if covered >= 2:
            indices.add(index)
    return indices


def _save_rgb_image(path: Path, image_rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    cv2.imencode(path.suffix, image_bgr)[1].tofile(str(path))


if __name__ == "__main__":
    raise SystemExit(main())

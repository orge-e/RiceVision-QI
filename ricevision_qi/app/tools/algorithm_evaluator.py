from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np

from ricevision_qi.app.core.config import load_profile_config
from ricevision_qi.app.core.image_io import SUPPORTED_IMAGE_EXTENSIONS, read_image
from ricevision_qi.app.core.pipeline import run_detection


CSV_FIELDS = [
    "image_name",
    "profile",
    "num_instances",
    "foreground_area_ratio",
    "mean_area",
    "median_area",
    "mean_aspect_ratio",
    "small_object_count",
    "large_object_count",
    "possible_merge_count",
    "possible_split_count",
    "border_artifact_count",
    "runtime_ms",
    "overlay_path",
    "mask_path",
]


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = _resolve_images(Path(args.input))

    rows = []
    for image_path in image_paths:
        loaded = read_image(image_path)
        for profile in args.profiles:
            cfg = load_profile_config(profile)
            start = time.perf_counter()
            result = run_detection(loaded.image_rgb, cfg)
            runtime_ms = (time.perf_counter() - start) * 1000.0
            profile_dir = output_dir / profile
            profile_dir.mkdir(parents=True, exist_ok=True)
            overlay_path = profile_dir / f"{image_path.stem}_overlay.jpg"
            _save_rgb_image(overlay_path, result.overlay_image)
            mask_path = ""
            if args.save_masks:
                mask_dir = output_dir / "masks" / profile
                mask_dir.mkdir(parents=True, exist_ok=True)
                mask_file = mask_dir / f"{image_path.stem}.png"
                _save_mask(mask_file, result.debug_results.get("cleaned_mask"))
                mask_path = str(mask_file)
            rows.append(_result_row(image_path.name, profile, result, runtime_ms, overlay_path, mask_path))

    _write_csv(output_dir / "algorithm_eval_results.csv", rows)
    _write_report(output_dir / "algorithm_eval_report.md", rows, args.profiles, image_paths)
    _write_contact_sheet(output_dir / "contact_sheet_by_profile.jpg", rows)
    print(f"Algorithm evaluation finished: {output_dir}")
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RiceVision-QI algorithms across image folders.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", dest="profiles", action="append")
    parser.add_argument("--profiles", nargs="+")
    parser.add_argument("--save-masks", action="store_true", help="Save binary prediction masks per image/profile.")
    args = parser.parse_args(argv)
    if not args.profiles:
        args.profiles = ["clean_background"]
    return args


def _resolve_images(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        return []
    return sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )


def _result_row(image_name: str, profile: str, result, runtime_ms: float, overlay_path: Path, mask_path: str = "") -> dict:
    areas = np.array([grain.features.get("area_px", grain.area_px) for grain in result.grains], dtype=float)
    aspects = np.array([grain.features.get("aspect_ratio", 0.0) for grain in result.grains], dtype=float)
    backend_stats = result.backend_stats or {}
    return {
        "image_name": image_name,
        "profile": profile,
        "num_instances": len(result.grains),
        "foreground_area_ratio": backend_stats.get("foreground_area_ratio", 0.0),
        "mean_area": float(np.mean(areas)) if areas.size else 0.0,
        "median_area": float(np.median(areas)) if areas.size else 0.0,
        "mean_aspect_ratio": float(np.mean(aspects)) if aspects.size else 0.0,
        "small_object_count": sum(1 for grain in result.grains if grain.status == "filtered_small"),
        "large_object_count": sum(1 for grain in result.grains if grain.status == "filtered_large"),
        "possible_merge_count": sum(1 for grain in result.grains if grain.status == "possible_merged"),
        "possible_split_count": sum(1 for grain in result.grains if grain.status == "possible_split"),
        "border_artifact_count": sum(1 for grain in result.grains if grain.status == "border_artifact"),
        "runtime_ms": float(runtime_ms),
        "overlay_path": str(overlay_path),
        "mask_path": mask_path,
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_report(path: Path, rows: list[dict], profiles: list[str], images: list[Path]) -> None:
    lines = [
        "# Algorithm Evaluation Report",
        "",
        f"- Images evaluated: {len(images)}",
        f"- Profiles: {', '.join(profiles)}",
        "",
        "## Profile Averages",
        "",
    ]
    if not rows:
        lines.extend(
            [
                "No supported images were found in the input folder.",
                "",
                "Recommendation: add JPG, PNG, BMP, TIF, or TIFF images and rerun evaluation.",
            ]
        )
        path.write_text("\n".join(lines), encoding="utf-8")
        return

    for profile in profiles:
        profile_rows = [row for row in rows if row["profile"] == profile]
        if not profile_rows:
            lines.append(f"- {profile}: no rows")
            continue
        lines.append(
            "- "
            + profile
            + f": avg_instances={_mean(profile_rows, 'num_instances'):.2f}, "
            + f"avg_foreground={_mean(profile_rows, 'foreground_area_ratio'):.4f}, "
            + f"avg_possible_split={_mean(profile_rows, 'possible_split_count'):.2f}, "
            + f"avg_possible_merge={_mean(profile_rows, 'possible_merge_count'):.2f}, "
            + f"avg_border_artifacts={_mean(profile_rows, 'border_artifact_count'):.2f}"
        )
    lines.extend(
        [
            "",
            "## Observations",
            "",
            "- High possible_split_count suggests over-segmentation or noisy masks.",
            "- High possible_merge_count suggests touching grains were not separated.",
            "- High border_artifact_count suggests ROI or acquisition background should be improved.",
            "- If all OpenCV profiles are unstable on stacked grains, instance segmentation models such as YOLO-seg should be evaluated after annotation and training.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_contact_sheet(path: Path, rows: list[dict]) -> None:
    tile_w, tile_h = 280, 230
    if not rows:
        canvas = np.full((tile_h, tile_w, 3), 245, dtype=np.uint8)
        cv2.putText(canvas, "No images", (30, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2, cv2.LINE_AA)
        _save_rgb_image(path, canvas)
        return
    cols = min(3, len(rows))
    rows_count = int(np.ceil(len(rows) / cols))
    canvas = np.full((rows_count * tile_h, cols * tile_w, 3), 245, dtype=np.uint8)
    for index, row in enumerate(rows):
        image = _read_rgb(Path(row["overlay_path"]))
        if image is None:
            continue
        thumb = _letterbox(image, tile_w, tile_h - 38)
        y = (index // cols) * tile_h
        x = (index % cols) * tile_w
        canvas[y : y + thumb.shape[0], x : x + thumb.shape[1]] = thumb
        label = f"{row['profile']} n={row['num_instances']}"
        cv2.putText(canvas, label, (x + 8, y + tile_h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)
    _save_rgb_image(path, canvas)


def _read_rgb(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        return None
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _letterbox(image: np.ndarray, width: int, height: int) -> np.ndarray:
    scale = min(width / image.shape[1], height / image.shape[0])
    new_size = (max(1, int(image.shape[1] * scale)), max(1, int(image.shape[0] * scale)))
    resized = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    canvas = np.full((height, width, 3), 235, dtype=np.uint8)
    y = (height - resized.shape[0]) // 2
    x = (width - resized.shape[1]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def _save_rgb_image(path: Path, image_rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode(path.suffix, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))[1].tofile(str(path))


def _save_mask(path: Path, mask: np.ndarray | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if mask is None or mask.size == 0:
        mask = np.zeros((1, 1), dtype=np.uint8)
    binary = np.where(mask > 0, 255, 0).astype(np.uint8)
    cv2.imencode(path.suffix, binary)[1].tofile(str(path))


def _mean(rows: list[dict], key: str) -> float:
    values = [float(row[key]) for row in rows]
    return float(np.mean(values)) if values else 0.0


if __name__ == "__main__":
    raise SystemExit(main())

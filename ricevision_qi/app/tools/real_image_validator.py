from __future__ import annotations

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from ricevision_qi.app.core.config import load_profile_config
from ricevision_qi.app.core.image_io import SUPPORTED_IMAGE_EXTENSIONS, read_image
from ricevision_qi.app.core.pipeline import run_detection


DEFAULT_PROFILES = ["yolo_seg_rice_v8s", "yolo_seg_rice_v8m"]

CSV_FIELDS = [
    "image_name",
    "profile",
    "width",
    "height",
    "num_instances",
    "normal_count",
    "broken_count",
    "defective_count",
    "impurity_count",
    "unknown_count",
    "valid_count",
    "false_positive_count",
    "possible_split_count",
    "possible_merged_count",
    "border_artifact_count",
    "mean_area_px",
    "mean_length_px",
    "mean_width_px",
    "mean_aspect_ratio",
    "runtime_ms",
    "backend_message",
    "overlay_path",
    "mask_path",
]


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    output_root = Path(args.output)
    run_dir = output_root / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    image_paths = _resolve_images(Path(args.input), recursive=args.recursive)
    if args.max_images > 0:
        image_paths = image_paths[: args.max_images]

    rows: list[dict] = []
    overlay_paths: list[Path] = []
    saved_debug_count = 0
    for image_index, image_path in enumerate(image_paths, start=1):
        try:
            loaded = read_image(image_path)
        except Exception as exc:  # pragma: no cover - defensive CLI boundary
            rows.append(_failed_row(image_path, args.profiles[0], str(exc)))
            continue

        height, width = loaded.image_rgb.shape[:2]
        for profile in args.profiles:
            print(
                f"[{image_index}/{len(image_paths)}][{profile}] validating {image_path.name}, size={width}x{height}",
                flush=True,
            )
            cfg = load_profile_config(profile)
            if args.max_processing_dimension > 0:
                cfg["max_processing_dimension"] = args.max_processing_dimension
            start = time.perf_counter()
            result = run_detection(loaded.image_rgb, cfg)
            runtime_ms = (time.perf_counter() - start) * 1000.0

            overlay_path = run_dir / "overlays" / profile / f"{image_path.stem}_overlay.jpg"
            _save_rgb_image(overlay_path, result.overlay_image)
            overlay_paths.append(overlay_path)

            mask_path = ""
            if args.save_masks:
                mask_path_obj = run_dir / "masks" / profile / f"{image_path.stem}.png"
                _save_mask(mask_path_obj, result.debug_results.get("cleaned_mask"))
                mask_path = str(mask_path_obj)

            if args.save_debug_limit < 0 or saved_debug_count < args.save_debug_limit:
                _save_debug_views(run_dir / "debug_views" / profile / image_path.stem, result.debug_results)
                saved_debug_count += 1

            rows.append(
                _result_row(
                    image_name=image_path.name,
                    profile=profile,
                    width=width,
                    height=height,
                    result=result,
                    runtime_ms=runtime_ms,
                    overlay_path=overlay_path,
                    mask_path=mask_path,
                )
            )
            print(
                f"[{image_index}/{len(image_paths)}][{profile}] done, instances={len(result.grains)}, "
                f"time={runtime_ms / 1000.0:.2f}s",
                flush=True,
            )

    _write_csv(run_dir / "real_validation_results.csv", rows)
    _write_report(run_dir / "real_validation_report.md", rows, image_paths, args.profiles)
    if not args.no_contact_sheet:
        _write_contact_sheet(run_dir / "contact_sheet_real_validation.jpg", rows)
    print(f"Real image validation finished: {run_dir}", flush=True)
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate RiceVision-QI validation outputs for real acquisition images.")
    parser.add_argument("--input", type=Path, required=True, help="Folder containing real acquisition images.")
    parser.add_argument("--output", type=Path, default=Path("samples/output/real_validation"))
    parser.add_argument("--profiles", nargs="+", default=DEFAULT_PROFILES)
    parser.add_argument("--recursive", action="store_true", help="Read images recursively from the input folder.")
    parser.add_argument("--max-images", type=int, default=0, help="Limit image count for a quick review run.")
    parser.add_argument(
        "--max-processing-dimension",
        type=int,
        default=0,
        help="Override profile max_processing_dimension. 0 keeps the profile default.",
    )
    parser.add_argument("--save-masks", action="store_true", help="Save binary masks for each profile and image.")
    parser.add_argument(
        "--save-debug-limit",
        type=int,
        default=0,
        help="Save intermediate debug views for at most N profile-image pairs. Use -1 to save all.",
    )
    parser.add_argument("--no-contact-sheet", action="store_true")
    return parser.parse_args(argv)


def _resolve_images(input_dir: Path, recursive: bool) -> list[Path]:
    if not input_dir.exists():
        return []
    iterator = input_dir.rglob("*") if recursive else input_dir.iterdir()
    images = sorted(
        path
        for path in iterator
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )
    return images


def _result_row(
    image_name: str,
    profile: str,
    width: int,
    height: int,
    result,
    runtime_ms: float,
    overlay_path: Path,
    mask_path: str,
) -> dict:
    grains = result.grains
    summary = result.summary
    areas = np.array([grain.features.get("area_px", grain.area_px) for grain in grains], dtype=float)
    lengths = np.array([grain.features.get("length_px", 0.0) for grain in grains], dtype=float)
    widths = np.array([grain.features.get("width_px", 0.0) for grain in grains], dtype=float)
    aspects = np.array([grain.features.get("aspect_ratio", 0.0) for grain in grains], dtype=float)
    return {
        "image_name": image_name,
        "profile": profile,
        "width": width,
        "height": height,
        "num_instances": len(grains),
        "normal_count": summary.get("normal_count", 0),
        "broken_count": summary.get("broken_count", 0),
        "defective_count": summary.get("defective_count", 0),
        "impurity_count": summary.get("impurity_count", 0),
        "unknown_count": summary.get("unknown_count", 0),
        "valid_count": sum(1 for grain in grains if grain.status == "valid"),
        "false_positive_count": sum(1 for grain in grains if grain.status == "false_positive"),
        "possible_split_count": sum(1 for grain in grains if grain.status == "possible_split"),
        "possible_merged_count": sum(1 for grain in grains if grain.status == "possible_merged"),
        "border_artifact_count": sum(1 for grain in grains if grain.status == "border_artifact"),
        "mean_area_px": _mean(areas),
        "mean_length_px": _mean(lengths),
        "mean_width_px": _mean(widths),
        "mean_aspect_ratio": _mean(aspects),
        "runtime_ms": float(runtime_ms),
        "backend_message": (result.backend_stats or {}).get("message", ""),
        "overlay_path": str(overlay_path),
        "mask_path": mask_path,
    }


def _failed_row(image_path: Path, profile: str, message: str) -> dict:
    row = {field: "" for field in CSV_FIELDS}
    row.update({"image_name": image_path.name, "profile": profile, "backend_message": message})
    return row


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})


def _write_report(path: Path, rows: list[dict], images: list[Path], profiles: list[str]) -> None:
    lines = [
        "# Real Image Validation Report",
        "",
        f"- Images found: {len(images)}",
        f"- Profiles: {', '.join(profiles)}",
        "- Ground truth annotations: not used",
        "",
        "This report is for visual review of real acquisition images. It is not a quantitative accuracy report.",
        "",
        "## Profile Summary",
        "",
    ]
    if not rows:
        lines.extend(
            [
                "No supported images were found.",
                "",
                "Put JPG, PNG, BMP, TIF, or TIFF images in the input folder and rerun this tool.",
            ]
        )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    lines.append("| Profile | Images | Avg Instances | Avg Runtime ms | Backend Messages |")
    lines.append("|---|---:|---:|---:|---:|")
    for profile in profiles:
        profile_rows = [row for row in rows if row.get("profile") == profile and row.get("num_instances") != ""]
        if not profile_rows:
            lines.append(f"| {profile} | 0 | 0.00 | 0.0 | 0 |")
            continue
        messages = sum(1 for row in profile_rows if row.get("backend_message"))
        lines.append(
            f"| {profile} | {len(profile_rows)} | "
            f"{_row_mean(profile_rows, 'num_instances'):.2f} | "
            f"{_row_mean(profile_rows, 'runtime_ms'):.1f} | {messages} |"
        )

    lines.extend(
        [
            "",
            "## Manual Review Checklist",
            "",
            "- Check missed grains: visible grains without outlines.",
            "- Check false positives: background, shadows, container edges, or reflections with outlines.",
            "- Check mask boundary quality: outlines should follow grain edges closely enough for measurement.",
            "- Check merged instances: multiple touching grains inside one outline.",
            "- Check split instances: one grain divided into multiple outlines.",
            "",
            "## Next Steps",
            "",
            "- Keep the best current profile as the default inference baseline only after visual review.",
            "- Select 20-30 failure-heavy images for high-quality mask annotation.",
            "- Retrain after real-scene annotation, then rerun this validation on the same image set.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_contact_sheet(path: Path, rows: list[dict]) -> None:
    valid_rows = [row for row in rows if row.get("overlay_path")]
    tile_w, tile_h = 320, 280
    if not valid_rows:
        canvas = np.full((tile_h, tile_w, 3), 245, dtype=np.uint8)
        cv2.putText(canvas, "No validation images", (24, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2, cv2.LINE_AA)
        _save_rgb_image(path, canvas)
        return
    cols = min(3, len(valid_rows))
    rows_count = int(np.ceil(len(valid_rows) / cols))
    canvas = np.full((rows_count * tile_h, cols * tile_w, 3), 245, dtype=np.uint8)
    for index, row in enumerate(valid_rows):
        image = _read_rgb(Path(row["overlay_path"]))
        if image is None:
            continue
        thumb = _letterbox(image, tile_w, tile_h - 44)
        y = (index // cols) * tile_h
        x = (index % cols) * tile_w
        canvas[y : y + thumb.shape[0], x : x + thumb.shape[1]] = thumb
        label = f"{row['profile']} n={row['num_instances']}"
        cv2.putText(canvas, label, (x + 8, y + tile_h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)
    _save_rgb_image(path, canvas)


def _save_debug_views(output_dir: Path, debug_results: dict) -> None:
    for name, image in debug_results.items():
        if image is None:
            continue
        output_path = output_dir / f"{name}.png"
        if isinstance(image, np.ndarray) and image.ndim == 2:
            _save_mask(output_path, image)
        else:
            _save_rgb_image(output_path, image)


def _save_mask(path: Path, mask: np.ndarray | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if mask is None or mask.size == 0:
        mask = np.zeros((1, 1), dtype=np.uint8)
    binary = np.where(mask > 0, 255, 0).astype(np.uint8)
    cv2.imencode(path.suffix, binary)[1].tofile(str(path))


def _save_rgb_image(path: Path, image_rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if image_rgb.ndim == 2:
        image_rgb = cv2.cvtColor(image_rgb.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    cv2.imencode(path.suffix, cv2.cvtColor(image_rgb.astype(np.uint8), cv2.COLOR_RGB2BGR))[1].tofile(str(path))


def _read_rgb(path: Path) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        return None
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _letterbox(image: np.ndarray, width: int, height: int) -> np.ndarray:
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(
        image,
        (max(1, int(image.shape[1] * scale)), max(1, int(image.shape[0] * scale))),
        interpolation=cv2.INTER_AREA,
    )
    canvas = np.full((height, width, 3), 235, dtype=np.uint8)
    y = (height - resized.shape[0]) // 2
    x = (width - resized.shape[1]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def _mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if values.size else 0.0


def _row_mean(rows: list[dict], key: str) -> float:
    values = [float(row[key]) for row in rows if row.get(key) != ""]
    return float(np.mean(values)) if values else 0.0


if __name__ == "__main__":
    raise SystemExit(main())

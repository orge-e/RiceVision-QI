from __future__ import annotations

import argparse
import csv
import itertools
import math
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from ricevision_qi.app.core.config import merge_config
from ricevision_qi.app.core.image_io import SUPPORTED_IMAGE_EXTENSIONS, read_image
from ricevision_qi.app.core.pipeline import run_detection


GRID_KEYS = (
    "segmentation_method",
    "kernel_size",
    "open_iter",
    "close_iter",
    "min_area",
    "watershed_dist_ratio",
    "fill_holes",
)

DEFAULT_GRID = {
    "segmentation_method": ["gray_otsu", "hsv_threshold", "lab_threshold", "adaptive_threshold"],
    "kernel_size": [3, 5, 7],
    "open_iter": [0, 1, 2],
    "close_iter": [1, 2, 3],
    "min_area": [50, 100, 200, 300],
    "watershed_dist_ratio": [0.25, 0.30, 0.35, 0.40, 0.45],
    "fill_holes": [True, False],
}


@dataclass
class TuningRun:
    params_id: str
    image_name: str
    params: dict
    stats: dict
    score: float
    failure_reasons: list[str]
    overlay_path: Path
    mask_path: Path


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    images = _resolve_images(args.image, args.input)
    if not images:
        raise SystemExit("No supported images found.")

    output_root = Path(args.output)
    run_dir = output_root / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    grid = load_tuning_grid(Path(args.grid) if args.grid else _default_grid_path())
    combinations = list(iter_parameter_grid(grid))
    if args.max_combinations:
        combinations = combinations[: args.max_combinations]
    if not combinations:
        raise SystemExit("No parameter combinations to evaluate.")

    all_runs = []
    for image_index, image_path in enumerate(images, start=1):
        image_dir = run_dir / f"image_{image_index:03d}"
        image_dir.mkdir(parents=True, exist_ok=True)
        all_runs.extend(
            tune_image(
                image_path=image_path,
                image_dir=image_dir,
                parameter_sets=combinations,
                roi=args.roi,
            )
        )

    all_runs.sort(key=lambda run: run.score, reverse=True)
    _write_results_csv(run_dir / "tuning_results.csv", all_runs)
    _write_results_csv(run_dir / "instance_stats.csv", all_runs)
    if all_runs:
        _write_best_params(run_dir / "best_params.yaml", all_runs[0])
        _write_report(run_dir / "tuning_report.md", all_runs)
        _write_contact_sheets(run_dir, all_runs, kind="overlay")
        _write_contact_sheets(run_dir, all_runs, kind="mask")

    print(f"Tuning finished: {run_dir}")
    return 0


def tune_image(
    image_path: Path,
    image_dir: Path,
    parameter_sets: Iterable[dict],
    roi: tuple[int, int, int, int] | None = None,
) -> list[TuningRun]:
    loaded = read_image(image_path)
    image_rgb = _crop_roi(loaded.image_rgb, roi)
    _save_rgb_image(image_dir / "original_thumbnail.jpg", _thumbnail(image_rgb, 900))

    runs = []
    for index, params in enumerate(parameter_sets, start=1):
        params_id = f"params_{index:04d}"
        cfg = merge_config(params)
        start = time.perf_counter()
        result = run_detection(image_rgb, cfg)
        runtime_ms = (time.perf_counter() - start) * 1000.0
        stats = calculate_segmentation_stats(result, image_rgb, cfg, runtime_ms)
        score, reasons = score_segmentation_result(stats, cfg)

        prefix = image_dir / params_id
        _save_rgb_image(prefix.with_name(f"{params_id}_overlay.jpg"), result.overlay_image)
        _save_rgb_image(prefix.with_name(f"{params_id}_illumination_corrected.jpg"), result.debug_results["illumination_corrected"])
        _save_mask(prefix.with_name(f"{params_id}_initial_mask.png"), result.debug_results["initial_mask"])
        mask_path = prefix.with_name(f"{params_id}_cleaned_mask.png")
        _save_mask(mask_path, result.debug_results["cleaned_mask"])
        _save_markers(prefix.with_name(f"{params_id}_markers.png"), result.debug_results["watershed_markers"])
        _save_rgb_image(prefix.with_name(f"{params_id}_debug.jpg"), _debug_mosaic(result.debug_results))

        run = TuningRun(
            params_id=params_id,
            image_name=image_path.name,
            params=params.copy(),
            stats=stats,
            score=score,
            failure_reasons=reasons,
            overlay_path=prefix.with_name(f"{params_id}_overlay.jpg"),
            mask_path=mask_path,
        )
        runs.append(run)

    runs.sort(key=lambda run: run.score, reverse=True)
    _write_results_csv(image_dir / "tuning_results.csv", runs)
    _write_results_csv(image_dir / "instance_stats.csv", runs)
    _write_contact_sheets(image_dir, runs, kind="overlay")
    _write_contact_sheets(image_dir, runs, kind="mask")
    return runs


def calculate_segmentation_stats(result, image_rgb: np.ndarray, config: dict, runtime_ms: float) -> dict:
    mask = result.debug_results.get("cleaned_mask")
    image_area = float(image_rgb.shape[0] * image_rgb.shape[1]) if image_rgb.size else 0.0
    foreground_area = float(np.count_nonzero(mask)) if mask is not None and mask.size else 0.0
    areas = np.array([float(grain.features.get("area_px", grain.area_px)) for grain in result.grains], dtype=float)
    lengths = np.array([float(grain.features.get("length_px", 0.0)) for grain in result.grains], dtype=float)
    widths = np.array([float(grain.features.get("width_px", 0.0)) for grain in result.grains], dtype=float)
    aspects = np.array([float(grain.features.get("aspect_ratio", 0.0)) for grain in result.grains], dtype=float)
    solidities, extents = _shape_ratios(result.grains)
    edge_count = _edge_touching_count(result.grains, image_rgb.shape[:2])

    min_area = float(config.get("min_area", 100))
    max_area = float(config.get("max_area", 100000))
    small_object_count, large_object_count = _component_size_counts(mask, min_area, max_area)
    split_count = int(np.sum((areas > 0) & (areas < max(min_area * 1.8, np.median(areas) * 0.35 if areas.size else min_area))))
    merge_count = int(np.sum((areas > min(max_area, np.median(areas) * 2.8 if areas.size else max_area))))

    return {
        "num_instances": int(len(result.grains)),
        "foreground_area_ratio": _safe_ratio(foreground_area, image_area),
        "mean_instance_area": _mean(areas),
        "median_instance_area": _median(areas),
        "std_instance_area": _std(areas),
        "mean_length": _mean(lengths),
        "mean_width": _mean(widths),
        "mean_aspect_ratio": _mean(aspects),
        "median_aspect_ratio": _median(aspects),
        "small_object_count": int(small_object_count),
        "large_object_count": int(large_object_count),
        "edge_touching_count": int(edge_count),
        "mean_solidity": _mean(np.array(solidities, dtype=float)),
        "mean_extent": _mean(np.array(extents, dtype=float)),
        "split_suspect_count": split_count,
        "merge_suspect_count": merge_count,
        "runtime_ms": float(runtime_ms),
    }


def score_segmentation_result(stats: dict, config: dict | None = None) -> tuple[float, list[str]]:
    reasons = []
    if int(stats.get("num_instances", 0)) <= 0:
        return 0.0, ["no instances detected"]

    score = 100.0
    foreground_ratio = float(stats.get("foreground_area_ratio", 0.0))
    if foreground_ratio < 0.005:
        score -= 35.0
        reasons.append("foreground area too small")
    elif foreground_ratio > 0.45:
        score -= 35.0
        reasons.append("foreground area too large")

    mean_aspect = float(stats.get("mean_aspect_ratio", 0.0))
    if mean_aspect < 1.6:
        score -= 18.0
        reasons.append("mean aspect ratio too low")
    elif mean_aspect > 8.0:
        score -= 12.0
        reasons.append("mean aspect ratio too high")

    mean_solidity = float(stats.get("mean_solidity", 0.0))
    if mean_solidity and mean_solidity < 0.78:
        score -= 15.0
        reasons.append("contours have low solidity")

    mean_extent = float(stats.get("mean_extent", 0.0))
    if mean_extent and mean_extent < 0.25:
        score -= 8.0
        reasons.append("low bbox extent")

    for key, weight in (
        ("small_object_count", 1.4),
        ("large_object_count", 4.0),
        ("edge_touching_count", 2.0),
        ("split_suspect_count", 1.2),
        ("merge_suspect_count", 4.0),
    ):
        count = int(stats.get(key, 0))
        if count:
            score -= min(22.0, count * weight)
            reasons.append(f"{key}={count}")

    runtime_ms = float(stats.get("runtime_ms", 0.0))
    if runtime_ms > 2500:
        score -= min(8.0, (runtime_ms - 2500.0) / 1000.0)
        reasons.append("runtime is high")

    return max(0.0, round(score, 3)), reasons


def load_tuning_grid(path: Path | None = None) -> dict:
    if path and path.exists():
        parsed = _parse_simple_yaml(path)
        grid = parsed.get("grid", parsed)
        return {key: grid.get(key, DEFAULT_GRID[key]) for key in GRID_KEYS}
    return DEFAULT_GRID.copy()


def iter_parameter_grid(grid: dict) -> Iterable[dict]:
    values = [grid.get(key, DEFAULT_GRID[key]) for key in GRID_KEYS]
    for combo in itertools.product(*values):
        params = dict(zip(GRID_KEYS, combo))
        params["method"] = params["segmentation_method"]
        yield params


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run segmentation parameter tuning for RiceVision-QI images.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path, help="Single image path.")
    source.add_argument("--input", type=Path, help="Directory containing images.")
    parser.add_argument("--output", type=Path, required=True, help="Output tuning directory.")
    parser.add_argument("--grid", type=Path, help="YAML file with scan ranges.")
    parser.add_argument("--roi", nargs=4, type=int, metavar=("X", "Y", "W", "H"), help="Optional ROI.")
    parser.add_argument("--max-combinations", type=int, default=0, help="Optional cap for quick exploratory runs.")
    return parser.parse_args(argv)


def _resolve_images(image: Path | None, input_dir: Path | None) -> list[Path]:
    if image:
        return [image]
    return sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )


def _crop_roi(image: np.ndarray, roi: tuple[int, int, int, int] | None) -> np.ndarray:
    if not roi:
        return image
    x, y, w, h = roi
    height, width = image.shape[:2]
    x = max(0, min(x, width))
    y = max(0, min(y, height))
    w = max(0, min(w, width - x))
    h = max(0, min(h, height - y))
    if w == 0 or h == 0:
        raise ValueError("ROI is outside image bounds.")
    return image[y : y + h, x : x + w].copy()


def _shape_ratios(grains) -> tuple[list[float], list[float]]:
    solidities = []
    extents = []
    for grain in grains:
        contour_area = float(cv2.contourArea(grain.contour))
        hull = cv2.convexHull(grain.contour)
        hull_area = float(cv2.contourArea(hull))
        x, y, w, h = grain.bbox
        bbox_area = float(w * h)
        solidities.append(_safe_ratio(contour_area, hull_area))
        extents.append(_safe_ratio(contour_area, bbox_area))
    return solidities, extents


def _component_size_counts(mask: np.ndarray | None, min_area: float, max_area: float) -> tuple[int, int]:
    if mask is None or mask.size == 0:
        return 0, 0
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    small = 0
    large = 0
    for label in range(1, num_labels):
        area = float(stats[label, cv2.CC_STAT_AREA])
        small += int(area < min_area)
        large += int(area > max_area)
    return small, large


def _edge_touching_count(grains, shape: tuple[int, int]) -> int:
    height, width = shape
    count = 0
    for grain in grains:
        x, y, w, h = grain.bbox
        if x <= 0 or y <= 0 or x + w >= width - 1 or y + h >= height - 1:
            count += 1
    return count


def _write_results_csv(path: Path, runs: list[TuningRun]) -> None:
    fieldnames = [
        "rank",
        "image_name",
        "params_id",
        "score",
        *GRID_KEYS,
        "max_area",
        "failure_reasons",
        "overlay_path",
        "mask_path",
        "num_instances",
        "foreground_area_ratio",
        "mean_instance_area",
        "median_instance_area",
        "std_instance_area",
        "mean_length",
        "mean_width",
        "mean_aspect_ratio",
        "median_aspect_ratio",
        "small_object_count",
        "large_object_count",
        "edge_touching_count",
        "mean_solidity",
        "mean_extent",
        "split_suspect_count",
        "merge_suspect_count",
        "runtime_ms",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for rank, run in enumerate(runs, start=1):
            row = {
                "rank": rank,
                "image_name": run.image_name,
                "params_id": run.params_id,
                "score": run.score,
                "max_area": merge_config(run.params).get("max_area"),
                "failure_reasons": "; ".join(run.failure_reasons),
                "overlay_path": str(run.overlay_path),
                "mask_path": str(run.mask_path),
            }
            row.update({key: run.params.get(key) for key in GRID_KEYS})
            row.update(run.stats)
            writer.writerow(row)


def _write_best_params(path: Path, run: TuningRun) -> None:
    cfg = merge_config(run.params)
    lines = ["segmentation:"]
    for key in ("method", "kernel_size", "open_iter", "close_iter", "min_area", "max_area", "watershed_dist_ratio", "fill_holes"):
        value = cfg.get(key)
        if key == "method":
            value = run.params.get("segmentation_method", cfg.get("segmentation_method"))
        lines.append(f"  {key}: {_yaml_value(value)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_report(path: Path, runs: list[TuningRun]) -> None:
    best = runs[0]
    lines = [
        "# Segmentation Tuning Report",
        "",
        "## Best Parameters",
        "",
        f"- image: {best.image_name}",
        f"- params_id: {best.params_id}",
        f"- score: {best.score}",
        f"- failure_reasons: {'; '.join(best.failure_reasons) if best.failure_reasons else 'none'}",
        "",
        "```yaml",
        "segmentation:",
    ]
    for key in ("segmentation_method", "kernel_size", "open_iter", "close_iter", "min_area", "watershed_dist_ratio", "fill_holes"):
        lines.append(f"  {key}: {_yaml_value(best.params.get(key))}")
    lines.extend(["```", "", "## Best Statistics", ""])
    for key, value in best.stats.items():
        lines.append(f"- {key}: {_format_number(value)}")

    lines.extend(["", "## Top 10 Parameter Sets", ""])
    for rank, run in enumerate(runs[:10], start=1):
        params = ", ".join(f"{key}={run.params.get(key)}" for key in GRID_KEYS)
        lines.append(f"{rank}. {run.image_name} {run.params_id} score={run.score}: {params}")

    lines.extend(["", "## Low Score Failure Inference", ""])
    for run in runs[-10:]:
        reason = "; ".join(run.failure_reasons) if run.failure_reasons else "low relative score"
        lines.append(f"- {run.image_name} {run.params_id} score={run.score}: {reason}")

    acquisition_note = _acquisition_note(best.stats)
    lines.extend(["", "## Image Acquisition Recommendation", "", acquisition_note, ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_contact_sheets(run_dir: Path, runs: list[TuningRun], kind: str, per_page: int = 20) -> None:
    if not runs:
        return
    selected_path = "overlay_path" if kind == "overlay" else "mask_path"
    for page_index, start in enumerate(range(0, len(runs), per_page), start=1):
        page_runs = runs[start : start + per_page]
        sheet = _make_contact_sheet(page_runs, selected_path)
        suffix = f"{kind}_page_{page_index:03d}.jpg" if len(runs) > per_page else f"{kind}.jpg"
        _save_rgb_image(run_dir / f"contact_sheet_{suffix}", sheet)


def _make_contact_sheet(runs: list[TuningRun], image_attr: str) -> np.ndarray:
    tile_w, tile_h = 320, 260
    cols = 4
    rows = int(math.ceil(len(runs) / cols))
    sheet = np.full((rows * tile_h, cols * tile_w, 3), 245, dtype=np.uint8)
    for index, run in enumerate(runs):
        row = index // cols
        col = index % cols
        path = getattr(run, image_attr)
        image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        thumb = _letterbox(image, tile_w, tile_h - 58)
        y0 = row * tile_h
        x0 = col * tile_w
        sheet[y0 : y0 + thumb.shape[0], x0 : x0 + thumb.shape[1]] = thumb
        label = (
            f"{run.params_id} score={run.score} n={run.stats['num_instances']}\n"
            f"{run.params['segmentation_method']} k={run.params['kernel_size']} "
            f"o={run.params['open_iter']} c={run.params['close_iter']} "
            f"w={run.params['watershed_dist_ratio']}"
        )
        _draw_multiline_text(sheet, label, (x0 + 6, y0 + tile_h - 48))
    return sheet


def _debug_mosaic(debug_results: dict) -> np.ndarray:
    panels = [
        ("original", debug_results["original"]),
        ("corrected", debug_results["illumination_corrected"]),
        ("initial", _gray_to_rgb(debug_results["initial_mask"])),
        ("cleaned", _gray_to_rgb(debug_results["cleaned_mask"])),
        ("markers", _markers_to_rgb(debug_results["watershed_markers"])),
        ("overlay", debug_results["overlay"]),
    ]
    tile_w, tile_h = 280, 220
    canvas = np.full((2 * tile_h, 3 * tile_w, 3), 245, dtype=np.uint8)
    for index, (name, image) in enumerate(panels):
        row = index // 3
        col = index % 3
        tile = _letterbox(image, tile_w, tile_h - 24)
        y0 = row * tile_h
        x0 = col * tile_w
        canvas[y0 : y0 + tile.shape[0], x0 : x0 + tile.shape[1]] = tile
        cv2.putText(canvas, name, (x0 + 8, y0 + tile_h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)
    return canvas


def _save_rgb_image(path: Path, image_rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if image_rgb is None or image_rgb.size == 0:
        image_rgb = np.zeros((1, 1, 3), dtype=np.uint8)
    bgr = cv2.cvtColor(np.ascontiguousarray(image_rgb), cv2.COLOR_RGB2BGR)
    cv2.imencode(path.suffix, bgr)[1].tofile(str(path))


def _save_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if mask is None or mask.size == 0:
        mask = np.zeros((1, 1), dtype=np.uint8)
    cv2.imencode(path.suffix, np.ascontiguousarray(mask.astype(np.uint8)))[1].tofile(str(path))


def _save_markers(path: Path, markers: np.ndarray) -> None:
    _save_rgb_image(path, _markers_to_rgb(markers))


def _markers_to_rgb(markers: np.ndarray) -> np.ndarray:
    if markers is None or markers.size == 0:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    normalized = markers.astype(np.float32)
    normalized -= normalized.min()
    if normalized.max() > 0:
        normalized = normalized / normalized.max() * 255.0
    colored = cv2.applyColorMap(normalized.astype(np.uint8), cv2.COLORMAP_TURBO)
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


def _gray_to_rgb(mask: np.ndarray) -> np.ndarray:
    if mask is None or mask.size == 0:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    return cv2.cvtColor(mask.astype(np.uint8), cv2.COLOR_GRAY2RGB)


def _thumbnail(image: np.ndarray, max_side: int) -> np.ndarray:
    scale = min(1.0, max_side / float(max(image.shape[:2])))
    if scale >= 1.0:
        return image
    return cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)), interpolation=cv2.INTER_AREA)


def _letterbox(image: np.ndarray, width: int, height: int) -> np.ndarray:
    if image.ndim == 2:
        image = _gray_to_rgb(image)
    scale = min(width / image.shape[1], height / image.shape[0])
    new_size = (max(1, int(image.shape[1] * scale)), max(1, int(image.shape[0] * scale)))
    resized = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    canvas = np.full((height, width, 3), 235, dtype=np.uint8)
    y = (height - resized.shape[0]) // 2
    x = (width - resized.shape[1]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def _draw_multiline_text(image: np.ndarray, text: str, origin: tuple[int, int]) -> None:
    x, y = origin
    for line in text.splitlines():
        cv2.putText(image, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (20, 20, 20), 1, cv2.LINE_AA)
        y += 19


def _parse_simple_yaml(path: Path) -> dict:
    result: dict[str, dict | list | str | int | float | bool] = {}
    current_section: str | None = None
    current_key: str | None = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        if indent == 0 and stripped.endswith(":"):
            current_section = stripped[:-1]
            result[current_section] = {}
            current_key = None
            continue
        target = result[current_section] if current_section else result
        if stripped.startswith("-") and current_key:
            target[current_key].append(_coerce_yaml_value(stripped[1:].strip()))
            continue
        if ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value == "":
            target[key] = []
            current_key = key
        elif value.startswith("[") and value.endswith("]"):
            target[key] = [_coerce_yaml_value(item.strip()) for item in value[1:-1].split(",") if item.strip()]
            current_key = None
        else:
            target[key] = _coerce_yaml_value(value)
            current_key = None
    return result


def _coerce_yaml_value(value: str):
    cleaned = value.strip().strip('"').strip("'")
    if cleaned.lower() in {"true", "false"}:
        return cleaned.lower() == "true"
    if cleaned.lower() in {"none", "null"}:
        return None
    try:
        return float(cleaned) if "." in cleaned else int(cleaned)
    except ValueError:
        return cleaned


def _yaml_value(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    return str(value)


def _format_number(value) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _acquisition_note(stats: dict) -> str:
    if stats.get("edge_touching_count", 0) > 3 or stats.get("foreground_area_ratio", 0) > 0.45:
        return "The image likely contains strong background or bag-edge interference. Use ROI cropping or improve lighting/background separation."
    if stats.get("merge_suspect_count", 0) > 3:
        return "Several large merged regions remain. Spread grains more sparsely or reduce overlap before capture."
    if stats.get("split_suspect_count", 0) > 8:
        return "Many small fragments were detected. Improve illumination uniformity and reduce glare before capture."
    return "The best result does not show severe acquisition issues, but visual review of contact sheets is still required."


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if values.size else 0.0


def _median(values: np.ndarray) -> float:
    return float(np.median(values)) if values.size else 0.0


def _std(values: np.ndarray) -> float:
    return float(np.std(values)) if values.size else 0.0


def _default_grid_path() -> Path:
    return Path(__file__).resolve().parents[1] / "config" / "tuning_grid.yaml"


if __name__ == "__main__":
    raise SystemExit(main())

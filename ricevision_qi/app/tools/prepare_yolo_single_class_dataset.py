from __future__ import annotations

import argparse
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SPLITS = ("train", "valid", "test")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    stats = {}
    empty_labels = []
    bad_labels = []
    for split in SPLITS:
        split_stats = _convert_split(src, dst, split, empty_labels, bad_labels)
        stats[split] = split_stats

    _write_data_yaml(Path(args.data_yaml), dst, args.class_name)
    _write_report(dst / "single_class_conversion_report.md", src, dst, stats, empty_labels, bad_labels)
    print(f"Single-class YOLO dataset prepared: {dst}", flush=True)
    return 0


def _convert_split(src: Path, dst: Path, split: str, empty_labels: list[str], bad_labels: list[str]) -> dict:
    src_images = src / split / "images"
    src_labels = src / split / "labels"
    dst_images = dst / split / "images"
    dst_labels = dst / split / "labels"
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)

    image_count = 0
    label_count = 0
    for image_path in sorted(src_images.iterdir() if src_images.exists() else []):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        _link_or_copy(image_path, dst_images / image_path.name)
        image_count += 1

    for label_path in sorted(src_labels.glob("*.txt") if src_labels.exists() else []):
        converted_lines, had_bad = convert_label_text(label_path.read_text(encoding="utf-8"))
        if not converted_lines:
            empty_labels.append(str(label_path))
        if had_bad:
            bad_labels.append(str(label_path))
        (dst_labels / label_path.name).write_text("\n".join(converted_lines) + ("\n" if converted_lines else ""), encoding="utf-8")
        label_count += 1

    return {"images": image_count, "labels": label_count}


def convert_label_text(text: str) -> tuple[list[str], bool]:
    converted = []
    had_bad = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 7 or (len(parts) - 1) % 2:
            had_bad = True
            continue
        converted.append(" ".join(["0", *parts[1:]]))
    return converted, had_bad


def _link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        dst.hardlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def _write_data_yaml(path: Path, dataset_root: Path, class_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset_path = dataset_root.resolve().as_posix()
    path.write_text(
        "\n".join(
            [
                f"path: {dataset_path}",
                "train: train/images",
                "val: valid/images",
                "test: test/images",
                "names:",
                f"  0: {class_name}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _write_report(
    path: Path,
    src: Path,
    dst: Path,
    stats: dict[str, dict],
    empty_labels: list[str],
    bad_labels: list[str],
) -> None:
    lines = [
        "# Single-Class YOLO Segmentation Dataset Conversion Report",
        "",
        f"- Source dataset: `{src}`",
        f"- Output dataset: `{dst}`",
        f"- Empty labels: {len(empty_labels)}",
        f"- Bad labels: {len(bad_labels)}",
        f"- All class IDs merged to 0: {'yes' if not bad_labels else 'yes, with bad lines skipped'}",
        "",
        "| Split | Images | Labels |",
        "|---|---:|---:|",
    ]
    for split in SPLITS:
        lines.append(f"| {split} | {stats[split]['images']} | {stats[split]['labels']} |")
    if empty_labels:
        lines.extend(["", "## Empty Label Files", *[f"- `{item}`" for item in empty_labels[:50]]])
    if bad_labels:
        lines.extend(["", "## Bad Label Files", *[f"- `{item}`" for item in bad_labels[:50]]])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a YOLO segmentation dataset to single-class rice_grain labels.")
    parser.add_argument("--src", required=True, type=Path)
    parser.add_argument("--dst", required=True, type=Path)
    parser.add_argument("--class-name", default="rice_grain")
    parser.add_argument("--data-yaml", required=True, type=Path)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())

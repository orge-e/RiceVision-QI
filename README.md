# RiceVision-QI

RiceVision-QI is a vision-based rice grain quality inspection system with a PySide6 GUI and OpenCV pipeline.
The current desktop prototype includes image import, OpenCV-based grain segmentation, debug views, parameter controls, and a command-line segmentation tuning tool.

## Requirements

- Windows
- Python 3.10+
- PySide6
- OpenCV

## Install

```powershell
conda create -n RiceVision-QI python=3.11 -y
conda activate RiceVision-QI
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run

```powershell
conda activate RiceVision-QI
python run_app.py
```

## Segmentation Tuning Tool

Run a grid search on one image:

```powershell
conda activate RiceVision-QI
python -m app.tools.segmentation_tuner --image samples/raw/test.jpg --output samples/output/tuning
```

Run on a folder:

```powershell
python -m app.tools.segmentation_tuner --input samples/raw --output samples/output/tuning
```

Limit processing to a region of interest:

```powershell
python -m app.tools.segmentation_tuner --image samples/raw/test.jpg --output samples/output/tuning --roi 100 200 800 600
```

The default scan grid is in `ricevision_qi/app/config/tuning_grid.yaml`. For quick experiments, create a smaller YAML grid and pass it with `--grid path/to/grid.yaml`.

The tool writes timestamped output under `samples/output/tuning`, including intermediate masks, overlays, contact sheets, `tuning_results.csv`, `instance_stats.csv`, `best_params.yaml`, and `tuning_report.md`.

## ROI Selection

The desktop image viewer supports mouse-drag ROI selection. ROI coordinates use original image pixels. When an ROI exists, detection runs only inside that region, but contours, bounding boxes, overlays, and exported CSV rows are mapped back to the original image coordinate system.

Use **清除ROI** to remove the current selection and run detection on the full image.

## Algorithm Profiles

Profiles live in `ricevision_qi/app/config/profiles/`:

- `clean_background.yaml`
- `plastic_bag_rice.yaml`
- `dense_rice_cluster.yaml`
- `debug_fast.yaml`

The UI exposes these profiles through the **Algorithm Profile** dropdown. If a profile cannot be loaded, the application falls back to the default configuration.

## Segmentation Backends

The pipeline selects a backend through profile/config key `segmentation.backend`:

- `opencv_threshold`: threshold, morphology, connected components.
- `opencv_watershed`: threshold, morphology, distance transform, watershed.
- `edge_enhanced_watershed`: local contrast enhancement before watershed.
- `yolo_seg`: optional YOLO segmentation backend. It returns a clear status message instead of crashing when `ultralytics` or weights are unavailable.

All backends return `instances`, `debug_results`, and `stats`.

## Batch Algorithm Evaluation

Run profile comparison on a folder:

```powershell
conda activate RiceVision-QI
python -m app.tools.algorithm_evaluator --input samples/raw --output samples/output/eval --profiles clean_background plastic_bag_rice dense_rice_cluster
```

The evaluator writes `algorithm_eval_results.csv`, `algorithm_eval_report.md`, and `contact_sheet_by_profile.jpg`.

## YOLO Segmentation Dataset Evaluation

Use the YOLO segmentation evaluator to compare RiceVision-QI pipeline output against YOLO polygon labels. This is intended for OpenCV pipeline benchmarking now and later YOLO-seg backend evaluation.

The Mendeley Rice-Variety-Classification-System-Dataset is useful for single-grain instance segmentation, counting, and mask quality evaluation. It is not a direct defective-kernel, impurity, chalkiness, or head-rice-yield quality dataset.

Run with flat image and label folders:

```powershell
conda activate RiceVision-QI
python -m app.tools.yolo_seg_evaluator --images samples/eval/mendeley/rice-variety-classification-system/images_for_eval --labels samples/eval/mendeley/rice-variety-classification-system/labels_for_eval --output samples/output/yolo_seg_eval --profile clean_background
```

Run with a YOLO split directory:

```powershell
python -m app.tools.yolo_seg_evaluator --dataset "samples/eval/mendeley/rice-variety-classification-system/Rice Grain Dataset/yolo_dataset" --split valid --output samples/output/yolo_seg_eval --profile clean_background
```

Compare multiple profiles:

```powershell
python -m app.tools.yolo_seg_evaluator --images samples/eval/mendeley/rice-variety-classification-system/images_for_eval --labels samples/eval/mendeley/rice-variety-classification-system/labels_for_eval --output samples/output/yolo_seg_eval --profiles clean_background plastic_bag_rice dense_rice_cluster
```

Each run writes a timestamped directory containing `yolo_seg_eval_results.csv`, `yolo_seg_eval_summary.md`, `overlays/`, and `contact_sheet_eval.jpg`.

## Preparing YOLO Segmentation Dataset

The Mendeley dataset ships with YOLO segmentation labels for multiple rice variety classes. For rice grain instance segmentation, convert all class IDs to a single class named `rice_grain`:

```powershell
conda activate RiceVision-QI
python -m app.tools.prepare_yolo_single_class_dataset --src "samples/eval/mendeley/rice-variety-classification-system/Rice Grain Dataset/yolo_dataset" --dst "samples/eval/mendeley/rice-variety-classification-system/Rice Grain Dataset/yolo_dataset_single_class" --class-name rice_grain --data-yaml "samples/eval/mendeley/rice-variety-classification-system/rice_grain_seg.yaml"
```

The script preserves the original dataset, links or copies images, rewrites labels with class ID `0`, and writes `single_class_conversion_report.md`.

## Training YOLO-seg

YOLO-seg is optional and requires `ultralytics` plus a compatible PyTorch installation:

```powershell
conda run -n RiceVision-QI pip install ultralytics
```

Start with a CPU/GPU smoke test before long training:

```powershell
python -m app.tools.train_yolo_seg --data "samples/eval/mendeley/rice-variety-classification-system/rice_grain_seg.yaml" --model yolo11n-seg.pt --epochs 1 --imgsz 320 --batch 2 --project samples/output/yolo_train_smoke --name rice_grain_yolo_seg_smoke
```

Formal training can use:

```powershell
python -m app.tools.train_yolo_seg --data "samples/eval/mendeley/rice-variety-classification-system/rice_grain_seg.yaml" --model yolo11n-seg.pt --epochs 50 --imgsz 640 --batch 8 --project samples/output/yolo_train --name rice_grain_yolo_seg
```

Training outputs are local artifacts under `samples/output/` and should not be committed.

## Using YOLOSegBackend

The `yolo_seg_rice` profile enables the optional YOLO-seg backend:

```text
ricevision_qi/app/config/profiles/yolo_seg_rice.yaml
```

By default it points to:

```text
samples/output/yolo_train/rice_grain_yolo_seg/weights/best.pt
```

If the weights file or `ultralytics` is missing, the backend returns an empty result with a clear message instead of crashing. OpenCV profiles remain available as baseline backends.

## Comparing OpenCV and YOLO-seg

After training produces `best.pt`, compare the YOLO profile against an OpenCV baseline on a small validation subset first:

```powershell
python -m app.tools.yolo_seg_evaluator --dataset "samples/eval/mendeley/rice-variety-classification-system/Rice Grain Dataset/yolo_dataset" --split valid --output samples/output/yolo_seg_eval_compare --profiles dense_rice_cluster yolo_seg_rice --max-images 10 --max-processing-dimension 768 --save-debug-limit 10
```

The Mendeley dataset is useful for single-grain instance segmentation and counting. It does not directly represent defective kernels, impurities, chalky grains, or other rice quality categories. OpenCV remains the baseline; YOLO-seg is an optional backend for dense instance segmentation.

## Real Image Validation

Use the real image validator when you have acquisition images from the actual inspection setup. This workflow is for visual review, not quantitative accuracy measurement, because it does not require ground-truth masks.

Put real images under:

```text
samples/real_validation/input/
```

Then run:

```powershell
conda activate RiceVision-QI
python -m app.tools.real_image_validator --input samples/real_validation/input --output samples/output/real_validation --profiles yolo_seg_rice_v8s yolo_seg_rice_v8m --save-debug-limit 5
```

Outputs are written to a timestamped folder under `samples/output/real_validation/`:

- `real_validation_results.csv`
- `real_validation_report.md`
- `contact_sheet_real_validation.jpg`
- `overlays/{profile}/*_overlay.jpg`
- optional `masks/{profile}/*.png` when `--save-masks` is used

Review the contact sheet for missed grains, false positives, merged instances, split instances, and poor mask boundaries. Select 20-30 failure-heavy images for high-quality mask annotation before retraining.

## Optional YOLO-seg Backend

YOLO-seg is optional. The default project still uses OpenCV. If `ultralytics` is not installed or no weights are configured, the YOLO backend returns an empty result with a status message instead of crashing. Real YOLO-seg use requires annotated data and trained weights.

## Manual Review Workflow

After detection, click a row or an instance on the overlay to highlight it. The right panel shows instance details. The first manual review version supports:

- changing the class;
- marking an instance as `false_positive`;
- syncing the table;
- exporting manual edits to CSV.

Mask editing, manual contour drawing, instance merging, and instance splitting are not implemented in this version.

## Current Features

- Main window title: `RiceVision-QI: Rice Quality Inspection System`
- Four-region layout:
  - Left control panel
  - Center image viewer
  - Right statistics panel
  - Bottom single-grain result table
- Detection type selector:
  - Rice Defective Kernel Inspection
  - Paddy Husked Rice Yield
  - Head Rice Yield
- Image import for JPG, PNG, BMP, TIF, and TIFF files
- Aspect-ratio-preserving image display
- OpenCV preprocessing, segmentation, measurement, simple rule classification, and overlay visualization
- Debug views for original image, illumination correction, masks, watershed markers, and final overlay
- UI controls for segmentation parameters
- Result table and row-to-image traceability highlighting

## Current Placeholders

- GB/T standard calculations
- Report export
- Hardware calibration and physical unit conversion
- Formal quality grading rules for defective kernels, impurities, chalky grains, and head-rice yield

## Test

```powershell
python -m pytest
```

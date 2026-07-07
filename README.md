# RiceVision-QI

RiceVision-QI is a desktop skeleton for a rice quality visual inspection system.
Phase 1 focuses on project structure, the PySide6 user interface, and image import/display.

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

## Phase 1 Features

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
- Placeholder statistics and result table
- Placeholder detection and report export messages

## Current Placeholders

- Detection pipeline
- GB/T standard calculations
- Report export
- Single-grain measurement and classification results

## Test

```powershell
python -m pytest
```

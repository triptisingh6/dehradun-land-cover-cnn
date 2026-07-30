# Land Cover Classification & Change Detection — Dehradun (2016 vs 2025)

CNN-based land cover classification and change detection over the
Dehradun / Doon Valley region using Sentinel-2 imagery, comparing winter
2015-16 against winter 2024-25 to map urban expansion and vegetation loss.

## Why this project

Reuses CNN transfer-learning experience (MobileNetV2 / EfficientNetB0,
Grad-CAM explainability) from a prior medical imaging project, applied to a
new domain: multispectral satellite imagery. Built as a self-contained
geospatial deep learning project targeting remote sensing research
(IIRS/ISRO-relevant workflow: GEE data acquisition -> custom CNN -> QGIS-
ready outputs).

## Pipeline

| Step | Script | What it does |
|---|---|---|
| 1 | `gee/01_export_from_gee.py` | Pulls Sentinel-2 imagery for two winters, cloud-masks it, loads ESA WorldCover as labels, exports labeled training patches + full composite images |
| 2 | `training/02_prepare_dataset.py` | Parses exported TFRecords into a train/val/test numpy dataset |
| 3 | `training/03_train_cnn.py` | Trains a baseline CNN + MobileNetV2 + EfficientNetB0 (transfer learning, 6-band input) |
| 4 | `training/04_evaluate.py` | Accuracy, Cohen's Kappa, confusion matrix, per-class report, Grad-CAM |
| 5 | `change_detection/05_classify_and_compare.py` | Classifies the full 2016 and 2025 scenes, produces land cover maps, a change map, and a transition matrix |

## Classes

Vegetation, Water, Urban/Built-up, Bare Soil, Agriculture — mapped from
ESA WorldCover's 11 classes.

## Study area

Dehradun / Doon Valley, Uttarakhand (`77.90, 30.20` to `78.20, 30.45`),
chosen for its mix of forest, agriculture, urban expansion, and river
systems, and its relevance to Dehradun-based remote sensing research.

## Setup

```bash
pip install -r requirements.txt
```

Step 1 requires a free Google Earth Engine account:
https://code.earthengine.google.com/register

Steps 2-5 run locally or in Colab once the Step 1 exports are downloaded
from Google Drive into a local `data/` folder.

## Results

See `results/` for confusion matrix, Grad-CAM examples, both land cover
maps, the change map, and the transition matrix (after running the full
pipeline). See `report/` for the full write-up.

## Known limitations

- Labels come from ESA WorldCover (2021), not hand-verified ground truth —
  trained/evaluated only on the 2016-period imagery; the 2025 map is
  inference-only, since no matching 2025 label layer exists.
- Train/val/test split is a documented random shuffle rather than a strict
  spatial block split; a production system would bucket by grid cell to
  fully eliminate spatial-autocorrelation leakage.

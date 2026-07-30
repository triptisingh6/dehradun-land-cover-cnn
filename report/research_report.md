# Land Cover Classification and Change Detection in the Dehradun Region Using Sentinel-2 Imagery and a Convolutional Neural Network

**Author:** [Your Name]
**Date:** July 2026

---

## Abstract

This study presents an end-to-end deep learning pipeline for land cover classification and multi-temporal change detection over the Dehradun / Doon Valley region, Uttarakhand, India. Sentinel-2 multispectral imagery was acquired via Google Earth Engine for three time periods — winter 2015–16, winter 2019–20, and winter 2024–25 — and classified into five land cover classes (Vegetation, Water, Urban/Built-up, Bare Soil, Agriculture) using a convolutional neural network (CNN) trained on patches labeled from the ESA WorldCover product. The trained model achieved an overall test accuracy of 61.3% and a Cohen's Kappa of 0.492, indicating moderate classification agreement. Post-classification change detection revealed a consistent increase in urban/built-up area and a corresponding decline in agricultural land across the nine-year study period, with approximately 19.5% of the study area changing classification between 2016 and 2025. Known sources of classification error, particularly confusion involving the Water class, are identified and discussed. The pipeline, code, and outputs are made available as a reproducible workflow.

---

## 1. Introduction

Land cover in rapidly urbanizing Himalayan foothill regions such as Dehradun is changing quickly, driven by population growth, infrastructure development, and encroachment on forest and agricultural land. Monitoring this change is important for urban planning, forest conservation, and disaster risk management, and satellite remote sensing offers a scalable way to do this over time.

This project explores whether a convolutional neural network — a deep learning approach more commonly used for medical image classification and general computer vision — can be trained directly on multispectral satellite imagery to classify land cover and detect change, using free and openly accessible tools (Google Earth Engine, Sentinel-2, ESA WorldCover). Rather than using Earth Engine's built-in classical classifiers (e.g. Random Forest), a custom CNN was trained from scratch on labeled Sentinel-2 image patches, in order to build hands-on experience with the full geospatial deep learning pipeline: data acquisition, label sourcing, patch extraction, model training, accuracy assessment, and change detection.

The specific objectives of this study were to:
1. Classify land cover in the Dehradun region into five classes using Sentinel-2 imagery and a custom-trained CNN.
2. Assess classification accuracy using standard remote sensing metrics (overall accuracy, per-class precision/recall/F1, Cohen's Kappa).
3. Apply the trained model to three time periods (2016, 2020, 2025) and quantify land cover change through post-classification comparison.

---

## 2. Study Area

The study area is a rectangular region covering Dehradun city and the surrounding Doon Valley, Uttarakhand (approximately 77.90°E–78.20°E, 30.20°N–30.45°N). This area was selected because it contains a representative mix of all five target land cover classes within a single scene: dense forest cover in the Shivalik foothills to the north, agricultural land across the valley floor, a rapidly expanding urban core, and river/water features including the Rispana and Song rivers. The region has also seen well-documented urban expansion over the past decade, providing a meaningful change signal for this study.

---

## 3. Data and Methods

### 3.1 Satellite Imagery

Sentinel-2 Level-2A surface reflectance imagery (`COPERNICUS/S2_SR_HARMONIZED`) was retrieved via Google Earth Engine for three winter periods, chosen to minimize cloud cover and monsoon-related atmospheric interference:

| Period | Date Range |
|---|---|
| 2016 | 1 Nov 2015 – 28 Feb 2016 |
| 2020 | 1 Nov 2019 – 28 Feb 2020 |
| 2025 | 1 Nov 2024 – 28 Feb 2025 |

For each period, images with cloudy-pixel-percentage below 20% were filtered and cloud-masked using the Sentinel-2 Cloud Probability collection (pixels with >40% cloud probability were masked out). The remaining images were combined using a per-pixel median composite. Six spectral bands were retained: Blue (B2), Green (B3), Red (B4), Near-Infrared (B8), and two Shortwave Infrared bands (B11, B12), all scaled to surface reflectance (0–1).

### 3.2 Label Source

Ground truth labels were derived from ESA WorldCover v200 (2021), a global 10 m land cover product. WorldCover's 11 original classes were remapped to five target classes:

| Target Class | WorldCover Source Classes |
|---|---|
| Vegetation | Tree cover, Shrubland, Grassland, Moss/lichen |
| Water | Water bodies, Wetland |
| Urban | Built-up |
| Bare Soil | Bare/sparse vegetation, Snow/ice |
| Agriculture | Cropland |

Because WorldCover reflects 2021 conditions, it was used only to generate training/validation/test labels paired with the 2016 imagery (the closest available period), under the assumption that land cover distribution — though not exact area — was reasonably representative for sampling purposes. The 2020 and 2025 maps were produced through inference only, using the model trained on 2016-paired data.

### 3.3 Training Data Preparation

Stratified random sampling was used to select 300 points per class (1,500 points total) within the study area, using WorldCover as the stratification layer. For each sample point, a 33×33 pixel patch (approximately 330 m × 330 m at 10 m resolution) was extracted from the 2016 Sentinel-2 composite across all six bands, together with the corresponding class label. Patches were exported from Earth Engine as TFRecord files.

The resulting dataset (1,500 labeled patches, perfectly balanced with 300 examples per class) was split into training (70%, n=1,050), validation (15%, n=225), and test (15%, n=225) sets using a random shuffle with a fixed seed.

*Note on data splitting: A random shuffle split was used here for simplicity rather than a strict spatial block split. Because neighboring patches in remote sensing data can be spatially autocorrelated, a random split may allow some information leakage between sets and can modestly inflate reported accuracy relative to a stricter spatial holdout. This is discussed further in Section 6.*

### 3.4 CNN Architecture and Training

A convolutional neural network was built and trained from scratch (rather than using transfer learning) to accommodate the six-band, non-RGB input structure of the satellite patches. The architecture consisted of three convolutional blocks (32, 64, and 128 filters respectively, each followed by max-pooling), a global average pooling layer, a 128-unit dense layer with dropout (0.3), and a final softmax layer over five classes. The model was trained for 20 epochs using the Adam optimizer and sparse categorical cross-entropy loss, with class weighting applied to account for any residual class imbalance.

### 3.5 Accuracy Assessment

Model performance was evaluated on the held-out test set (n=225) using overall accuracy, per-class precision/recall/F1-score, a confusion matrix, and Cohen's Kappa — the standard chance-corrected agreement metric used in remote sensing accuracy assessment.

### 3.6 Full-Scene Classification and Change Detection

The trained model was applied to the full Sentinel-2 composite for all three time periods using a sliding-window approach (33×33 patches, stride of 16 pixels, majority voting across overlapping windows) to produce complete classified land cover rasters. Change detection was performed via post-classification comparison: the classified rasters for each pair of time periods were compared pixel-by-pixel to generate a transition matrix (in hectares) and visual change maps.

---

## 4. Results

### 4.1 Classification Accuracy

On the held-out test set, the model achieved an overall accuracy of **61.33%** and a Cohen's Kappa of **0.492**, which falls in the "moderate agreement" range on the standard Landis and Koch classification scale.

**Per-class performance:**

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| Vegetation | 0.84 | 0.65 | 0.73 | 40 |
| Water | 0.64 | 0.31 | 0.42 | 52 |
| Urban | 0.83 | 0.63 | 0.72 | 46 |
| Bare Soil | 0.60 | 0.60 | 0.60 | 47 |
| Agriculture | 0.45 | 0.97 | 0.61 | 40 |

Vegetation and Urban classes showed the strongest performance, with high precision (0.83–0.84) indicating that when the model predicted these classes, it was usually correct. The Water class showed the weakest recall (0.31), indicating the model frequently failed to identify true water pixels. The Agriculture class showed the opposite pattern — very high recall (0.97) but low precision (0.45) — suggesting the model over-predicted Agriculture as a "default" class when uncertain, likely due to spectral similarity with Vegetation and Bare Soil in the visible/NIR range.

*(Insert: confusion matrix figure — `results/confusion_matrix.png`)*
*(Insert: Grad-CAM example figure — `results/gradcam_examples.png`)*

### 4.2 Land Cover Maps

Full-scene classification was performed for all three time periods. The resulting maps show forest cover concentrated in the Shivalik foothills, urban/built-up area concentrated in and around Dehradun city center, and agricultural land distributed across the valley floor — consistent with known regional geography.

*(Insert: `results/landcover_2016.png`, `results/landcover_2020.png`, `results/landcover_2025.png`)*

### 4.3 Land Cover Area by Class

| Class | 2016 (ha) | 2020 (ha) | 2025 (ha) |
|---|---|---|---|
| Vegetation | ~69,200 | ~72,300 | ~66,500 |
| Water | ~5,800 | ~5,700 | ~9,000 |
| Urban | ~5,800 | ~7,900 | ~8,700 |
| Bare Soil | ~2,000 | ~900 | ~1,800 |
| Agriculture | ~10,400 | ~6,200 | ~7,000 |

*(Insert: `results/area_change_3years_barchart.png`)*

Urban area increased consistently across all three time points (2016 → 2020 → 2025), while Agriculture declined sharply between 2016 and 2020 before stabilizing. This pattern is consistent with continued urban expansion into agricultural land over the study period.

### 4.4 Change Detection

Overall, **19.52%** of the study area changed classified land cover class between 2016 and 2025.

**Key transitions, 2016 → 2025 (hectares):**

| Transition | Area (ha) |
|---|---|
| Vegetation → Urban | 2,178.6 |
| Agriculture → Urban | 2,449.9 |
| Vegetation → Agriculture | 2,071.0 |
| Agriculture → Vegetation | 4,631.0 |

**Key transitions, 2016 → 2020 (hectares):**

| Transition | Area (ha) |
|---|---|
| Vegetation → Urban | 2,035.2 |
| Agriculture → Urban | 691.2 |
| Agriculture → Vegetation | 4,679.7 |

**Key transitions, 2020 → 2025 (hectares):**

| Transition | Area (ha) |
|---|---|
| Agriculture → Urban | 3,668.5 |
| Urban → Urban (stable) | 5,422.1 |
| Vegetation → Agriculture | 7,536.6 |

*(Insert: `results/change_map.png`)*

The combined Vegetation→Urban and Agriculture→Urban transitions across the full period total approximately 4,600 hectares converted to built-up land between 2016 and 2025 — a substantial urban expansion footprint for the region. Notably, the rate of Agriculture→Urban conversion accelerated in the second half of the study period (691 ha in 2016–2020 vs. 3,669 ha in 2020–2025), suggesting urban growth pressure on agricultural land intensified in recent years.

---

## 5. Discussion

The results demonstrate that a CNN trained from scratch on a relatively small, free-to-acquire dataset (1,500 labeled patches) can produce a usable, if imperfect, land cover classification and detect a plausible urban expansion signal consistent with known regional development patterns. The Urban and Agriculture area trends in particular — showing steady urban growth and farmland decline — align with the expected real-world change story for the Doon Valley over this period.

However, several limitations affect the precision of these results and should be considered when interpreting them:

**Water-class confusion.** The model's weakest performance was on the Water class (31% recall), likely due to spectral similarity between water, shadows, and certain urban surface materials (e.g. wet pavement, some rooftop materials) at 10 m resolution. This is visible in the transition matrices, where implausibly large areas (e.g. ~1,800–2,300 ha) appear to transition between Urban and Water — a pattern more consistent with model misclassification than genuine land cover change. Apparent Water-class transitions should therefore be interpreted with caution; the Vegetation, Urban, and Agriculture transitions are considered more reliable indicators of genuine change.

**Agriculture as a "default" prediction.** The Agriculture class showed high recall but low precision, indicating the model frequently defaulted to this label when uncertain, likely due to spectral overlap with Vegetation in the visible and near-infrared bands. This may partially inflate the apparent Vegetation→Agriculture transition observed between 2020 and 2025.

**Label source mismatch.** Training labels were derived from ESA WorldCover (2021), the closest available date to the 2016 imagery but not an exact contemporaneous match. Some label noise is therefore expected, and the 2020/2025 classifications are inference-only, without independent validation against ground truth for those specific years.

**Train/test split methodology.** A random (rather than spatially blocked) train/validation/test split was used. Because nearby patches can be spatially correlated, this may modestly overstate the reported accuracy relative to what would be achieved on a fully independent spatial holdout. A stricter spatial block split is recommended for future iterations of this work.

**Model capacity.** A relatively simple CNN was trained from scratch rather than using transfer learning from a pretrained backbone, in order to accommodate the six-band (non-RGB) input. Accuracy could likely be improved with a larger training set, additional training epochs, or an adapted transfer-learning approach using pretrained RGB backbones (e.g. MobileNetV2 or EfficientNetB0) with modified input layers.

---

## 6. Conclusion

This study developed and applied a complete, reproducible geospatial deep learning pipeline — from Sentinel-2 data acquisition through CNN-based classification to multi-temporal change detection — for the Dehradun region, achieving moderate classification accuracy (61.3%, Kappa = 0.492) and identifying a consistent, quantifiable pattern of urban expansion into agricultural and forested land between 2016 and 2025. While classification accuracy leaves room for improvement, particularly for the Water class, the resulting change detection results align with known regional development trends, demonstrating that the overall approach is sound and extensible.

Future work could improve on this baseline by: incorporating a larger and more spatially independent training dataset; testing transfer-learning approaches with adapted pretrained backbones; incorporating SAR (radar) data to improve water and built-up discrimination independent of cloud cover; and validating the 2020 and 2025 classifications against higher-resolution reference imagery or field-verified ground truth specific to those years.

---

## Appendix: Reproducibility

All code, the trained model, and full results (land cover maps for 2016/2020/2025, change maps, transition matrices, confusion matrix, and Grad-CAM visualizations) are available in the accompanying GitHub repository. The pipeline can be re-run or extended to additional time periods or study regions by modifying the area-of-interest geometry and date ranges in the Earth Engine export script.

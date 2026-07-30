"""
STEP 5 — Color in the ENTIRE 2016 map and the ENTIRE 2025 map, then compare
them to make the change map.

Input: dehradun_2016_composite.tif and dehradun_2025_composite.tif
       (downloaded from Google Drive after Step 1's export finished)
Output: dehradun_2016_landcover.tif, dehradun_2025_landcover.tif,
        dehradun_change_map.tif, a transition table, and a bar chart.

Needs: pip install rasterio
"""

import numpy as np
import rasterio
from rasterio.windows import Window
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

CLASS_NAMES = ["Vegetation", "Water", "Urban", "Bare Soil", "Agriculture"]
CLASS_COLORS = {
    0: (34, 139, 34),    # vegetation - green
    1: (30, 144, 255),   # water - blue
    2: (220, 20, 60),    # urban - red
    3: (210, 180, 140),  # bare soil - tan
    4: (255, 215, 0),    # agriculture - yellow
}
PATCH_SIZE = 64
MODEL_PATH = "models/mobilenetv2_landcover_best.keras"

model = tf.keras.models.load_model(MODEL_PATH)


def classify_full_image(tif_path, output_path, stride=32):
    """Slide a PATCH_SIZE x PATCH_SIZE window across the whole satellite
    image, classify each window, and stitch the results into one big
    land-cover map. Overlapping windows (stride < PATCH_SIZE) and averaging
    votes at the edges gives smoother results than non-overlapping tiles."""

    with rasterio.open(tif_path) as src:
        img = src.read()                      # (bands, height, width)
        profile = src.profile
        img = np.transpose(img, (1, 2, 0))    # (height, width, bands)
        img = np.clip(img, 0, 1)

    h, w, bands = img.shape
    vote_counts = np.zeros((h, w, len(CLASS_NAMES)), dtype=np.int32)

    half = PATCH_SIZE // 2
    for y in range(half, h - half, stride):
        row_patches, row_positions = [], []
        for x in range(half, w - half, stride):
            patch = img[y - half:y + half, x - half:x + half, :]
            if patch.shape[:2] == (PATCH_SIZE, PATCH_SIZE):
                row_patches.append(patch)
                row_positions.append((y, x))

        if not row_patches:
            continue

        preds = model.predict(np.array(row_patches), verbose=0)
        pred_classes = np.argmax(preds, axis=1)

        for (yy, xx), cls in zip(row_positions, pred_classes):
            vote_counts[yy - stride // 2:yy + stride // 2,
                        xx - stride // 2:xx + stride // 2, cls] += 1

    classified = np.argmax(vote_counts, axis=-1).astype(np.uint8)

    profile.update(count=1, dtype="uint8")
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(classified, 1)

    print(f"Saved classified map: {output_path}")
    return classified


def save_colored_png(classified, out_png, title):
    h, w = classified.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, color in CLASS_COLORS.items():
        rgb[classified == cls] = color

    plt.figure(figsize=(8, 8))
    plt.imshow(rgb)
    plt.title(title)
    plt.axis("off")
    handles = [plt.Rectangle((0, 0), 1, 1, color=np.array(c) / 255)
               for c in CLASS_COLORS.values()]
    plt.legend(handles, CLASS_NAMES, loc="lower left", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"Saved preview image: {out_png}")


# ---------------------------------------------------------------------------
# Run classification on both years
# ---------------------------------------------------------------------------
lc_2016 = classify_full_image("data/dehradun_2016_composite.tif",
                               "results/dehradun_2016_landcover.tif")
save_colored_png(lc_2016, "results/dehradun_2016_landcover.png", "Land Cover 2016")

lc_2025 = classify_full_image("data/dehradun_2025_composite.tif",
                               "results/dehradun_2025_landcover.tif")
save_colored_png(lc_2025, "results/dehradun_2025_landcover.png", "Land Cover 2025")

# ---------------------------------------------------------------------------
# Change detection: post-classification comparison
# ---------------------------------------------------------------------------
assert lc_2016.shape == lc_2025.shape, "The two rasters must be the same size/grid"

changed_mask = lc_2016 != lc_2025
print(f"\n{changed_mask.mean():.2%} of pixels changed class between 2016 and 2025")

# Build a transition matrix: rows = 2016 class, cols = 2025 class
transition = np.zeros((5, 5), dtype=np.int64)
for i in range(5):
    for j in range(5):
        transition[i, j] = np.sum((lc_2016 == i) & (lc_2025 == j))

# Convert pixel counts to area in hectares (10m x 10m pixel = 100 sq m = 0.01 ha)
transition_ha = transition * 0.01

df = pd.DataFrame(transition_ha, index=[f"2016_{c}" for c in CLASS_NAMES],
                   columns=[f"2025_{c}" for c in CLASS_NAMES])
df.to_csv("results/transition_matrix_hectares.csv")
print("\nTransition matrix (hectares) saved to results/transition_matrix_hectares.csv")
print(df.round(1))

# Highlight the two most policy-relevant transitions
forest_to_urban = transition_ha[0, 2]
agri_to_urban = transition_ha[4, 2]
print(f"\nForest -> Urban:      {forest_to_urban:,.1f} ha")
print(f"Agriculture -> Urban: {agri_to_urban:,.1f} ha")

# Net area change per class, as a bar chart
area_2016 = np.array([(lc_2016 == i).sum() for i in range(5)]) * 0.01
area_2025 = np.array([(lc_2025 == i).sum() for i in range(5)]) * 0.01

fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(5)
ax.bar(x - 0.2, area_2016, width=0.4, label="2016")
ax.bar(x + 0.2, area_2025, width=0.4, label="2025")
ax.set_xticks(x)
ax.set_xticklabels(CLASS_NAMES, rotation=20)
ax.set_ylabel("Area (hectares)")
ax.set_title("Land Cover Area: 2016 vs 2025")
ax.legend()
plt.tight_layout()
plt.savefig("results/area_change_barchart.png", dpi=150)
print("Saved results/area_change_barchart.png")

# Save a visual change map: gray = unchanged, red-ish = changed, colored by
# what it changed INTO
change_rgb = np.zeros((*lc_2016.shape, 3), dtype=np.uint8)
change_rgb[~changed_mask] = (200, 200, 200)  # unchanged = light gray
for cls, color in CLASS_COLORS.items():
    change_rgb[(lc_2025 == cls) & changed_mask] = color

plt.figure(figsize=(8, 8))
plt.imshow(change_rgb)
plt.title("Change Map (gray = unchanged, colored = changed, color = new class)")
plt.axis("off")
plt.tight_layout()
plt.savefig("results/change_map.png", dpi=150)
print("Saved results/change_map.png")

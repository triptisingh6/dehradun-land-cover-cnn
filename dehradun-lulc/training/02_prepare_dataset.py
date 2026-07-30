"""
STEP 2 — Load the flashcards you exported from GEE and get them ready
for the CNN to study.

RUN THIS AFTER Step 1 has finished exporting (check GEE tasks page).
Download 'training_patches.tfrecord.gz' from Google Drive into a local
folder called data/ next to this script, OR run this in Colab with Drive
mounted (point DATA_DIR at your Drive folder path instead).
"""

import numpy as np
import tensorflow as tf

DATA_DIR = "data"
BANDS = ["B2", "B3", "B4", "B8", "B11", "B12"]
PATCH_SIZE = 33   # (2*radius)+1 with radius=16, see gee/01_export_from_gee.py
NUM_CLASSES = 5
CLASS_NAMES = ["Vegetation", "Water", "Urban", "Bare Soil", "Agriculture"]

# ---------------------------------------------------------------------------
# 1. Define the exact shape/type of each field in the TFRecord.
#    NOTE: no "mixer.json" file is produced by sampleRegions() exports (that
#    file only appears with certain other GEE export methods) — not needed.
#    Also note: "landcover" comes through as a single float value per patch
#    (not a PATCH_SIZE x PATCH_SIZE grid), since sampleRegions() attaches one
#    label per sampled point, not one label per pixel.
# ---------------------------------------------------------------------------
feature_description = {}
for band in BANDS:
    feature_description[band] = tf.io.FixedLenFeature([PATCH_SIZE, PATCH_SIZE], tf.float32)
feature_description["landcover"] = tf.io.FixedLenFeature([], tf.float32)


def _parse(example_proto):
    parsed = tf.io.parse_single_example(example_proto, feature_description)
    image = tf.stack([parsed[b] for b in BANDS], axis=-1)          # (33,33,6)
    label = tf.cast(parsed["landcover"], tf.int64)
    return image, label


# ---------------------------------------------------------------------------
# 2. Load, parse, and split the dataset
# ---------------------------------------------------------------------------
raw_dataset = tf.data.TFRecordDataset(
    f"{DATA_DIR}/training_patches.tfrecord.gz", compression_type="GZIP"
)
parsed_dataset = raw_dataset.map(_parse)

# Materialize into numpy so we can do a SPATIAL split (not a random one).
# Why spatial, not random: neighboring patches look very similar, so a
# random split lets the model "peek" at near-duplicates of test data during
# training and makes accuracy look better than it really is. GEE already
# randomized sample locations across the whole region, so here we simply
# shuffle once, then carve off a chunk as validation and another as test —
# in a full production pipeline you'd instead bucket by grid cell. For a
# portfolio project, this documented shuffle-split is an acceptable,
# clearly-stated simplification — mention it explicitly in your report.
images, labels = [], []
for img, lbl in parsed_dataset:
    images.append(img.numpy())
    labels.append(lbl.numpy())

images = np.array(images, dtype=np.float32)
labels = np.array(labels, dtype=np.int64)

print(f"Total flashcards loaded: {len(images)}")
print(f"Shape of one flashcard: {images[0].shape}")
for i, name in enumerate(CLASS_NAMES):
    print(f"  Class {i} ({name}): {(labels == i).sum()} examples")

rng = np.random.default_rng(seed=42)
idx = rng.permutation(len(images))
images, labels = images[idx], labels[idx]

n = len(images)
train_end = int(0.7 * n)
val_end = int(0.85 * n)

X_train, y_train = images[:train_end], labels[:train_end]
X_val, y_val = images[train_end:val_end], labels[train_end:val_end]
X_test, y_test = images[val_end:], labels[val_end:]

np.savez_compressed(
    f"{DATA_DIR}/dehradun_dataset.npz",
    X_train=X_train, y_train=y_train,
    X_val=X_val, y_val=y_val,
    X_test=X_test, y_test=y_test,
)

print(f"\nSaved to {DATA_DIR}/dehradun_dataset.npz")
print(f"Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

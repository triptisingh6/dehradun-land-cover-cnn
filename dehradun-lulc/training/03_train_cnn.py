"""
STEP 3 — Train the CNN on the flashcards.

Reuses your NeuroVision approach (MobileNetV2 / EfficientNetB0 transfer
learning), adapted for:
  - 6 input channels (Blue,Green,Red,NIR,SWIR1,SWIR2) instead of 3 (RGB MRI)
  - 5 output classes instead of tumor/no-tumor
  - a from-scratch baseline CNN for comparison, same as your NeuroVision report did
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, applications

DATA_DIR = "data"
NUM_CLASSES = 5
CLASS_NAMES = ["Vegetation", "Water", "Urban", "Bare Soil", "Agriculture"]
PATCH_SIZE = 64
NUM_BANDS = 6

data = np.load(f"{DATA_DIR}/dehradun_dataset.npz")
X_train, y_train = data["X_train"], data["y_train"]
X_val, y_val = data["X_val"], data["y_val"]
X_test, y_test = data["X_test"], data["y_test"]

# Simple per-band normalization (reflectance is already ~0-1 from GEE, but
# clip outliers and rescale to be safe)
X_train = np.clip(X_train, 0, 1)
X_val = np.clip(X_val, 0, 1)
X_test = np.clip(X_test, 0, 1)


# ---------------------------------------------------------------------------
# Adapting a 3-channel pretrained backbone to 6 channels:
# we duplicate/average the pretrained RGB filters into the extra bands so
# the network starts from useful edge/texture detectors instead of random
# noise, then let training fine-tune everything.
# ---------------------------------------------------------------------------
def build_transfer_model(backbone_name="MobileNetV2"):
    if backbone_name == "MobileNetV2":
        base = applications.MobileNetV2(
            input_shape=(PATCH_SIZE, PATCH_SIZE, 3), include_top=False, weights="imagenet"
        )
    elif backbone_name == "EfficientNetB0":
        base = applications.EfficientNetB0(
            input_shape=(PATCH_SIZE, PATCH_SIZE, 3), include_top=False, weights="imagenet"
        )
    else:
        raise ValueError(backbone_name)

    # Grab the pretrained first-layer filters (shape: kh, kw, 3, out_channels)
    first_conv = next(l for l in base.layers if isinstance(l, layers.Conv2D))
    old_weights = first_conv.get_weights()[0]                      # (k,k,3,out)
    extra = np.mean(old_weights, axis=2, keepdims=True)             # (k,k,1,out)
    extra = np.repeat(extra, NUM_BANDS - 3, axis=2)                 # (k,k,3,out)
    new_weights = np.concatenate([old_weights, extra], axis=2)      # (k,k,6,out)

    inputs = layers.Input(shape=(PATCH_SIZE, PATCH_SIZE, NUM_BANDS))
    x = layers.Conv2D(
        filters=old_weights.shape[-1],
        kernel_size=old_weights.shape[:2],
        strides=first_conv.strides,
        padding=first_conv.padding,
        use_bias=False,
        name="adapted_first_conv",
    )(inputs)

    # Rebuild the rest of the backbone on top, reusing pretrained weights
    # for every layer AFTER the first conv (skip the original input + first conv).
    backbone_rest = models.Model(inputs=base.layers[2].input, outputs=base.output)
    x = backbone_rest(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = models.Model(inputs, outputs, name=f"{backbone_name}_landcover")
    model.get_layer("adapted_first_conv").set_weights([new_weights])
    model.get_layer("adapted_first_conv").trainable = True
    return model


def build_baseline_cnn():
    """Simple from-scratch CNN — your comparison baseline, same idea as the
    ablation you likely ran in NeuroVision."""
    model = models.Sequential([
        layers.Input(shape=(PATCH_SIZE, PATCH_SIZE, NUM_BANDS)),
        layers.Conv2D(32, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(),
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ], name="baseline_cnn")
    return model


def train_and_save(model, name, epochs=25, batch_size=32):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            f"models/{name}_best.keras", save_best_only=True, monitor="val_accuracy"
        ),
        tf.keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        class_weight=_compute_class_weights(y_train),
    )
    return history


def _compute_class_weights(y):
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y)
    weights = compute_class_weight("balanced", classes=classes, y=y)
    return dict(zip(classes, weights))


if __name__ == "__main__":
    import os
    os.makedirs("models", exist_ok=True)

    print("Training baseline CNN (from scratch)...")
    baseline = build_baseline_cnn()
    train_and_save(baseline, "baseline_cnn")

    print("\nTraining MobileNetV2 transfer model...")
    mnv2 = build_transfer_model("MobileNetV2")
    train_and_save(mnv2, "mobilenetv2_landcover")

    print("\nTraining EfficientNetB0 transfer model...")
    effnet = build_transfer_model("EfficientNetB0")
    train_and_save(effnet, "efficientnetb0_landcover")

    print("\nAll three models trained and saved to models/. Now run 04_evaluate.py")

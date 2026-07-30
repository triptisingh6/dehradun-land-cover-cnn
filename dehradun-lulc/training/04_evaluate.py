"""
STEP 4 — Check how good the model actually is.

Produces:
  - accuracy + Cohen's Kappa (the standard remote-sensing metric)
  - a confusion matrix plot
  - a per-class precision/recall/F1 table
  - Grad-CAM heatmaps for a few example patches (reusing your NeuroVision
    explainability approach)
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, classification_report, cohen_kappa_score, accuracy_score
)
import seaborn as sns

DATA_DIR = "data"
CLASS_NAMES = ["Vegetation", "Water", "Urban", "Bare Soil", "Agriculture"]
MODEL_PATH = "models/mobilenetv2_landcover_best.keras"   # swap to compare models

data = np.load(f"{DATA_DIR}/dehradun_dataset.npz")
X_test, y_test = data["X_test"], data["y_test"]

model = tf.keras.models.load_model(MODEL_PATH)

# ---------------------------------------------------------------------------
# 1. Predictions + core metrics
# ---------------------------------------------------------------------------
probs = model.predict(X_test)
y_pred = np.argmax(probs, axis=1)

acc = accuracy_score(y_test, y_pred)
kappa = cohen_kappa_score(y_test, y_pred)

print(f"Overall accuracy: {acc:.4f}")
print(f"Cohen's Kappa:    {kappa:.4f}")
print("\nPer-class report:")
print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

# ---------------------------------------------------------------------------
# 2. Confusion matrix plot
# ---------------------------------------------------------------------------
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion Matrix — Accuracy {acc:.2%}, Kappa {kappa:.3f}")
plt.tight_layout()
plt.savefig("results/confusion_matrix.png", dpi=150)
print("Saved results/confusion_matrix.png")

# ---------------------------------------------------------------------------
# 3. Grad-CAM — same technique as your NeuroVision project, just retargeted
#    to whichever class the model predicted for a given patch.
# ---------------------------------------------------------------------------
def grad_cam(model, image, class_index, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image[np.newaxis, ...])
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


# Find the name of the last convolutional layer automatically
last_conv_name = None
for layer in model.layers[::-1]:
    if isinstance(layer, tf.keras.layers.Conv2D) or "conv" in layer.name.lower():
        last_conv_name = layer.name
        break

if last_conv_name:
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    rng = np.random.default_rng(0)
    sample_idx = rng.choice(len(X_test), size=4, replace=False)

    for i, idx in enumerate(sample_idx):
        img = X_test[idx]
        rgb_display = img[:, :, [2, 1, 0]]  # bands are B2,B3,B4,... -> show as RGB
        rgb_display = np.clip(rgb_display * 3.5, 0, 1)  # brighten for viewing

        pred_class = np.argmax(model.predict(img[np.newaxis, ...], verbose=0))
        heatmap = grad_cam(model, img, pred_class, last_conv_name)

        axes[0, i].imshow(rgb_display)
        axes[0, i].set_title(f"Actual: {CLASS_NAMES[y_test[idx]]}")
        axes[0, i].axis("off")

        axes[1, i].imshow(rgb_display)
        axes[1, i].imshow(heatmap, cmap="jet", alpha=0.5,
                           extent=(0, img.shape[1], img.shape[0], 0))
        axes[1, i].set_title(f"Grad-CAM: predicted {CLASS_NAMES[pred_class]}")
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.savefig("results/gradcam_examples.png", dpi=150)
    print("Saved results/gradcam_examples.png")
else:
    print("Could not auto-find a conv layer for Grad-CAM — check model.summary()")

# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # MNIST CNN Filter and Complexity Lab
#
# **Aim:** Explore how convolutional layers transform image structure, compare hand-designed and learned filters, calculate CNN complexity, and avoid common beginner mistakes.
#
# Each section uses a short **what / why** explanation before the code.

# %% [markdown]
# ## 1. Setup and reproducibility
#
# **What:** Import the numerical, plotting, deep-learning, and evaluation tools and fix random seeds.
# **Why:** Reproducibility makes filter visualizations and model comparisons easier to trust.

# %%
import os
import random
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from IPython.display import display, Markdown
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
np.set_printoptions(precision=3, suppress=True)
print(f"Python: {sys.version.split()[0]}")
print(f"TensorFlow: {tf.__version__}")
print(f"NumPy: {np.__version__}")

# %% [markdown]
# ## 2. Inspect MNIST before transformation
#
# **What:** Load the official MNIST partitions and display ten examples.
# **Why:** Inspecting raw dimensions and labels prevents silent shape and label assumptions.

# %%
(x_train_raw, y_train_raw), (x_test_raw, y_test_raw) = tf.keras.datasets.mnist.load_data()
print(f"Raw train images: {x_train_raw.shape}; raw train labels: {y_train_raw.shape}")
print(f"Raw test images:  {x_test_raw.shape}; raw test labels:  {y_test_raw.shape}")
assert x_train_raw.ndim == x_test_raw.ndim == 3
assert y_train_raw.ndim == y_test_raw.ndim == 1
assert np.min(y_train_raw) >= 0 and np.max(y_train_raw) <= 9
assert np.min(y_test_raw) >= 0 and np.max(y_test_raw) <= 9
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for ax, image, label in zip(axes.ravel(), x_train_raw[:10], y_train_raw[:10]):
    ax.imshow(image, cmap="gray")
    ax.set_title(f"Label: {label}")
    ax.axis("off")
plt.tight_layout()

# %% [markdown]
# ## 3. Leakage-safe preprocessing
#
# **What:** Split raw training data first, then scale pixels and add a channel dimension.
# **Why:** The test set remains untouched and every CNN input has explicit `(height, width, channels)` structure.

# %%
x_train_raw, x_val_raw, y_train_raw, y_val_raw = train_test_split(
    x_train_raw, y_train_raw, test_size=0.10, stratify=y_train_raw, random_state=SEED
)

def prepare_images(images):
    images = images.astype("float32") / 255.0
    images = images[..., np.newaxis]
    return images

x_train = prepare_images(x_train_raw)
x_val = prepare_images(x_val_raw)
x_test = prepare_images(x_test_raw)
y_train, y_val, y_test = y_train_raw.astype("int64"), y_val_raw.astype("int64"), y_test_raw.astype("int64")
for name, images, labels in [("train", x_train, y_train), ("validation", x_val, y_val), ("test", x_test, y_test)]:
    assert images.shape[1:] == (28, 28, 1)
    assert images.dtype == np.float32
    assert np.isfinite(images).all() and 0.0 <= images.min() <= images.max() <= 1.0
    assert labels.ndim == 1 and labels.min() >= 0 and labels.max() <= 9
    print(f"{name:10s}: images {images.shape}, labels {labels.shape}, range [{images.min():.1f}, {images.max():.1f}]")
assert len(x_train) + len(x_val) == 60000
assert len(x_test) == 10000


# %% [markdown]
# ## 4. Fixed filter exploration
#
# **What:** Apply several hand-designed `3 x 3` kernels with a reusable user function.
# **Why:** Different kernels emphasize edges, smoothness, contrast, or texture before a learned CNN is introduced.

# %%
def apply_kernel(image, kernel):
    """Apply a 3x3 correlation kernel while preserving image dimensions."""
    image = np.asarray(image, dtype=np.float32)
    kernel = np.asarray(kernel, dtype=np.float32)
    assert image.ndim == 2 and kernel.shape == (3, 3)
    padded = np.pad(image, 1, mode="reflect")
    response = np.empty_like(image)
    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            response[row, col] = np.sum(padded[row:row + 3, col:col + 3] * kernel)
    return response

FILTERS = {
    "Horizontal Sobel": np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32),
    "Vertical Sobel": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32),
    "Laplacian": np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32),
    "Sharpen": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32),
    "Box blur": np.ones((3, 3), dtype=np.float32) / 9,
    "Gaussian-like blur": np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32) / 16,
    "Emboss": np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], dtype=np.float32),
}
source_image = x_train[0, ..., 0]
filter_responses = {name: apply_kernel(source_image, kernel) for name, kernel in FILTERS.items()}
assert all(response.shape == source_image.shape for response in filter_responses.values())
print("Filter response structure:", source_image.shape, "->", next(iter(filter_responses.values())).shape)
fig, axes = plt.subplots(2, 4, figsize=(14, 7))
axes = axes.ravel()
axes[0].imshow(source_image, cmap="gray", vmin=0, vmax=1)
axes[0].set_title("Original digit")
for ax, (name, response) in zip(axes[1:], filter_responses.items()):
    ax.imshow(response, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_title(name)
for ax in axes: ax.axis("off")
plt.tight_layout()

# %% [markdown]
# **Filter notes:** Sobel kernels detect directional edges; Laplacian detects rapid intensity changes; sharpen increases local contrast; blur kernels suppress noise; emboss highlights directional relief.

# %% [markdown]
# ## 4b. Multi-channel RGB image breakdown and convolution
#
# **What:** Load `sons_of_anarchy.jpg`, decompose it into Red, Green, and Blue channels, apply hand-crafted filters to each color channel, and run through a 3-channel Keras `Conv2D` layer.
# **Why:** Natural images have 3 input channels `(H, W, 3)` unlike 1-channel grayscale MNIST `(H, W, 1)`. Convolutions on RGB images calculate 3D spatial-channel dot products across all input channels simultaneously.

# %%
from PIL import Image

rgb_path = "sons_of_anarchy.jpg"
assert os.path.exists(rgb_path), f"Image file not found at {rgb_path}"

# Load RGB image and normalize pixels to [0, 1]
rgb_img_raw = Image.open(rgb_path).convert("RGB")
rgb_img_resized = rgb_img_raw.resize((224, 224))
rgb_array = np.array(rgb_img_resized, dtype=np.float32) / 255.0
assert rgb_array.shape == (224, 224, 3)

# 1. Visualize RGB channels
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
axes[0].imshow(rgb_array)
axes[0].set_title("Input RGB Image (224×224×3)")
channel_names = ["Red Channel", "Green Channel", "Blue Channel"]
colormaps = ["Reds", "Greens", "Blues"]
for idx, (c_name, cmap) in enumerate(zip(channel_names, colormaps)):
    axes[idx + 1].imshow(rgb_array[..., idx], cmap=cmap)
    axes[idx + 1].set_title(f"{c_name}\nShape: {rgb_array[..., idx].shape}")
for ax in axes: ax.axis("off")
plt.tight_layout()

# 2. Apply Sobel and Blur filters across each RGB channel independently
sobel_h = FILTERS["Horizontal Sobel"]
sharpen_k = FILTERS["Sharpen"]
r_filtered = apply_kernel(rgb_array[..., 0], sobel_h)
g_filtered = apply_kernel(rgb_array[..., 1], sharpen_k)
b_filtered = apply_kernel(rgb_array[..., 2], FILTERS["Gaussian-like blur"])

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
axes[0].imshow(r_filtered, cmap="Reds", vmin=-1, vmax=1)
axes[0].set_title("Red Channel — Horizontal Sobel")
axes[1].imshow(g_filtered, cmap="Greens", vmin=-1, vmax=1)
axes[1].set_title("Green Channel — Sharpen")
axes[2].imshow(b_filtered, cmap="Blues", vmin=0, vmax=1)
axes[2].set_title("Blue Channel — Gaussian Blur")
for ax in axes: ax.axis("off")
plt.tight_layout()

# 3. Pass 3-channel RGB tensor through Keras Conv2D (3-channel input)
rgb_conv = tf.keras.layers.Conv2D(
    filters=8, kernel_size=3, padding="same", activation="relu",
    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED)
)
rgb_tensor_batch = tf.convert_to_tensor(rgb_array[np.newaxis, ...])
rgb_conv_features = rgb_conv(rgb_tensor_batch)
assert rgb_conv_features.shape == (1, 224, 224, 8)

fig, axes = plt.subplots(2, 4, figsize=(12, 6))
fig.suptitle("RGB Image Conv2D Layer Outputs (8 Channels)", fontsize=13)
for idx, ax in enumerate(axes.ravel()):
    ax.imshow(rgb_conv_features[0, ..., idx], cmap="plasma")
    ax.set_title(f"RGB Conv Feature Map {idx + 1}")
    ax.axis("off")
plt.tight_layout()


# %% [markdown]
# ## 5. Built-in Keras convolution and pooling
#
# **What:** Run a Keras `Conv2D` layer followed by `MaxPooling2D`.
# **Why:** Built-in layers handle batched tensors and learn their kernels during training.

# %%
builtin_conv = tf.keras.layers.Conv2D(
    filters=4, kernel_size=3, padding="same", activation="relu",
    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED),
)
builtin_pool = tf.keras.layers.MaxPooling2D(pool_size=2)
builtin_input = tf.convert_to_tensor(x_train[:1])
builtin_features = builtin_conv(builtin_input)
builtin_pooled = builtin_pool(builtin_features)
print("Built-in structures:", builtin_input.shape, "->", builtin_features.shape, "->", builtin_pooled.shape)
assert tuple(builtin_features.shape) == (1, 28, 28, 4)
assert tuple(builtin_pooled.shape) == (1, 14, 14, 4)
fig, axes = plt.subplots(1, 4, figsize=(12, 3))
for index, ax in enumerate(axes):
    ax.imshow(builtin_features[0, ..., index], cmap="viridis")
    ax.set_title(f"Built-in map {index + 1}")
    ax.axis("off")
plt.tight_layout()

# %% [markdown]
# ## 6. CNN architecture and layer-by-layer structure
#
# **What:** Build a CNN with two convolution/pooling stages, dropout, and dense classification.
# **Why:** Convolutions add learned channels, pooling reduces spatial cost, and dropout regularizes the classifier.

# %%
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1), name="input"),
    tf.keras.layers.Conv2D(8, 3, padding="same", activation="relu", name="conv1"),
    tf.keras.layers.MaxPooling2D(2, name="pool1"),
    tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu", name="conv2"),
    tf.keras.layers.MaxPooling2D(2, name="pool2"),
    tf.keras.layers.Dropout(0.25, name="dropout"),
    tf.keras.layers.Flatten(name="flatten"),
    tf.keras.layers.Dense(64, activation="relu", name="dense"),
    tf.keras.layers.Dropout(0.25, name="dense_dropout"),
    tf.keras.layers.Dense(10, activation="softmax", name="classifier"),
])
model.compile(optimizer=tf.keras.optimizers.Adam(), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()
print("Key structure: (28, 28, 1) -> (28, 28, 8) -> (14, 14, 8) -> (14, 14, 16) -> (7, 7, 16)")


# %% [markdown]
# ### Shape trace
#
# **What:** Forward a zero tensor through each layer and record the output shape.
# **Why:** Explicit shape tracing catches channel and flattening mistakes before training.

# %%
def trace_model_shapes(model, input_shape):
    tensor = tf.zeros((1, *input_shape), dtype=tf.float32)
    rows = [{"layer": "input", "output_shape": tuple(tensor.shape[1:]), "activations": int(np.prod(tensor.shape[1:]))}]
    for layer in model.layers:
        tensor = layer(tensor, training=False)
        rows.append({"layer": layer.name, "output_shape": tuple(tensor.shape[1:]), "activations": int(np.prod(tensor.shape[1:]))})
    return pd.DataFrame(rows)

shape_table = trace_model_shapes(model, (28, 28, 1))
display(shape_table)
assert tuple(shape_table.iloc[0].output_shape) == (28, 28, 1)
assert tuple(shape_table.loc[shape_table.layer == "pool1", "output_shape"].iloc[0]) == (14, 14, 8)
assert tuple(shape_table.loc[shape_table.layer == "pool2", "output_shape"].iloc[0]) == (7, 7, 16)


# %% [markdown]
# ## 7. Complexity analysis
#
# **What:** Count parameters, activation elements, and approximate convolution MACs.
# **Why:** Parameters measure stored learnable capacity, while MACs estimate computation.

# %%
def complexity_rows(model, input_shape):
    tensor = tf.zeros((1, *input_shape), dtype=tf.float32)
    rows = []
    for layer in model.layers:
        input_tensor = tensor
        tensor = layer(tensor, training=False)
        output_shape = tuple(int(dim) for dim in tensor.shape[1:])
        params = int(layer.count_params())
        macs = 0
        if isinstance(layer, tf.keras.layers.Conv2D):
            kh, kw = layer.kernel_size
            macs = int(np.prod(output_shape) * kh * kw * input_tensor.shape[-1])
        elif isinstance(layer, tf.keras.layers.Dense):
            macs = int(input_tensor.shape[-1] * layer.units)
        rows.append({"layer": layer.name, "output_shape": output_shape, "parameters": params, "activations": int(np.prod(output_shape)), "approx_MACs": macs})
    return pd.DataFrame(rows)

complexity_table = complexity_rows(model, (28, 28, 1))
display(complexity_table)
print("Total trainable parameters:", complexity_table.parameters.sum())
print("Keras model parameters:", model.count_params())
print("Total approximate MACs:", complexity_table.approx_MACs.sum())
assert complexity_table.parameters.sum() == model.count_params()
assert complexity_table.loc[complexity_table.layer == "conv1", "parameters"].iloc[0] == 8 * 3 * 3 * 1 + 8
assert complexity_table.loc[complexity_table.layer == "dense", "parameters"].iloc[0] == 7 * 7 * 16 * 64 + 64

# %% [markdown]
# ## 8. Training with validation monitoring
#
# **What:** Train with early stopping on validation loss and restore the best weights.
# **Why:** Validation monitoring detects overfitting without using the test set for model selection.

# %%
early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)
history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=12, batch_size=128, callbacks=[early_stopping], verbose=2)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(history.history["loss"], label="train")
axes[0].plot(history.history["val_loss"], label="validation")
axes[0].set_title("Loss"); axes[0].legend()
axes[1].plot(history.history["accuracy"], label="train")
axes[1].plot(history.history["val_accuracy"], label="validation")
axes[1].set_title("Accuracy"); axes[1].legend()
plt.tight_layout()

# %% [markdown]
# ## 9. Final test evaluation
#
# **What:** Evaluate once on the untouched test set with class-balanced metrics.
# **Why:** A confusion matrix and macro/weighted scores reveal errors that accuracy can hide.

# %%
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
probabilities = model.predict(x_test, verbose=0)
y_pred = probabilities.argmax(axis=1)
assert len(y_pred) == len(y_test)
report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
report_df = pd.DataFrame(report).T
test_metrics = {"loss": float(test_loss), "accuracy": float(test_accuracy), "macro_f1": float(report["macro avg"]["f1-score"]), "weighted_f1": float(report["weighted avg"]["f1-score"])}
display(pd.DataFrame([test_metrics]))
display(report_df.round(3))
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(7, 6))
image = ax.imshow(cm, cmap="Blues")
fig.colorbar(image, ax=ax)
ax.set(xlabel="Predicted label", ylabel="True label", title="MNIST confusion matrix")
ax.set_xticks(range(10)); ax.set_yticks(range(10))
for row in range(10):
    for col in range(10):
        ax.text(col, row, cm[row, col], ha="center", va="center", fontsize=8)
plt.tight_layout()

# %% [markdown]
# ## 10. Learned filters and feature maps
#
# **What:** Visualize the first layer's learned kernels and feature maps for one test digit.
# **Why:** Learned kernels adapt to training examples rather than following a fixed hand-designed rule.

# %%
conv1 = model.get_layer("conv1")
learned_kernels = conv1.get_weights()[0]
fig, axes = plt.subplots(2, 4, figsize=(10, 5))
for index, ax in enumerate(axes.ravel()):
    ax.imshow(learned_kernels[..., 0, index], cmap="coolwarm")
    ax.set_title(f"Kernel {index + 1}"); ax.axis("off")
plt.tight_layout()
feature_input = tf.keras.Input(shape=(28, 28, 1), name="feature_input")
feature_model = tf.keras.Model(feature_input, conv1(feature_input))
feature_maps = feature_model.predict(x_test[:1], verbose=0)[0]
fig, axes = plt.subplots(2, 4, figsize=(10, 5))
for index, ax in enumerate(axes.ravel()):
    ax.imshow(feature_maps[..., index], cmap="viridis")
    ax.set_title(f"Feature map {index + 1}"); ax.axis("off")
plt.tight_layout()

# %% [markdown]
# ## 10b. Full Layer-by-Layer Activation Trace (Conv, Pool, Flatten, Latent Vector)
#
# **What:** Extract and visualize activations after every layer (`conv1`, `pool1`, `conv2`, `pool2`, `flatten`, and `dense`).
# **Why:** Visualizing the transition from 2D spatial feature grids (`28×28` -> `14×14` -> `7×7`) down to 1D latent vector embeddings (`784-dim` -> `64-dim`) demonstrates how CNNs compress spatial inputs into classification features.

# %%
# Build multi-output Keras model for full activation tracing
trace_layers = ["conv1", "pool1", "conv2", "pool2", "flatten", "dense", "classifier"]
multi_output_model = tf.keras.Model(inputs=model.input, outputs=[model.get_layer(l).output for l in trace_layers])

sample_digit = x_test[:1]
sample_label = y_test[0]
layer_activations = multi_output_model.predict(sample_digit, verbose=0)
act_dict = dict(zip(trace_layers, layer_activations))

# 1. Output channels after Conv1 Layer (8 channels, 28x28)
fig, axes = plt.subplots(2, 4, figsize=(10, 5))
fig.suptitle(f"Conv1 Output Channels — Shape: {act_dict['conv1'].shape[1:]} (Input Digit: {sample_label})", fontsize=12)
for idx, ax in enumerate(axes.ravel()):
    ax.imshow(act_dict['conv1'][0, ..., idx], cmap="viridis")
    ax.set_title(f"Conv1 Ch {idx + 1}"); ax.axis("off")
plt.tight_layout()

# 2. Pooling1 Outputs (8 channels downsampled to 14x14)
fig, axes = plt.subplots(2, 4, figsize=(10, 5))
fig.suptitle(f"Pool1 Outputs (Max Pooling 2x2) — Shape: {act_dict['pool1'].shape[1:]}", fontsize=12)
for idx, ax in enumerate(axes.ravel()):
    ax.imshow(act_dict['pool1'][0, ..., idx], cmap="viridis")
    ax.set_title(f"Pool1 Ch {idx + 1}"); ax.axis("off")
plt.tight_layout()

# 3. Output channels after Conv2 Layer (16 channels, 14x14)
fig, axes = plt.subplots(2, 8, figsize=(16, 4))
fig.suptitle(f"Conv2 Output Channels — Shape: {act_dict['conv2'].shape[1:]}", fontsize=12)
for idx, ax in enumerate(axes.ravel()):
    ax.imshow(act_dict['conv2'][0, ..., idx], cmap="plasma")
    ax.set_title(f"Ch {idx + 1}"); ax.axis("off")
plt.tight_layout()

# 4. Pooling2 Outputs (16 channels downsampled to 7x7)
fig, axes = plt.subplots(2, 8, figsize=(16, 4))
fig.suptitle(f"Pool2 Outputs (Max Pooling 2x2) — Shape: {act_dict['pool2'].shape[1:]}", fontsize=12)
for idx, ax in enumerate(axes.ravel()):
    ax.imshow(act_dict['pool2'][0, ..., idx], cmap="plasma")
    ax.set_title(f"Ch {idx + 1}"); ax.axis("off")
plt.tight_layout()

# 5. Flattened feature representation & Latent Vector before classification layer
fig, (ax_flat, ax_dense) = plt.subplots(2, 1, figsize=(12, 6))

flatten_vec = act_dict['flatten'][0]
ax_flat.imshow(flatten_vec.reshape(1, -1), aspect="auto", cmap="magma")
ax_flat.set_title(f"Flattened Feature Representation — 1D Vector Shape: ({len(flatten_vec)},) [7×7×16 = 784]")
ax_flat.set_yticks([])
ax_flat.set_xlabel("Flattened Feature Index (0 to 783)")

dense_latent_vec = act_dict['dense'][0]
ax_dense.bar(range(len(dense_latent_vec)), dense_latent_vec, color="indigo", edgecolor="black", alpha=0.8)
ax_dense.set_title(f"Latent Vector Obtained Before Classification Layer — Shape: ({len(dense_latent_vec)},)")
ax_dense.set_xlabel("Latent Feature Units (0 to 63)")
ax_dense.set_ylabel("Activation Intensity")
plt.tight_layout()

# %% [markdown]
# ## 10c. Hierarchical Feature Extraction Comparison
#
# **What:** Compare how features evolve from low-level edges to mid-level patterns and high-level latent concepts.
# **Why:** Understanding the progression from low-level spatial geometry to abstract class embeddings is fundamental to CNN interpretability.
#
# | Stage | Spatial Resolution | Output Shape | Feature Abstraction | Visual / Feature Characteristics |
# |---|---|---|---|---|
# | **Input** | `28 × 28` | `(28, 28, 1)` | Raw Pixel Input | Raw grayscale intensity values. |
# | **Conv1** | `28 × 28` | `(28, 28, 8)` | **Low-level Features** | Basic edge orientation, stroke boundaries, local contrast gradients. |
# | **Pool1** | `14 × 14` | `(14, 14, 8)` | Spatial Downsampling | Max-pooled local edge responses; provides translation invariance. |
# | **Conv2** | `14 × 14` | `(14, 14, 16)` | **Mid-level Features** | Combinations of edges: loops, curves, intersections, and digit components. |
# | **Pool2** | `7 × 7` | `(7, 7, 16)` | Spatial Downsampling | Compact localized feature grids highlighting presence of key digit parts. |
# | **Flatten** | `1 × 1` | `(784,)` | Vector Conversion | Concatenated 1D stream of all 7×7×16 spatial feature responses. |
# | **Dense** | `1 × 1` | `(64,)` | **High-level / Latent Vector** | Abstract semantic fingerprint of the input image prior to softmax logits. |

# %% [markdown]
# ## 11. Beginner-error audit
#
# | Common mistake | Notebook fix |
# |---|---|
# | Data leakage | Split before transformations and keep test data isolated. |
# | Test-set tuning | Use train/validation data for model decisions. |
# | Ignoring overfitting | Plot validation curves and use early stopping. |
# | Shape/channel errors | Assert input rank and print each layer structure. |
# | Uncontrolled randomness | Seed Python, NumPy, and TensorFlow. |
# | Accuracy-only evaluation | Report macro/weighted metrics and a confusion matrix. |
# | Misreading filter output | Clip only for visualization; keep model inputs unchanged. |

# %% [markdown]
# ## 12. Additional Learnings and Architectural Enhancements
#
# **What:** Key strategies and techniques to improve CNN performance and interpretability beyond basic architectures.
# **Why:** Applying modern deep learning practices makes models train faster, generalize better, and offer inspectable decisions.
#
# 1. **Batch Normalization (`BatchNormalization`):**
#    - Insert `tf.keras.layers.BatchNormalization()` immediately after `Conv2D` layers (or before activation functions).
#    - *Benefit:* Reduces internal covariate shift, enables higher learning rates, acts as a regularizer, and significantly speeds up training convergence.
#
# 2. **Global Average Pooling (`GlobalAveragePooling2D`) vs. `Flatten`:**
#    - Replace `Flatten()` with `GlobalAveragePooling2D()`.
#    - *Benefit:* Reduces parameters dramatically by taking the spatial mean of each feature map (e.g. `7×7×16` -> `16` units instead of `784`), preventing dense layer overfitting and enabling variable input spatial resolutions.
#
# 3. **Data Augmentation:**
#    - Integrate input transformation layers such as `tf.keras.layers.RandomRotation(0.05)` and `RandomTranslation(0.05, 0.05)`.
#    - *Benefit:* Exposes the network to realistic spatial perturbations, boosting generalization on un-padded or slightly rotated test digits.
#
# 4. **Grad-CAM (Gradient-weighted Class Activation Mapping):**
#    - Calculate gradients of the winning class output with respect to `conv2` feature maps, weighting activation channels by their gradient importance.
#    - *Benefit:* Produces a spatial visual heatmap overlaid on the input image, showing *exactly* which region of the digit triggered the model's classification decision.

# %% [markdown]
# **References:** [scikit-learn common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html), [Google overfitting guidance](https://developers.google.com/machine-learning/crash-course/overfitting), [Keras Conv2D](https://keras.io/api/layers/convolution_layers/convolution2d/), [Keras EarlyStopping](https://keras.io/api/callbacks/early_stopping/), and [Grad-CAM paper (Selvaraju et al.)](https://arxiv.org/abs/1610.02391).



# %% [markdown]
# ## 13. RGB Input Channel Decomposition — Enhanced Visualization
#
# **What:** Load `sons_of_anarchy.jpg`, decompose it into R, G, B channels with annotated statistics.
# **Why:** An RGB image is a 3D tensor `(H, W, 3)`. Each channel is an independent 2D grayscale map.
# A Conv2D on RGB computes 3D dot products — one per input channel — and sums them to produce each output map.

# %%
import os
from PIL import Image

rgb_path = 'sons_of_anarchy.jpg'
assert os.path.exists(rgb_path), f'RGB image not found: {rgb_path}'

rgb_img_resized = Image.open(rgb_path).convert('RGB').resize((224, 224))
rgb_array = np.array(rgb_img_resized, dtype=np.float32) / 255.0  # (224, 224, 3)

fig, axes = plt.subplots(1, 5, figsize=(22, 4))
fig.suptitle('SEC 13 — RGB Channel Decomposition  (tensor shape: 224x224x3)', fontsize=13, fontweight='bold')

axes[0].imshow(rgb_array)
axes[0].set_title('Original RGB Input\n(224x224x3)', fontsize=9)
axes[0].axis('off')

ch_cfg = [('Red', 0, 'Reds'), ('Green', 1, 'Greens'), ('Blue', 2, 'Blues')]
for ax, (name, idx, cmap) in zip(axes[1:4], ch_cfg):
    ch = rgb_array[..., idx]
    ax.imshow(ch, cmap=cmap, vmin=0, vmax=1)
    ax.set_title(f'{name} Channel\nshape: {ch.shape}\nmean={ch.mean():.3f}', fontsize=8)
    ax.axis('off')

axes[4].bar(['R', 'G', 'B'],
            [rgb_array[..., i].mean() for i in range(3)],
            color=['red', 'green', 'blue'], alpha=0.8, edgecolor='black')
axes[4].set_title('Per-channel mean\nintensity', fontsize=9)
axes[4].set_ylabel('Mean pixel value')
axes[4].set_ylim(0, 1)
axes[4].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

print(f'\n RGB tensor: {rgb_array.shape}, dtype={rgb_array.dtype}')
print(f'   Channel means: R={rgb_array[...,0].mean():.3f}  G={rgb_array[...,1].mean():.3f}  B={rgb_array[...,2].mean():.3f}')


# %% [markdown]
# ## 14. Per-Channel Convolution with Annotated Kernel Weights
#
# **What:** Apply a different hand-crafted 3x3 filter to each colour channel.
# Display kernel weight values inside each cell, filter response, and overlay.
# **Why:** Shows that Conv2D on RGB = 3 independent 2D cross-correlations summed per output filter.
#
# | Filter | What it detects |
# |---|---|
# | Horizontal Sobel | Horizontal intensity gradients |
# | Sharpen | Local contrast amplification |
# | Gaussian Blur | Smooth low-frequency regions |

# %%
filter_channel_map = [
    ('Red',   0, 'Reds',   'Horizontal Sobel',    'RdBu_r'),
    ('Green', 1, 'Greens', 'Sharpen',             'PuOr_r'),
    ('Blue',  2, 'Blues',  'Gaussian-like blur',  'BuGn'),
]

fig, axes = plt.subplots(3, 4, figsize=(20, 14))
fig.suptitle('SEC 14 — Per-Channel Convolution: Kernel Weights | Response | Overlay', fontsize=13, fontweight='bold')

for row, (ch_name, ch_idx, ch_cmap, filt_name, resp_cmap) in enumerate(filter_channel_map):
    ch_data  = rgb_array[..., ch_idx]
    kernel   = FILTERS[filt_name]
    response = apply_kernel(ch_data, kernel)
    vabs     = max(abs(response.min()), abs(response.max())) + 1e-8
    overlay  = np.clip(ch_data + 0.3 * response / vabs, 0, 1)

    axes[row, 0].imshow(ch_data, cmap=ch_cmap, vmin=0, vmax=1)
    axes[row, 0].set_title(f'{ch_name} Channel (input)\nshape: {ch_data.shape}', fontsize=9)
    axes[row, 0].axis('off')

    axes[row, 1].imshow(kernel, cmap='coolwarm',
                        vmin=-abs(kernel).max(), vmax=abs(kernel).max(),
                        interpolation='nearest')
    axes[row, 1].set_title(f'Kernel: {filt_name}\n(3x3 weights shown)', fontsize=9)
    axes[row, 1].axis('off')
    for ri in range(3):
        for ci in range(3):
            val = kernel[ri, ci]
            txt_color = 'white' if abs(val) > abs(kernel).max() * 0.5 else 'black'
            axes[row, 1].text(ci, ri, f'{val:.2f}', ha='center', va='center',
                               fontsize=9, color=txt_color, fontweight='bold')

    axes[row, 2].imshow(response, cmap=resp_cmap, vmin=-vabs, vmax=vabs)
    axes[row, 2].set_title(f'{filt_name}\nConvolution Response', fontsize=9)
    axes[row, 2].axis('off')

    axes[row, 3].imshow(overlay, cmap=ch_cmap, vmin=0, vmax=1)
    axes[row, 3].set_title(f'Overlay: Channel + Filter\n(0.3 x response added)', fontsize=9)
    axes[row, 3].axis('off')

plt.tight_layout()
plt.show()

print('\nKey learning: In Keras Conv2D(filters=K, kernel_size=3) on RGB:')
print('  Each of the K filters has shape (3, 3, 3) -- one (3,3) slice per colour channel.')
print('  All 3 slices are summed to produce ONE output feature map per filter.')


# %% [markdown]
# ## 15. Grad-CAM -- Explainability Heatmap
#
# **What:** Gradient-weighted Class Activation Mapping overlaid on the input digit.
# **Why:** Answers WHERE did the CNN look to make its classification decision?
# Essential for interpretability and demonstrating that the model attends to meaningful regions.
#
# **Algorithm:**
# 1. Forward pass -> predicted class logit
# 2. Backpropagate gradients to conv2 feature maps
# 3. Global-average-pool gradients -> importance weight alpha_k per channel
# 4. Weighted sum of maps -> raw heatmap; apply ReLU
# 5. Upsample to input size; overlay on digit

# %%
_sample = x_test[:1]
_true_label = int(y_test[0])

grad_cam_model = tf.keras.Model(
    inputs=model.input,
    outputs=[model.get_layer('conv2').output, model.output]
)

with tf.GradientTape() as tape:
    inputs_tf = tf.cast(_sample, tf.float32)
    conv_out_gc, preds_gc = grad_cam_model(inputs_tf)
    pred_class_idx = tf.argmax(preds_gc[0])
    class_score = preds_gc[:, pred_class_idx]

grads_gc = tape.gradient(class_score, conv_out_gc)
pooled_grads_gc = tf.reduce_mean(grads_gc, axis=(0, 1, 2))

cam_conv = conv_out_gc[0].numpy()
for k in range(cam_conv.shape[-1]):
    cam_conv[:, :, k] *= pooled_grads_gc[k].numpy()

heatmap_gc = np.mean(cam_conv, axis=-1)
heatmap_gc = np.maximum(heatmap_gc, 0)
heatmap_norm_gc = heatmap_gc / (heatmap_gc.max() + 1e-8)
heatmap_up_gc = np.array(Image.fromarray(heatmap_norm_gc).resize((28, 28), Image.BILINEAR))

digit_rgb_gc = np.stack([_sample[0, ..., 0]] * 3, axis=-1)
heat_rgb_gc  = plt.cm.jet(heatmap_up_gc)[..., :3]
overlay_gc   = 0.6 * digit_rgb_gc + 0.4 * heat_rgb_gc

fig, axes = plt.subplots(1, 4, figsize=(18, 5))
fig.suptitle(
    f'SEC 15 -- Grad-CAM | True: {_true_label}  Predicted: {int(pred_class_idx)}  '
    f'Confidence: {float(preds_gc[0, pred_class_idx]) * 100:.1f}%',
    fontsize=12, fontweight='bold'
)
axes[0].imshow(_sample[0, ..., 0], cmap='gray')
axes[0].set_title(f'Input Digit: {_true_label}\n(28x28)', fontsize=9)
axes[0].axis('off')
axes[1].imshow(heatmap_norm_gc, cmap='jet')
axes[1].set_title('Raw Grad-CAM Heatmap\n(14x14 from conv2)', fontsize=9)
axes[1].axis('off')
axes[2].imshow(heatmap_up_gc, cmap='jet')
axes[2].set_title('Upsampled Heatmap\n(28x28 bilinear)', fontsize=9)
axes[2].axis('off')
axes[3].imshow(overlay_gc)
axes[3].set_title('Grad-CAM Overlay\n(60% digit + 40% heat)', fontsize=9)
axes[3].axis('off')
plt.tight_layout()
plt.show()

print('\nGrad-CAM: hot regions (red/yellow) = areas most responsible for prediction.')
print('  If heatmap focuses on digit strokes: model is correct for the RIGHT reasons!')


# %% [markdown]
# ## 16. Effective Receptive Field Growth
#
# **What:** Show how large a region each layer neuron effectively sees in the original 28x28 input.
# **Why:** Receptive field explains why deeper layers capture global digit structure
# while early layers only respond to tiny local patterns.
#
# | Layer | RF Size | Notes |
# |---|---|---|
# | Input | 1x1 | Single pixel |
# | Conv1 (3x3) | 3x3 | One kernel |
# | Pool1 (2x2 stride 2) | 6x6 | Stride doubles |
# | Conv2 (3x3) | 10x10 | +4 pixels |
# | Pool2 (2x2 stride 2) | 16x16 | Doubled again |

# %%
rf_stages = [
    ('Input',  1,  '(28x28 single pixel)'),
    ('Conv1',  3,  '(3x3 kernel)'),
    ('Pool1',  6,  '(stride 2 doubles)'),
    ('Conv2', 10,  '(adds 4 per side)'),
    ('Pool2', 16,  '(halved again)'),
]

fig, axes = plt.subplots(1, 5, figsize=(20, 4))
fig.suptitle('SEC 16 -- Effective Receptive Field Growth per Layer\n'
             'Red box = region in original 28x28 input that each neuron sees',
             fontsize=12, fontweight='bold')

digit_bg = x_test[0, ..., 0]

for col, (stage_name, rf, note) in enumerate(rf_stages):
    ax = axes[col]
    ax.imshow(digit_bg, cmap='gray', alpha=0.45, extent=[0, 28, 28, 0])
    cx = cy = 14
    half = rf // 2
    rect = plt.Rectangle((cx - half, cy - half), rf, rf,
                           linewidth=2.5, edgecolor='red', facecolor='red', alpha=0.25)
    ax.add_patch(rect)
    ax.set_xlim(0, 28)
    ax.set_ylim(28, 0)
    ax.set_title(f'{stage_name}\nRF = {rf}x{rf}\n{note}', fontsize=8.5)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()


# %% [markdown]
# ## 17. Flattened Representation + Latent Vector -- Enhanced
#
# **What:** Three simultaneous views: flatten heat-strip (784-D), dense latent bar (64-D), softmax output.
# **Why:** Shows the complete bridge from 2D spatial feature grids to the 1D classification vector.

# %%
_trace_names = ['conv1', 'pool1', 'conv2', 'pool2', 'flatten', 'dense', 'classifier']
_multi = tf.keras.Model(inputs=model.input,
                        outputs=[model.get_layer(l).output for l in _trace_names])
_acts  = dict(zip(_trace_names, _multi.predict(x_test[:1], verbose=0)))

flatten_v  = _acts['flatten'][0]     # (784,)
dense_v    = _acts['dense'][0]       # (64,)
softmax_v  = _acts['classifier'][0]  # (10,)
_pred      = int(softmax_v.argmax())
_true      = int(y_test[0])

fig, axes = plt.subplots(3, 1, figsize=(18, 9))
fig.suptitle(f'SEC 17 -- Flatten (784-D) | Latent Dense (64-D) | Softmax Output\n'
             f'True: {_true}  Predicted: {_pred}  Confidence: {softmax_v[_pred] * 100:.1f}%',
             fontsize=12, fontweight='bold')

axes[0].imshow(flatten_v.reshape(1, -1), aspect='auto', cmap='magma')
axes[0].set_title(f'Flatten -> (784,) = 7x7x16  '
                  f'Non-zero: {np.count_nonzero(flatten_v)}/784 '
                  f'({100 * np.count_nonzero(flatten_v) / 784:.0f}%)', fontsize=9)
axes[0].set_yticks([])
axes[0].set_xlabel('Flat index (0 to 783)')

latent_colors = ['crimson' if v > 0.5 * dense_v.max() else 'steelblue' for v in dense_v]
axes[1].bar(range(64), dense_v, color=latent_colors, edgecolor='black', linewidth=0.3, alpha=0.85)
axes[1].axhline(y=0.5 * dense_v.max(), color='red', linestyle='--', lw=0.8, alpha=0.6, label='50% max')
axes[1].set_title('Dense Latent Vector -> (64,) ReLU  Red bars = most active units', fontsize=9)
axes[1].set_xlabel('Latent unit (0-63)')
axes[1].set_ylabel('Activation')
axes[1].legend(fontsize=8)
axes[1].grid(axis='y', alpha=0.3)

bar_c = ['gold' if i == _pred else 'lightsteelblue' for i in range(10)]
bars  = axes[2].bar(range(10), softmax_v * 100, color=bar_c, edgecolor='black', lw=0.5)
axes[2].set_xticks(range(10))
axes[2].set_xticklabels([f'Digit {i}' for i in range(10)], rotation=30, fontsize=8)
axes[2].set_ylabel('Probability (%)')
axes[2].set_title(f'Softmax Output (10-D)  Predicted: {_pred}  True: {_true}', fontsize=9)
for bar, p in zip(bars, softmax_v):
    axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f'{p * 100:.1f}%', ha='center', va='bottom', fontsize=7)
axes[2].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()


# %% [markdown]
# ## 18. CNN Pipeline Summary Poster
#
# **What:** A single annotated figure showing the complete CNN pipeline at every stage.
# **Why:** Gives invigilators an at-a-glance view of the entire feature extraction journey.

# %%
import matplotlib.patches as mpatches

fig, ax = plt.subplots(figsize=(22, 5))
ax.axis('off')
fig.patch.set_facecolor('#1a1a2e')
ax.set_facecolor('#1a1a2e')

pipeline_stages = [
    ('RGB Input\n224x224x3',    '#e74c3c'),
    ('R/G/B\nChannels',         '#e74c3c'),
    ('Conv2D x8\n224x224x8',    '#3498db'),
    ('MNIST\nInput 28x28x1',    '#7f8c8d'),
    ('Conv1\n28x28x8',          '#2ecc71'),
    ('Pool1\n14x14x8',          '#f39c12'),
    ('Conv2\n14x14x16',         '#9b59b6'),
    ('Pool2\n7x7x16',           '#e67e22'),
    ('Flatten\n(784,)',          '#1abc9c'),
    ('Dense\nLatent (64,)',      '#e74c3c'),
    ('Softmax\n(10,)',           '#2ecc71'),
]

xs = np.linspace(0.04, 0.96, len(pipeline_stages))

for xi, (label, color) in zip(xs, pipeline_stages):
    ax.text(xi, 0.58, label, ha='center', va='center', fontsize=7.5,
            color='white', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor=color, alpha=0.9, edgecolor='white', lw=1.2))

for x0, x1 in zip(xs[:-1], xs[1:]):
    ax.annotate('', xy=(x1 - 0.025, 0.58), xytext=(x0 + 0.025, 0.58),
                xycoords='axes fraction', textcoords='axes fraction',
                arrowprops=dict(arrowstyle='->', color='white', lw=1.5))

ax.text(0.5, 0.15,
        'Low-level Edges  ->  Mid-level Combinations  ->  High-level Latent  ->  Class Prediction',
        ha='center', va='center', fontsize=10, color='lightgray', style='italic')

ax.set_title('SEC 18 -- CNN Full Processing Pipeline (RGB Photo + MNIST Digit)',
             fontsize=13, fontweight='bold', color='white', pad=12)

plt.tight_layout()
plt.show()

print('\nAll sections complete (1-18).')
print('Open in Jupyter for inline visualizations with annotations.')

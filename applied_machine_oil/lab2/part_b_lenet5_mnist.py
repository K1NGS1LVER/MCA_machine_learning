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
# # Part B: LeNet-5 on MNIST with Hyperparameter Tuning
#
# **Aim:** Implement the classic LeNet-5 architecture on MNIST, measure its layer-wise complexity, and use a bounded KerasTuner search to select a stronger configuration.
#
# Each section uses short **what / why** explanations.

# %% [markdown]
# ## 1. Setup and leakage-safe preprocessing
#
# **What:** Load MNIST, split training data into train/validation, and pad images to 32×32.
# **Why:** The 32×32 input matches classic LeNet-5 while keeping the official test set untouched.

# %%
import os
import random
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras_tuner as kt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)
print(f"Python: {sys.version.split()[0]}; TensorFlow: {tf.__version__}; KerasTuner: {kt.__version__}")

# %%
(x_train_raw, y_train_raw), (x_test_raw, y_test) = tf.keras.datasets.mnist.load_data()
x_train_raw, x_val_raw, y_train, y_val = train_test_split(x_train_raw, y_train_raw, test_size=0.10, stratify=y_train_raw, random_state=SEED)

def prepare_mnist(images):
    images = images.astype("float32") / 255.0
    images = np.pad(images, ((0, 0), (2, 2), (2, 2)), mode="constant")
    return images[..., np.newaxis]

x_train, x_val, x_test = prepare_mnist(x_train_raw), prepare_mnist(x_val_raw), prepare_mnist(x_test_raw)
y_train, y_val, y_test = y_train.astype("int64"), y_val.astype("int64"), y_test.astype("int64")
for name, images, labels in [("train", x_train, y_train), ("validation", x_val, y_val), ("test", x_test, y_test)]:
    assert images.shape[1:] == (32, 32, 1) and images.dtype == np.float32
    assert np.isfinite(images).all() and 0 <= images.min() <= images.max() <= 1
    assert labels.ndim == 1 and labels.min() >= 0 and labels.max() <= 9
    print(f"{name}: images={images.shape}, labels={labels.shape}")
print("The original 28×28 images are zero-padded to 32×32 for classic LeNet-5.")

# %%
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for ax, image, label in zip(axes.ravel(), x_train[:10], y_train[:10]):
    ax.imshow(image[..., 0], cmap="gray"); ax.set_title(f"Digit {label}"); ax.axis("off")
plt.tight_layout()
class_counts = np.bincount(y_train, minlength=10)
display_table = pd.DataFrame({"digit": np.arange(10), "training_count": class_counts})
display(display_table)


# %% [markdown]
# ## 2. Classic LeNet-5 baseline
#
# **What:** Build LeNet-5 with `tanh`, average pooling, and the historical 6→16→120 channel progression.
# **Why:** These choices reproduce the classic architecture rather than silently modernizing it to ReLU/max pooling.

# %%
def build_baseline_lenet():
    model = tf.keras.Sequential([
        tf.keras.layers.Input((32, 32, 1), name="input"),
        tf.keras.layers.Conv2D(6, 5, activation="tanh", name="conv1"),
        tf.keras.layers.AveragePooling2D(2, name="pool1"),
        tf.keras.layers.Conv2D(16, 5, activation="tanh", name="conv2"),
        tf.keras.layers.AveragePooling2D(2, name="pool2"),
        tf.keras.layers.Conv2D(120, 5, activation="tanh", name="conv3"),
        tf.keras.layers.Flatten(name="flatten"),
        tf.keras.layers.Dense(84, activation="tanh", name="dense1"),
        tf.keras.layers.Dense(10, activation="softmax", name="classifier"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

baseline_model = build_baseline_lenet()
baseline_model.summary()


# %%
def complexity_rows(model, input_shape):
    tensor = tf.zeros((1, *input_shape), dtype=tf.float32); rows=[]
    for layer in model.layers:
        input_tensor=tensor; tensor=layer(tensor, training=False)
        output_shape=tuple(int(d) for d in tensor.shape[1:]); macs=0
        if isinstance(layer, tf.keras.layers.Conv2D):
            kh, kw=layer.kernel_size; macs=int(np.prod(output_shape)*kh*kw*int(input_tensor.shape[-1]))
        elif isinstance(layer, tf.keras.layers.Dense):
            macs=int(input_tensor.shape[-1]*layer.units)
        rows.append({"layer":layer.name,"output_shape":output_shape,"parameters":int(layer.count_params()),"activations":int(np.prod(output_shape)),"approx_MACs":macs})
    return pd.DataFrame(rows)

baseline_complexity=complexity_rows(baseline_model,(32,32,1))
display(baseline_complexity)
print("Baseline parameters:", baseline_model.count_params(), "Approximate MACs:", baseline_complexity.approx_MACs.sum())
assert baseline_complexity.parameters.sum()==baseline_model.count_params()
assert tuple(baseline_complexity.iloc[-1].output_shape)==(10,)

# %%
baseline_stop=tf.keras.callbacks.EarlyStopping(monitor="val_loss",patience=2,restore_best_weights=True)
baseline_history=baseline_model.fit(x_train,y_train,validation_data=(x_val,y_val),epochs=8,batch_size=128,callbacks=[baseline_stop],verbose=2)
fig,axes=plt.subplots(1,2,figsize=(12,4))
axes[0].plot(baseline_history.history["loss"],label="train"); axes[0].plot(baseline_history.history["val_loss"],label="validation"); axes[0].set_title("Baseline loss"); axes[0].legend()
axes[1].plot(baseline_history.history["accuracy"],label="train"); axes[1].plot(baseline_history.history["val_accuracy"],label="validation"); axes[1].set_title("Baseline accuracy"); axes[1].legend(); plt.tight_layout()


# %% [markdown]
# ## 3. Bounded KerasTuner RandomSearch
#
# **What:** Search a small, reproducible hyperparameter space using validation accuracy.
# **Why:** Validation data selects configurations; the test set remains reserved for the final comparison.

# %%
def build_tunable_lenet(hp):
    filters1=hp.Choice("filters1",[6,12]); filters2=hp.Choice("filters2",[16,32])
    dense_units=hp.Choice("dense_units",[84,120]); dropout_rate=hp.Choice("dropout_rate",[0.0,0.2,0.4])
    learning_rate=hp.Choice("learning_rate",[1e-3,3e-4]); batch_size=hp.Choice("batch_size",[64,128])
    model=tf.keras.Sequential([
        tf.keras.layers.Input((32,32,1),name="input"),
        tf.keras.layers.Conv2D(filters1,5,activation="tanh",name="conv1"),
        tf.keras.layers.AveragePooling2D(2,name="pool1"),
        tf.keras.layers.Conv2D(filters2,5,activation="tanh",name="conv2"),
        tf.keras.layers.AveragePooling2D(2,name="pool2"),
        tf.keras.layers.Conv2D(120,5,activation="tanh",name="conv3"),
        tf.keras.layers.Flatten(name="flatten"),
        tf.keras.layers.Dense(dense_units,activation="tanh",name="dense1"),
        tf.keras.layers.Dropout(dropout_rate,name="dropout"),
        tf.keras.layers.Dense(10,activation="softmax",name="classifier"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),loss="sparse_categorical_crossentropy",metrics=["accuracy"])
    return model

class BatchSizeTuner(kt.RandomSearch):
    def run_trial(self, trial, *args, **kwargs):
        kwargs["batch_size"] = trial.hyperparameters.get("batch_size")
        return super().run_trial(trial, *args, **kwargs)

tuner=BatchSizeTuner(build_tunable_lenet,objective="val_accuracy",max_trials=8,seed=SEED,directory="/tmp/keras_tuner_part_b",project_name="lenet5_mnist",overwrite=True)
tuner.search(x_train,y_train,epochs=8,validation_data=(x_val,y_val),callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss",patience=2,restore_best_weights=True)],verbose=0)
tuner.results_summary(num_trials=8)

# %%
trial_rows=[]
for trial in tuner.oracle.trials.values():
    row={"trial_id":trial.trial_id,"val_accuracy":trial.score,"status":trial.status}; row.update(trial.hyperparameters.values); trial_rows.append(row)
best_trial_summary=pd.DataFrame(trial_rows).sort_values("val_accuracy",ascending=False)
display(best_trial_summary)
best_hyperparameters=tuner.get_best_hyperparameters(1)[0]
print("Best hyperparameters:", best_hyperparameters.values)

# %% [markdown]
# ## 4. Clean retraining and final comparison
#
# **What:** Rebuild the selected configuration from fresh weights and evaluate it once on the untouched test set.
# **Why:** A tuner trial is for selection; clean retraining avoids reporting a partially trained search state.

# %%
tuned_model=build_tunable_lenet(best_hyperparameters)
tuned_stop=tf.keras.callbacks.EarlyStopping(monitor="val_loss",patience=2,restore_best_weights=True)
tuned_history=tuned_model.fit(x_train,y_train,validation_data=(x_val,y_val),epochs=10,batch_size=best_hyperparameters.get("batch_size"),callbacks=[tuned_stop],verbose=2)


# %%
tuned_loss,tuned_accuracy=tuned_model.evaluate(x_test,y_test,verbose=0)
y_pred=tuned_model.predict(x_test,verbose=0).argmax(axis=1)
report=classification_report(y_test,y_pred,labels=np.arange(10),target_names=[str(i) for i in range(10)],output_dict=True,zero_division=0)
report_df=pd.DataFrame(report).T
tuned_metrics={"accuracy":float(tuned_accuracy),"loss":float(tuned_loss),"macro_f1":float(report["macro avg"]["f1-score"]),"weighted_f1":float(report["weighted avg"]["f1-score"])}
cm=confusion_matrix(y_test,y_pred,labels=np.arange(10))
assert cm.shape==(10,10) and len(y_pred)==len(y_test)
comparison=pd.DataFrame([
 {"model":"baseline","test_accuracy":float(baseline_model.evaluate(x_test,y_test,verbose=0)[1]),"parameters":baseline_model.count_params(),"val_accuracy":max(baseline_history.history["val_accuracy"])},
 {"model":"tuned","test_accuracy":tuned_metrics["accuracy"],"parameters":tuned_model.count_params(),"val_accuracy":max(tuned_history.history["val_accuracy"])}
])
display(comparison); display(report_df.round(3))
fig,ax=plt.subplots(figsize=(7,6)); image=ax.imshow(cm,cmap="Blues"); fig.colorbar(image,ax=ax); ax.set(xlabel="Predicted",ylabel="True",title="Tuned LeNet-5 confusion matrix"); ax.set_xticks(range(10)); ax.set_yticks(range(10))
for r in range(10):
    for c in range(10): ax.text(c,r,cm[r,c],ha="center",va="center",fontsize=8)
plt.tight_layout()

# %% [markdown]
# ## 4b. LeNet-5 Layer-by-Layer Activations & Latent Vector Trace
#
# **What:** Extract feature maps after each LeNet-5 layer (`conv1`, `pool1`, `conv2`, `pool2`, `conv3`, `flatten`, `dense1`, `classifier`).
# **Why:** Visualizing activations through classic LeNet-5 demonstrates how 32×32 inputs shrink through 5×5 kernels and pooling down to 84-dimensional dense latent vectors before final class selection.

# %%
lenet_trace_layers = ["conv1", "pool1", "conv2", "pool2", "conv3", "flatten", "dense1", "classifier"]
lenet_extractor = tf.keras.Model(inputs=tuned_model.input, outputs=[tuned_model.get_layer(l).output for l in lenet_trace_layers])

lenet_sample = x_test[:1]
lenet_acts = dict(zip(lenet_trace_layers, lenet_extractor.predict(lenet_sample, verbose=0)))

# 1. Conv1 Feature Maps (6 channels, 28x28)
fig, axes = plt.subplots(1, 6, figsize=(15, 3))
fig.suptitle(f"LeNet-5 Conv1 Feature Maps — Shape: {lenet_acts['conv1'].shape[1:]}", fontsize=12)
for idx, ax in enumerate(axes.ravel()):
    ax.imshow(lenet_acts['conv1'][0, ..., idx], cmap="viridis")
    ax.set_title(f"Conv1 Ch {idx + 1}"); ax.axis("off")
plt.tight_layout()

# 2. Pool1 Feature Maps (6 channels, 14x14)
fig, axes = plt.subplots(1, 6, figsize=(15, 3))
fig.suptitle(f"LeNet-5 Pool1 Feature Maps — Shape: {lenet_acts['pool1'].shape[1:]}", fontsize=12)
for idx, ax in enumerate(axes.ravel()):
    ax.imshow(lenet_acts['pool1'][0, ..., idx], cmap="viridis")
    ax.set_title(f"Pool1 Ch {idx + 1}"); ax.axis("off")
plt.tight_layout()

# 3. Conv2 Feature Maps (16 channels, 10x10)
fig, axes = plt.subplots(2, 8, figsize=(16, 4))
fig.suptitle(f"LeNet-5 Conv2 Feature Maps — Shape: {lenet_acts['conv2'].shape[1:]}", fontsize=12)
for idx, ax in enumerate(axes.ravel()):
    ax.imshow(lenet_acts['conv2'][0, ..., idx], cmap="plasma")
    ax.set_title(f"Ch {idx + 1}"); ax.axis("off")
plt.tight_layout()

# 4. Pool2 Feature Maps (16 channels, 5x5)
fig, axes = plt.subplots(2, 8, figsize=(16, 4))
fig.suptitle(f"LeNet-5 Pool2 Feature Maps — Shape: {lenet_acts['pool2'].shape[1:]}", fontsize=12)
for idx, ax in enumerate(axes.ravel()):
    ax.imshow(lenet_acts['pool2'][0, ..., idx], cmap="plasma")
    ax.set_title(f"Ch {idx + 1}"); ax.axis("off")
plt.tight_layout()

# 5. Latent Feature Vector (dense1: 84 units)
fig, (ax_lat, ax_cls) = plt.subplots(2, 1, figsize=(12, 6))
latent_vec = lenet_acts['dense1'][0]
ax_lat.bar(range(len(latent_vec)), latent_vec, color="teal", edgecolor="black", alpha=0.8)
ax_lat.set_title(f"LeNet-5 Latent Vector (Dense1 Layer) — Shape: ({len(latent_vec)},)")
ax_lat.set_xlabel("Latent Unit Index (0 to 83)")
ax_lat.set_ylabel("Activation (tanh)")

# Classifier Probabilities
class_probs = lenet_acts['classifier'][0]
ax_cls.bar(range(10), class_probs, color="coral", edgecolor="black")
ax_cls.set_title("Final Output Softmax Probabilities (Classes 0 to 9)")
ax_cls.set_xticks(range(10))
ax_cls.set_xlabel("Digit Class")
ax_cls.set_ylabel("Probability")
plt.tight_layout()

# %% [markdown]
# ## 4c. LeNet-5 Feature Hierarchy & Evolution
#
# | Layer | Output Shape | Feature Abstraction Level | Key Transformation |
# |---|---|---|---|
# | **Input** | `(32, 32, 1)` | Raw Padded Pixel Image | 28×28 digit padded to 32×32. |
# | **Conv1** | `(28, 28, 6)` | Low-level Edge Features | 5×5 kernels extract 6 spatial edge maps. |
# | **Pool1** | `(14, 14, 6)` | Spatial Subsampling | Average pooling downsizes resolution by 2×. |
# | **Conv2** | `(10, 10, 16)` | Mid-level Combination Features | Combines basic edges into curves and corners. |
# | **Pool2** | `(5, 5, 16)` | Spatial Subsampling | Reduces spatial resolution to 5×5. |
# | **Conv3** | `(1, 1, 120)` | Full Receptive Field Convolutions | 5×5 kernels spanning the entire 5×5 spatial map. |
# | **Dense1** | `(84,)` | **Latent Vector Representation** | 84-dimensional dense feature representation. |
# | **Classifier** | `(10,)` | Target Softmax Probabilities | Softmax output vector for digit classes 0–9. |

# %% [markdown]
# ## 5. Tuning safeguards
#
# | Mistake | Fix in this notebook |
# |---|---|
# | Tune on test data | The tuner sees only train and validation splits. |
# | Unbounded search | RandomSearch is limited to eight trials and short epochs. |
# | Select by training accuracy | The objective is `val_accuracy`. |
# | Reuse a partial trial | The best hyperparameters are rebuilt and retrained from fresh weights. |
# | Ignore cost | The comparison includes parameters and validation/test performance. |

# %% [markdown]
# ## 6. Recommended Future Enhancements
#
# 1. **Modern Activation Functions:** Replace classic `tanh` with `ReLU` or `LeakyReLU` to prevent vanishing gradients.
# 2. **Max Pooling over Average Pooling:** Use `MaxPooling2D` to retain dominant feature activations.
# 3. **Batch Normalization & Dropout:** Add `BatchNormalization()` after convolutions and `Dropout(0.3)` in dense layers to mitigate overfitting.

# %% [markdown]
# **References:** [KerasTuner getting started](https://keras.io/keras_tuner/getting_started/), [TensorFlow KerasTuner guide](https://www.tensorflow.org/tutorials/keras/keras_tuner), and [LeCun et al., 1998 LeNet-5 Paper](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf).


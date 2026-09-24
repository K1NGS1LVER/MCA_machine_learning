# MNIST CNN Filter and Complexity Lab Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create and execute a teaching notebook that explores MNIST CNN filters, layer structures, model complexity, and beginner-error prevention.

**Architecture:** Build one self-contained notebook with short markdown explanations followed by executable Python cells. Keep reusable helpers near the top: a fixed-kernel application helper, a shape/complexity reporter, and visualization utilities; later cells consume those helpers for preprocessing, filter exploration, CNN training, and evaluation.

**Tech Stack:** Python 3, Jupyter, TensorFlow/Keras, NumPy, Matplotlib, pandas, scikit-learn, nbformat, nbclient, parent `../.venv`.

## Global Constraints

- Use `applied_machine_oil/lab2/lab2_cnn_filters.ipynb`.
- Use the existing parent `machine_learning/.venv`; do not create a virtual environment under `lab2`.
- Use MNIST and keep the official test set untouched until final evaluation.
- Keep explanations to one-line “what / why” notes.
- Use assertions for invalid shapes, labels, and split sizes.
- Prefer existing dependencies and do not add unrelated packages.
- Validate with the parent environment, `nbformat`, and `nbclient`.

---

### Task 1: Create notebook setup and safe MNIST preprocessing

**Files:**
- Create: `applied_machine_oil/lab2/lab2_cnn_filters.ipynb`

**Interfaces:**
- Produces `x_train`, `x_val`, `x_test` with shape `(N, 28, 28, 1)` and `float32` values in `[0, 1]`.
- Produces `y_train`, `y_val`, `y_test` with integer labels in `[0, 9]`.
- Produces seeded imports and display helpers used by later tasks.

- [ ] **Step 1: Create the notebook with title, learning objectives, and environment cell**

  Add markdown explaining that the lab studies spatial dimensions, channels,
  fixed versus learned filters, and complexity. Add a code cell importing:

  ```python
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
  ```

  Seed Python, NumPy, and TensorFlow with `SEED = 42`, set NumPy print options,
  and print Python, TensorFlow, and NumPy versions.

- [ ] **Step 2: Load and inspect MNIST before transformation**

  Use `tf.keras.datasets.mnist.load_data()`, display a `2 x 5` sample grid,
  print the original train/test structures, and assert that images are rank 3,
  labels are rank 1, and labels are between 0 and 9.

- [ ] **Step 3: Split before preprocessing and transform all partitions**

  Use `train_test_split(..., test_size=0.10, stratify=y_train_raw,
  random_state=SEED)` on the original training partition only. Convert each
  image array with:

  ```python
  images = images.astype("float32") / 255.0
  images = images[..., np.newaxis]
  ```

  Assert channel-aware shapes, finite values, and disjoint expected split sizes.
  Print the structure after each transformation with one-line markdown.

- [ ] **Step 4: Commit the preprocessing notebook skeleton**

  Run `git diff --check`, then commit:

  ```bash
  git add applied_machine_oil/lab2/lab2_cnn_filters.ipynb
  git -c commit.gpgsign=false commit -m "feat: scaffold MNIST CNN filter lab"
  ```

### Task 2: Add custom and built-in filter exploration

**Files:**
- Modify: `applied_machine_oil/lab2/lab2_cnn_filters.ipynb`

**Interfaces:**
- Produces `apply_kernel(image, kernel) -> np.ndarray`.
- Produces `FILTERS`, a mapping of filter names to `3 x 3` float kernels.
- Produces `filter_responses`, a mapping of names to displayed response arrays.

- [ ] **Step 1: Define and test the custom kernel helper**

  Add:

  ```python
  def apply_kernel(image, kernel):
      image = np.asarray(image, dtype=np.float32)
      kernel = np.asarray(kernel, dtype=np.float32)
      assert image.ndim == 2 and kernel.shape == (3, 3)
      padded = np.pad(image, 1, mode="reflect")
      response = np.empty_like(image)
      for row in range(image.shape[0]):
          for col in range(image.shape[1]):
              response[row, col] = np.sum(
                  padded[row:row + 3, col:col + 3] * kernel
              )
      return response
  ```

  Apply it to one grayscale image and assert the result has the same height and
  width as the input.

- [ ] **Step 2: Add the required filter bank and visual explanation**

  Define horizontal Sobel, vertical Sobel, Laplacian, sharpen, box blur,
  Gaussian-like blur, and emboss kernels in `FILTERS`. Apply every kernel to
  `x_train[0, ..., 0]`, clip only for display, and show the original plus all
  responses in a compact grid. Put one-line markdown above the grid explaining
  the visual purpose of each filter.

- [ ] **Step 3: Demonstrate built-in convolution and pooling**

  Create deterministic Keras layers:

  ```python
  builtin_conv = tf.keras.layers.Conv2D(
      filters=4, kernel_size=3, padding="same", activation="relu",
      kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED),
  )
  builtin_pool = tf.keras.layers.MaxPooling2D(pool_size=2)
  ```

  Run one image batch through both layers, print `(1, 28, 28, 1)`,
  `(1, 28, 28, 4)`, and `(1, 14, 14, 4)`, and show several output maps.
  Explain that Keras convolution learns kernels while the fixed helper does not.

- [ ] **Step 4: Commit filter exploration**

  Execute the filter cells with the parent interpreter and commit:

  ```bash
  git add applied_machine_oil/lab2/lab2_cnn_filters.ipynb
  git -c commit.gpgsign=false commit -m "feat: explore custom and built-in CNN filters"
  ```

### Task 3: Add CNN, layer structure trace, and complexity analysis

**Files:**
- Modify: `applied_machine_oil/lab2/lab2_cnn_filters.ipynb`

**Interfaces:**
- Produces `model`, a compiled Keras classifier.
- Produces `complexity_table`, a pandas DataFrame with layer, output shape,
  parameters, activations, and approximate MACs.
- Produces `trace_model_shapes(model, input_shape) -> pd.DataFrame`.

- [ ] **Step 1: Build the readable CNN**

  Use this architecture:

  ```python
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
  model.compile(
      optimizer=tf.keras.optimizers.Adam(),
      loss="sparse_categorical_crossentropy",
      metrics=["accuracy"],
  )
  ```

  Print `model.summary()` and add one-line markdown explaining each spatial or
  channel change, including the explicit `(28, 28, 1) -> (28, 28, 8) ->
  (14, 14, 8)` example.

- [ ] **Step 2: Implement a shape trace**

  Implement `trace_model_shapes` by creating a zero input tensor and forwarding
  it through each model layer, recording layer name and output shape in a
  DataFrame. Assert that the first output is `(1, 28, 28, 1)` and that pooling
  changes spatial dimensions from 28 to 14 and from 14 to 7.

- [ ] **Step 3: Implement complexity calculations**

  For each layer, record parameter count and activation element count. For each
  `Conv2D`, compute MACs using:

  ```python
  output_height * output_width * output_channels * kernel_height \
      * kernel_width * input_channels
  ```

  For dense layers, calculate parameters as `inputs * outputs + outputs`.
  Assert that the table total equals `model.count_params()`. Display totals and
  one-line markdown distinguishing parameters, activations, and MACs.

- [ ] **Step 4: Commit the model and complexity analysis**

  Run the shape and complexity cells, then commit:

  ```bash
  git add applied_machine_oil/lab2/lab2_cnn_filters.ipynb
  git -c commit.gpgsign=false commit -m "feat: add CNN shape and complexity analysis"
  ```

### Task 4: Add training, evaluation, learned filters, and error audit

**Files:**
- Modify: `applied_machine_oil/lab2/lab2_cnn_filters.ipynb`

**Interfaces:**
- Produces `history`, `test_metrics`, `report_df`, and `cm`.
- Produces learned kernel and feature-map visualizations.
- Produces the final beginner-error audit and references.

- [ ] **Step 1: Train with validation monitoring and early stopping**

  Add:

  ```python
  early_stopping = tf.keras.callbacks.EarlyStopping(
      monitor="val_loss", patience=2, restore_best_weights=True
  )
  history = model.fit(
      x_train, y_train, validation_data=(x_val, y_val),
      epochs=12, batch_size=128, callbacks=[early_stopping], verbose=2,
  )
  ```

  Plot training versus validation loss and accuracy, and state in one line why
  validation monitoring and restored best weights reduce overfitting risk.

- [ ] **Step 2: Evaluate only on the untouched test set**

  Call `model.evaluate(x_test, y_test, verbose=0)`, predict with
  `model.predict(x_test, verbose=0)`, and create a classification report with
  `output_dict=True`. Display macro and weighted precision/recall/F1, accuracy,
  and a labeled `10 x 10` confusion matrix. Assert prediction count equals
  `len(y_test)`.

- [ ] **Step 3: Visualize learned kernels and feature maps**

  Read `conv1.get_weights()[0]`, show all eight learned `3 x 3` kernels, then
  create an intermediate model ending at `conv1`, display several feature maps
  for one test image, and explain in one line that these filters are learned
  from training data rather than hand-authored.

- [ ] **Step 4: Add the beginner-error audit and references**

  Add a concise markdown table with these rows:

  | Mistake | Notebook fix |
  |---|---|
  | Data leakage | Split before transformations and keep test data isolated |
  | Test-set tuning | Use train/validation for model decisions |
  | No overfitting check | Plot validation curves and use early stopping |
  | Shape/channel error | Assert rank and print every layer structure |
  | Uncontrolled randomness | Seed Python, NumPy, and TensorFlow |
  | Accuracy-only evaluation | Report macro/weighted metrics and confusion matrix |
  | Misread filter output | Separate display clipping from model input |

  Link scikit-learn common pitfalls, Google/TensorFlow overfitting guidance,
  Keras `Conv2D`, and Keras `EarlyStopping` documentation.

- [ ] **Step 5: Execute and validate the finished notebook**

  From the repository root, run:

  ```bash
  ../.venv/bin/python -m jupyter nbconvert --to notebook --execute \
      applied_machine_oil/lab2/lab2_cnn_filters.ipynb \
      --output-dir applied_machine_oil/lab2 \
      --output lab2_cnn_filters_executed.ipynb
  ```

  Then validate all cells:

  ```bash
  ../.venv/bin/python - <<'PY'
  import nbformat
  from pathlib import Path
  path = Path("applied_machine_oil/lab2/lab2_cnn_filters_executed.ipynb")
  notebook = nbformat.read(path, as_version=4)
  errors = [cell for cell in notebook.cells
            if cell.cell_type == "code"
            for output in cell.get("outputs", [])
            if output.output_type == "error"]
  assert not errors, errors
  assert any("complexity_table" in cell.source for cell in notebook.cells)
  assert any("confusion_matrix" in cell.source for cell in notebook.cells)
  print(f"Validated {len(notebook.cells)} cells with no execution errors.")
  PY
  ```

  Replace the source notebook with the executed notebook only if it is
  repository convention to commit populated outputs; otherwise retain the
  clean source notebook and remove the temporary executed artifact.

- [ ] **Step 6: Commit the completed notebook**

  Run `git diff --check`, inspect the final notebook status, and commit:

  ```bash
  git add applied_machine_oil/lab2/lab2_cnn_filters.ipynb
  git -c commit.gpgsign=false commit -m "feat: complete MNIST CNN filter and complexity lab"
  ```

## Self-review checklist

- The setup task covers Python/TensorFlow imports, reproducibility, MNIST, safe
  splitting, normalization, channel shape, and assertions.
- The filter task covers all seven requested fixed filters, a reusable user
  function, built-in Keras filters/pooling, and visual comparison.
- The complexity task covers every layer's structure plus parameter, activation,
  and convolution MAC calculations with formulas from the spec.
- The evaluation task covers early stopping, untouched test metrics, confusion
  matrix, learned filters, one-line explanations, researched references, and
  executable validation.
- No placeholder instructions, undefined neighboring interfaces, or local
  environment creation steps remain.

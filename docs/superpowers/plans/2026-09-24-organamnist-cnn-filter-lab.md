# OrganAMNIST CNN Filter and Complexity Lab Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the executed MNIST notebook with an executed OrganAMNIST medical-image CNN lab that teaches filters, augmentation, class imbalance, custom Keras code, layer shapes, and complexity.

**Architecture:** Keep one self-contained notebook and replace dataset-specific MNIST cells with MedMNIST `OrganAMNIST` official train/validation/test loading. Insert focused educational cells for medical-data inspection, training-only augmentation, class-weighted learning, and a custom fixed-kernel Keras layer; retain the existing filter, shape-trace, complexity, learned-feature-map, and evaluation sections with dynamic 11-class metadata.

**Tech Stack:** Python 3.14, Jupyter, TensorFlow/Keras 2.22 prerelease, MedMNIST, NumPy, Matplotlib, pandas, scikit-learn, nbformat, nbclient, parent `../.venv`.

## Global Constraints

- Use `applied_machine_oil/lab2/lab2_cnn_filters.ipynb`.
- Use the existing parent `machine_learning/.venv`; do not create an environment under `lab2`.
- Use the official OrganAMNIST `train`, `val`, and `test` splits.
- Keep the official test split untouched until final evaluation.
- Keep explanations to short one-line “what / why” notes.
- Use assertions for invalid shapes, labels, split sizes, and class counts.
- Treat resized CT slices as educational benchmarking data, not clinical diagnostic evidence.
- Install only `medmnist` if it is missing from the parent environment.
- Preserve populated outputs in the committed notebook after successful execution.

---

### Task 1: Replace MNIST loading with OrganAMNIST metadata and preprocessing

**Files:**
- Modify: `applied_machine_oil/lab2/lab2_cnn_filters.ipynb`

**Interfaces:**
- Produces `train_images`, `val_images`, `test_images` with shape `(N, 28, 28, 1)` and float values in `[0, 1]`.
- Produces `y_train`, `y_val`, `y_test` as rank-1 integer arrays with labels `0..10`.
- Produces `class_names`, `num_classes`, and `split_summary`.

- [ ] **Step 1: Ensure MedMNIST is available in the parent environment**

  Run from the repository root:

  ```bash
  ../../.venv/bin/python -c "import medmnist; print(medmnist.__version__)"
  ```

  If that import fails, install only the missing package:

  ```bash
  uv pip install --python ../../.venv/bin/python medmnist
  ```

- [ ] **Step 2: Replace imports and dataset loading**

  Add:

  ```python
  import medmnist
  from medmnist import INFO
  from medmnist.dataset import OrganAMNIST
  from sklearn.utils.class_weight import compute_class_weight
  ```

  Derive metadata:

  ```python
  DATA_FLAG = "organamnist"
  info = INFO[DATA_FLAG]
  class_names = [info["label"][str(i)] for i in range(len(info["label"]))]
  num_classes = len(class_names)
  assert num_classes == 11
  ```

  Load `OrganAMNIST(split="train", download=True)`, `split="val"`, and
  `split="test"` separately. Use `.imgs` and `.labels` so the official splits
  remain explicit.

- [ ] **Step 3: Normalize labels and images with assertions**

  Implement:

  ```python
  def prepare_medical_split(dataset):
      images = dataset.imgs.astype("float32") / 255.0
      labels = dataset.labels.reshape(-1).astype("int64")
      images = images[..., np.newaxis]
      assert images.shape[1:] == (28, 28, 1)
      assert labels.shape[0] == images.shape[0]
      assert labels.min() >= 0 and labels.max() < num_classes
      assert np.isfinite(images).all() and 0.0 <= images.min() <= images.max() <= 1.0
      return images, labels
  ```

  Produce `train_images`, `val_images`, `test_images`, labels, and a
  `split_summary` table with split size and shape.

- [ ] **Step 4: Add labeled samples and class distribution**

  Display a `2 x 5` image grid with organ names from `class_names`, print raw
  and prepared structures, and display a training class-count table. Add the
  one-line medical caution that resized slices are educational examples, not
  complete patient studies.

- [ ] **Step 5: Commit the dataset conversion**

  ```bash
  git add applied_machine_oil/lab2/lab2_cnn_filters.ipynb
  git -c commit.gpgsign=false commit -m "feat: switch CNN lab to OrganAMNIST"
  ```

### Task 2: Add augmentation, imbalance handling, and custom Keras layer

**Files:**
- Modify: `applied_machine_oil/lab2/lab2_cnn_filters.ipynb`

**Interfaces:**
- Produces `augmentation`, `class_weights`, and `weighted_model`.
- Produces `FixedKernelLayer`, a custom Keras layer.
- Produces `augmented_preview`.

- [ ] **Step 1: Demonstrate training-only augmentation**

  Define:

  ```python
  augmentation = tf.keras.Sequential([
      tf.keras.layers.RandomRotation(0.05, seed=SEED),
      tf.keras.layers.RandomTranslation(0.05, 0.05, seed=SEED),
  ], name="training_augmentation")
  ```

  Apply it only to a training batch, display original/augmented pairs, and
  assert the batch remains `(batch, 28, 28, 1)`. State that validation and test
  arrays are never augmented.

- [ ] **Step 2: Compute class weights from training labels only**

  Use:

  ```python
  class_ids = np.arange(num_classes)
  weights = compute_class_weight("balanced", classes=class_ids, y=y_train)
  class_weights = dict(zip(class_ids.tolist(), weights.tolist()))
  ```

  Display a table containing class name, sample count, and weight. Assert all
  weights are finite and positive.

- [ ] **Step 3: Implement and run the custom fixed-kernel layer**

  Add:

  ```python
  class FixedKernelLayer(tf.keras.layers.Layer):
      def __init__(self, kernel, **kwargs):
          super().__init__(**kwargs)
          kernel = np.asarray(kernel, dtype="float32")
          assert kernel.shape == (3, 3)
          self.kernel_values = kernel

      def build(self, input_shape):
          channels = int(input_shape[-1])
          kernel = np.repeat(self.kernel_values[:, :, None, None], channels, axis=2)
          self.kernel = tf.constant(kernel, dtype=tf.float32)

      def call(self, inputs):
          return tf.nn.depthwise_conv2d(
              inputs, self.kernel, strides=[1, 1, 1, 1], padding="SAME"
          )
  ```

  Run it on one image batch, print input/output structures, and compare the
  output shape with `apply_kernel` for one channel. Explain that this layer has
  no trainable weights and demonstrates custom TensorFlow extension.

- [ ] **Step 4: Commit educational extensions**

  ```bash
  git add applied_machine_oil/lab2/lab2_cnn_filters.ipynb
  git -c commit.gpgsign=false commit -m "feat: add augmentation imbalance and custom layer lessons"
  ```

### Task 3: Adapt filters, CNN shape trace, complexity, and classifier

**Files:**
- Modify: `applied_machine_oil/lab2/lab2_cnn_filters.ipynb`

**Interfaces:**
- Produces `model`, `shape_table`, and `complexity_table` for 11 classes.
- Produces `filter_responses` and learned feature maps from OrganAMNIST.

- [ ] **Step 1: Reuse fixed filters on an OrganAMNIST slice**

  Keep the existing seven kernels and `apply_kernel` function, but use
  `train_images[0, ..., 0]`. Keep clipping limited to plotting and assert every
  response remains `(28, 28)`.

- [ ] **Step 2: Build the 11-class CNN**

  Preserve the existing convolution/pooling/dropout architecture, but define
  the classifier with:

  ```python
  tf.keras.layers.Dense(num_classes, activation="softmax", name="classifier")
  ```

  Compile with sparse categorical cross-entropy. Use `train_images`, `y_train`,
  and `val_images`, `y_val`. Print the dynamic structure path ending at
  `(7, 7, 16) -> (num_classes,)`.

- [ ] **Step 3: Preserve dynamic shape tracing**

  Keep `trace_model_shapes(model, input_shape) -> pd.DataFrame` and assert
  `(28, 28, 1)`, `(14, 14, 8)`, and `(7, 7, 16)` outputs. Do not assert a
  ten-class output; assert the final output width equals `num_classes`.

- [ ] **Step 4: Recompute complexity dynamically**

  Keep `complexity_rows(model, input_shape) -> pd.DataFrame`, parameter
  formulas, activation counts, and MAC formulas. Assert the parameter total
  equals `model.count_params()` and display the final dense row using
  `num_classes`, not a hard-coded MNIST value.

- [ ] **Step 5: Commit the 11-class model**

  ```bash
  git add applied_machine_oil/lab2/lab2_cnn_filters.ipynb
  git -c commit.gpgsign=false commit -m "feat: adapt CNN filters and complexity to organs"
  ```

### Task 4: Train with class weights and evaluate OrganAMNIST

**Files:**
- Modify: `applied_machine_oil/lab2/lab2_cnn_filters.ipynb`

**Interfaces:**
- Produces `history`, `test_metrics`, `report_df`, and `cm`.
- Produces an organ-labeled confusion matrix and learned kernel/feature-map
  figures.

- [ ] **Step 1: Train using augmentation and class weights**

  Insert `augmentation` as the first layer of the training model or apply it
  inside the model while leaving validation/test arrays unchanged. Train with:

  ```python
  early_stopping = tf.keras.callbacks.EarlyStopping(
      monitor="val_loss", patience=3, restore_best_weights=True
  )
  history = model.fit(
      train_images, y_train,
      validation_data=(val_images, y_val),
      epochs=20, batch_size=128,
      class_weight=class_weights,
      callbacks=[early_stopping],
      verbose=2,
  )
  ```

  Plot training/validation loss and accuracy and state why augmentation and
  class weights address different problems.

- [ ] **Step 2: Evaluate on the untouched official test split**

  Call `model.evaluate(test_images, y_test, verbose=0)`, predict once, and
  create `classification_report(..., labels=np.arange(num_classes),
  target_names=class_names, output_dict=True, zero_division=0)`. Display
  accuracy, macro/weighted precision/recall/F1, and per-class metrics. Assert
  prediction count and confusion-matrix shape `(num_classes, num_classes)`.

- [ ] **Step 3: Draw organ-labeled confusion matrix and learned maps**

  Label both axes with `class_names`, display learned first-layer kernels, and
  produce feature maps from `test_images[:1]` using an explicit Keras input
  tensor compatible with Keras 3:

  ```python
  feature_input = tf.keras.Input(shape=(28, 28, 1))
  feature_model = tf.keras.Model(feature_input, conv1(feature_input))
  ```

- [ ] **Step 4: Expand the beginner-error audit**

  Add rows for class imbalance and medical overinterpretation, and link
  MedMNIST documentation along with existing Keras/scikit-learn/Google
  references. Explain that the experiment is educational and non-clinical.

- [ ] **Step 5: Execute and validate the notebook**

  Install `medmnist` first if needed, then run from the repository root:

  ```bash
  ../../.venv/bin/python -m jupyter nbconvert --to notebook --execute \
      applied_machine_oil/lab2/lab2_cnn_filters.ipynb \
      --output-dir applied_machine_oil/lab2 \
      --output lab2_cnn_filters_executed.ipynb \
      --ExecutePreprocessor.timeout=900
  ```

  Validate:

  ```bash
  ../../.venv/bin/python - <<'PY'
  import nbformat
  from pathlib import Path
  path = Path("applied_machine_oil/lab2/lab2_cnn_filters_executed.ipynb")
  nb = nbformat.read(path, as_version=4)
  errors = [(i, o["ename"], o["evalue"])
            for i, cell in enumerate(nb.cells)
            if cell.cell_type == "code"
            for o in cell.get("outputs", [])
            if o.output_type == "error"]
  assert not errors, errors
  source = "\n".join("".join(c.source) for c in nb.cells)
  assert "OrganAMNIST" in source and "num_classes" in source
  assert "class_weights" in source and "FixedKernelLayer" in source
  print(f"Validated {len(nb.cells)} cells with no execution errors.")
  PY
  ```

  Replace the source notebook with the executed notebook after validation and
  remove the temporary executed copy.

- [ ] **Step 6: Commit the completed notebook**

  ```bash
  git add applied_machine_oil/lab2/lab2_cnn_filters.ipynb
  git -c commit.gpgsign=false commit -m "feat: complete OrganAMNIST CNN medical lab"
  ```

## Self-review checklist

- Dataset task covers official MedMNIST splits, medical sample labels,
  preprocessing, class names, class counts, and safety assertions.
- Extension task covers training-only augmentation, training-only class weights,
  and the custom non-trainable Keras layer.
- Model task preserves all seven filters, built-in operations, 11-class CNN,
  dynamic shape trace, and dynamic complexity formulas.
- Evaluation task covers early stopping, untouched test metrics, organ-labeled
  confusion matrix, learned maps, medical caution, and executed validation.
- No local environment creation, hard-coded ten-class assumptions, or
  unbounded placeholder steps remain.

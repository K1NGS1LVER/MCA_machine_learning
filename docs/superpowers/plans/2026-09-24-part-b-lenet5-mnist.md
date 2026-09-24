# Part B LeNet-5 MNIST Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create and execute a LeNet-5 MNIST notebook with bounded KerasTuner search, then generate PDFs for both executed Part A and Part B notebooks.

**Architecture:** Create a separate Part B notebook with shared educational conventions from Part A: short what/why markdown, explicit shape and complexity tables, leakage-safe splits, and final untouched-test evaluation. Use a classic LeNet baseline, KerasTuner RandomSearch for a small bounded tuning study, cleanly retrain the selected model, and convert both populated notebooks to PDFs.

**Tech Stack:** Python 3.14, TensorFlow/Keras 2.22 prerelease, KerasTuner, Jupyter, nbformat, nbclient, nbconvert, Pandas, NumPy, Matplotlib, scikit-learn, parent `../../.venv`.

## Global Constraints

- Use `applied_machine_oil/lab2/part_b_lenet5_mnist.ipynb`.
- Preserve `applied_machine_oil/lab2/lab2_cnn_filters.ipynb` unchanged.
- Use the existing parent `machine_learning/.venv`; do not create a local environment under `lab2`.
- Keep the official MNIST test partition untouched until final evaluation.
- Zero-pad MNIST images from 28×28 to 32×32 before adding the channel dimension.
- Use classic LeNet choices (`tanh`, average pooling) intentionally and explain them.
- Tune only with training and validation data; never use the test set for selection.
- Keep tuner artifacts outside the repository or remove them after execution.
- Preserve executed notebook outputs before PDF conversion.

---

### Task 1: Create the executed Part B LeNet notebook

**Files:**
- Create: `applied_machine_oil/lab2/part_b_lenet5_mnist.ipynb`

**Interfaces:**
- Produces `x_train`, `x_val`, `x_test` shaped `(N, 32, 32, 1)`.
- Produces `y_train`, `y_val`, `y_test` with labels `0..9`.
- Produces `baseline_model`, `baseline_complexity`, `tuned_model`, `best_hyperparameters`, `tuned_metrics`, and `cm`.

- [ ] **Step 1: Install and verify KerasTuner in the parent environment**

  Run:

  ```bash
  ../../.venv/bin/python -c "import keras_tuner; print(keras_tuner.__version__)"
  ```

  If missing, install only the dependency:

  ```bash
  uv pip install --python ../../.venv/bin/python keras-tuner
  ```

- [ ] **Step 2: Add imports, seeds, and MNIST preprocessing**

  Import TensorFlow, `keras_tuner as kt`, NumPy, Pandas, Matplotlib,
  `train_test_split`, classification metrics, and `Path`. Seed Python, NumPy,
  and TensorFlow with `SEED = 42`.

  Load MNIST, split only the original training partition with
  `test_size=0.10`, `stratify=y_train_raw`, and `random_state=SEED`. Normalize
  to `float32` `[0, 1]`, pad using:

  ```python
  images = np.pad(images, ((0, 0), (2, 2), (2, 2)), mode="constant")
  images = images[..., np.newaxis]
  ```

  Assert structures `(N, 32, 32, 1)`, finite ranges, and labels in `0..9`.

- [ ] **Step 3: Add baseline LeNet-5 and complexity helpers**

  Define `build_baseline_lenet()` with:

  ```python
  Input((32, 32, 1))
  Conv2D(6, 5, activation="tanh")
  AveragePooling2D(2)
  Conv2D(16, 5, activation="tanh")
  AveragePooling2D(2)
  Conv2D(120, 5, activation="tanh")
  Flatten()
  Dense(84, activation="tanh")
  Dense(10, activation="softmax")
  ```

  Compile with Adam and sparse categorical cross-entropy. Add a shape/parameter/
  activation/MAC helper and assert the final output width is 10. Explain that
  average pooling and `tanh` are historical LeNet choices.

- [ ] **Step 4: Train and evaluate the baseline**

  Train with validation data, early stopping, and a bounded epoch count. Produce
  baseline curves, classification metrics, and a 10×10 confusion matrix without
  using test data for model selection.

- [ ] **Step 5: Commit the initial Part B notebook**

  ```bash
  git add applied_machine_oil/lab2/part_b_lenet5_mnist.ipynb
  git -c commit.gpgsign=false commit -m "feat: add Part B LeNet5 MNIST notebook"
  ```

### Task 2: Add bounded KerasTuner search and final evaluation

**Files:**
- Modify: `applied_machine_oil/lab2/part_b_lenet5_mnist.ipynb`

**Interfaces:**
- Produces `tuner`, `best_hyperparameters`, `best_trial_summary`, and `tuned_model`.
- Produces `tuned_metrics`, `report_df`, and `cm` for the selected model.

- [ ] **Step 1: Define the tunable model builder**

  Define `build_tunable_lenet(hp)` with these choices:

  ```python
  filters1 = hp.Choice("filters1", [6, 12])
  filters2 = hp.Choice("filters2", [16, 32])
  dense_units = hp.Choice("dense_units", [84, 120])
  dropout_rate = hp.Choice("dropout_rate", [0.0, 0.2, 0.4])
  learning_rate = hp.Choice("learning_rate", [1e-3, 3e-4])
  ```

  Preserve the LeNet ordering and use dropout only after flatten/dense layers.
  Compile with Adam at the selected learning rate.

- [ ] **Step 2: Run bounded RandomSearch**

  Create a tuner with:

  ```python
  tuner = kt.RandomSearch(
      build_tunable_lenet,
      objective="val_accuracy",
      max_trials=8,
      seed=SEED,
      directory="/tmp/keras_tuner_part_b",
      project_name="lenet5_mnist",
      overwrite=True,
  )
  ```

  Search with `epochs=8`, `batch_size=128`, validation data, and early stopping.
  Display `tuner.results_summary()` and a DataFrame containing each trial’s
  score and hyperparameters.

- [ ] **Step 3: Retrain the selected hyperparameters cleanly**

  Retrieve `best_hyperparameters = tuner.get_best_hyperparameters(1)[0]`,
  rebuild `tuned_model = build_tunable_lenet(best_hyperparameters)`, and train
  it from fresh weights with early stopping. Do not reuse a partially trained
  trial model.

- [ ] **Step 4: Evaluate once on the untouched test set**

  Evaluate the tuned model on `x_test`, predict labels, produce macro/weighted
  precision/recall/F1, and draw the final labeled 10×10 confusion matrix.
  Assert prediction count and confusion-matrix shape. Compare baseline and tuned
  test metrics plus parameter counts and complexity.

- [ ] **Step 5: Add the tuning safeguards and references**

  Add a concise table covering test-set tuning leakage, unbounded searches,
  inconsistent trial budgets, training-accuracy selection, failure to retrain,
  and ignoring runtime/parameter count. Link official KerasTuner and TensorFlow
  tuning documentation.

### Task 3: Execute both notebooks and generate PDFs

**Files:**
- Modify: `applied_machine_oil/lab2/lab2_cnn_filters.ipynb` only if re-execution is needed.
- Create: `applied_machine_oil/lab2/part_b_lenet5_mnist.pdf`
- Create or refresh: `applied_machine_oil/lab2/lab2_cnn_filters.pdf`

**Interfaces:**
- Both notebooks have populated outputs and no execution-error cells.
- Both PDFs are generated from executed notebooks.

- [ ] **Step 1: Execute Part B**

  Run:

  ```bash
  ../../.venv/bin/python -m jupyter nbconvert --to notebook --execute \
      applied_machine_oil/lab2/part_b_lenet5_mnist.ipynb \
      --output-dir applied_machine_oil/lab2 \
      --output part_b_lenet5_mnist_executed.ipynb \
      --ExecutePreprocessor.timeout=1200
  ```

  Validate no error outputs, then replace the source notebook with the executed
  copy and remove the temporary copy.

- [ ] **Step 2: Verify Part A execution**

  Read `lab2_cnn_filters.ipynb` with `nbformat` and assert that code cells have
  outputs and no output has `output_type == "error"`. Re-execute Part A only if
  those conditions are not met.

- [ ] **Step 3: Convert both executed notebooks to PDF**

  Run:

  ```bash
  ../../.venv/bin/python -m jupyter nbconvert --to pdf \
      applied_machine_oil/lab2/lab2_cnn_filters.ipynb \
      --output-dir applied_machine_oil/lab2 \
      --output lab2_cnn_filters.pdf
  ../../.venv/bin/python -m jupyter nbconvert --to pdf \
      applied_machine_oil/lab2/part_b_lenet5_mnist.ipynb \
      --output-dir applied_machine_oil/lab2 \
      --output part_b_lenet5_mnist.pdf
  ```

  If PDF conversion requires a missing TeX dependency, use the available
  `webpdf` exporter only if Chromium is installed; otherwise report the exact
  missing converter rather than claiming PDF completion.

- [ ] **Step 4: Validate final artifacts and commit**

  Assert both notebooks contain no errors, both PDFs exist and are non-empty,
  run `git diff --check`, and commit:

  ```bash
  git add applied_machine_oil/lab2/lab2_cnn_filters.ipynb \
      applied_machine_oil/lab2/part_b_lenet5_mnist.ipynb \
      applied_machine_oil/lab2/lab2_cnn_filters.pdf \
      applied_machine_oil/lab2/part_b_lenet5_mnist.pdf
  git -c commit.gpgsign=false commit -m "feat: complete Part B LeNet5 and notebook PDFs"
  ```

## Self-review checklist

- MNIST preprocessing, 32×32 padding, baseline LeNet-5, and complexity are
  covered in Task 1.
- Bounded KerasTuner search, validation-only selection, clean retraining, and
  final test evaluation are covered in Task 2.
- Both executed notebooks and both PDFs are covered in Task 3.
- No local environment creation, test-set tuning, or unbounded search remains.

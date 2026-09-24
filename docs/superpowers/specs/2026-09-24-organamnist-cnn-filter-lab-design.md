# OrganAMNIST CNN Filter and Complexity Lab Design

## Goal

Revise `applied_machine_oil/lab2/lab2_cnn_filters.ipynb` from MNIST to the
small, educational MedMNIST `OrganAMNIST` medical-image dataset while
preserving the existing filter exploration, layer-structure tracing, CNN
complexity, and beginner-error audit.

OrganAMNIST contains resized 28×28 grayscale abdominal CT slices for
11-organ classification. The notebook must explicitly state that these slices
are for educational benchmarking and are not clinical diagnostic data.

## Dataset and preprocessing

- Use the `medmnist` package and `OrganAMNIST` dataset class.
- Load the official `train`, `val`, and `test` splits separately.
- Keep the official test split untouched until final evaluation.
- Report raw images as `(N, 28, 28)` and CNN inputs as `(N, 28, 28, 1)`.
- Convert labels from `(N, 1)` to `(N,)` explicitly and assert class IDs are
  in `0..10`.
- Convert images to `float32`, scale to `[0, 1]`, and assert finite values.
- Print the dataset class names and display a compact class-distribution table.
- Show labeled sample CT slices with organ names.
- Preserve short one-line “what / why” explanations.

## Filters and model

- Preserve the reusable custom `apply_kernel(image, kernel)` helper.
- Preserve fixed horizontal Sobel, vertical Sobel, Laplacian, sharpen, box blur,
  Gaussian-like blur, and emboss filters.
- Preserve the built-in Keras `Conv2D` and `MaxPooling2D` demonstration.
- Apply fixed filters to grayscale OrganAMNIST slices and clearly distinguish
  fixed image-processing kernels from learned CNN kernels.
- Preserve the two-convolution/two-pooling CNN with dropout and dense layers.
- Change the final classifier to `Dense(11, activation="softmax")`.
- Use sparse categorical cross-entropy and derive `num_classes` from the dataset
  rather than hard-coding ten-class assumptions outside the dataset check.
- Print the shape path:
  `(28, 28, 1) -> (28, 28, 8) -> (14, 14, 8) -> (14, 14, 16) ->
  (7, 7, 16)`.
- Visualize first-layer learned kernels and feature maps from an OrganAMNIST
  slice.

## Complexity analysis

- Preserve the layer-by-layer output structure table.
- Recompute trainable parameters from the actual model and assert the total
  matches `model.count_params()`.
- Preserve activation-element counts.
- Preserve approximate convolution MAC calculations:

  `output_height * output_width * output_channels * kernel_height *
  kernel_width * input_channels`

- Preserve dense parameter calculations:

  `inputs * outputs + outputs`

- Explain separately that parameters represent stored learnable capacity and
  MACs estimate computation.
- Do not hard-code the previous MNIST parameter total.

## Training and evaluation

- Train with validation-loss monitoring and early stopping that restores best
  weights.
- Evaluate only once on the untouched official test split.
- Report accuracy, macro and weighted precision/recall/F1, per-class metrics,
  and an 11×11 confusion matrix labeled with organ names.
- Include a class-imbalance note explaining why accuracy alone is insufficient.
- Add a medical-data caution that resized CT slices are not complete patient
  studies and model results are not clinical validation.

## Beginner-error audit and references

Preserve the existing audit for data leakage, test-set tuning, overfitting,
shape/channel errors, uncontrolled randomness, accuracy-only evaluation, and
display normalization. Add:

- class imbalance: use distribution inspection, macro metrics, and a confusion
  matrix;
- medical overinterpretation: clearly label the experiment as educational and
  non-clinical.

Keep links to MedMNIST documentation, scikit-learn common pitfalls, Google or
TensorFlow overfitting guidance, Keras `Conv2D`, and Keras `EarlyStopping`.

## Environment and validation

- Use the existing parent `machine_learning/.venv`; do not create an
  environment under `lab2`.
- Install only `medmnist` into the parent environment if missing.
- Execute the revised notebook with the parent interpreter.
- Validate JSON and notebook format, then assert no error outputs.
- Verify outputs include the 11-class shape/complexity results, class names,
  metrics, confusion matrix, filter plots, and learned feature maps.
- Preserve populated outputs in the committed notebook if repository convention
  continues to favor executed notebooks.


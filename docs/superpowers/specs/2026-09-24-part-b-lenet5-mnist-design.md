# Part B: LeNet-5 MNIST and Hyperparameter Tuning Design

## Goal

Create a separate executed notebook at
`applied_machine_oil/lab2/part_b_lenet5_mnist.ipynb` that applies a classic
LeNet-5-style architecture to MNIST and demonstrates bounded hyperparameter
tuning with KerasTuner. Preserve the existing OrganAMNIST Part A notebook
unchanged.

## Dataset and preprocessing

- Load MNIST with TensorFlow/Keras.
- Keep train, validation, and official test data separate.
- Split only the original training partition into train and validation.
- Keep the official test partition untouched until final evaluation.
- Convert images to `float32`, normalize to `[0, 1]`, and zero-pad 28×28 images
  to 32×32 to match the classic LeNet-5 input convention.
- Add an explicit channel dimension, producing `(N, 32, 32, 1)`.
- Assert image rank, channel shape, finite range, label rank, and label range.
- Display representative images and class counts.

## Baseline LeNet-5 model

Use a readable classic LeNet-style baseline:

1. `Conv2D(6, 5×5, activation="tanh")`
2. `AveragePooling2D(2)`
3. `Conv2D(16, 5×5, activation="tanh")`
4. `AveragePooling2D(2)`
5. `Conv2D(120, 5×5, activation="tanh")`
6. `Flatten`
7. `Dense(84, activation="tanh")`
8. `Dense(10, activation="softmax")`

The notebook must explain that average pooling and `tanh` are intentional
historical LeNet choices, not accidental omissions of modern ReLU/max-pooling
defaults.

Include:

- layer-by-layer output structures;
- trainable parameter counts;
- activation counts;
- approximate convolution MACs;
- one-line “what / why” explanations;
- baseline training curves and validation metrics.

## KerasTuner search

Use `keras-tuner.RandomSearch` with a bounded, reproducible space:

- first convolution filters: 6 or 12;
- second convolution filters: 16 or 32;
- dense units: 84 or 120;
- dropout: 0.0, 0.2, or 0.4;
- learning rate: `1e-3` or `3e-4`;
- batch size: 64 or 128.

Limit the search to at most 8 trials with a short epoch budget and early
stopping. Optimize `val_accuracy` using only training and validation data.
Display the trial summary and selected hyperparameters.

After selection, rebuild and retrain the best configuration cleanly rather than
reusing a trial's partially trained state. Evaluate the selected model exactly
once on the untouched test set.

Install only `keras-tuner` into the existing parent `.venv` if it is missing.
Do not create a local environment under `lab2`.

## Comparison and evaluation

Compare baseline and tuned models using:

- selected hyperparameters;
- validation accuracy;
- final test accuracy;
- macro and weighted precision/recall/F1;
- parameter count;
- approximate complexity;
- final confusion matrix.

Include concise tuning safeguards:

- never tune on the test set;
- keep trial budgets bounded;
- use consistent epochs, callbacks, and validation data;
- choose by validation performance rather than training accuracy;
- retrain the selected configuration;
- consider runtime and parameter count alongside accuracy.

## Validation

- Execute the notebook with the parent `../.venv`.
- Validate notebook format and assert no error outputs.
- Verify baseline and tuned metrics, shape/complexity tables, tuner summary,
  and confusion matrix are present.
- Keep temporary KerasTuner directories outside the repository or remove them
  after execution.
- Preserve populated outputs in the committed notebook.


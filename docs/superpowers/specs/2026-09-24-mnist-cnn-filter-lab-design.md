# MNIST CNN Filter and Complexity Lab Design

## Goal

Create a self-contained Python notebook at
`applied_machine_oil/lab2/lab2_cnn_filters.ipynb` that teaches convolutional
image classification through MNIST. The notebook must cover layer-by-layer
image/tensor structure, multiple filters, custom and built-in operations,
model-complexity analysis, and common beginner mistakes in CNN projects.

The notebook will use the existing parent `uv` environment at
`machine_learning/.venv` via the symlink visible from the repository root. It
must not create a virtual environment inside `lab2`.

## Notebook flow

1. **Setup and reproducibility**
   - Import TensorFlow/Keras, NumPy, Matplotlib, and sklearn metrics already
     available or installable through the parent environment.
   - Seed Python, NumPy, and TensorFlow.
   - State the expected execution environment and verify key package versions.

2. **Dataset inspection and safe preprocessing**
   - Load MNIST and display representative images and labels.
   - Report the original image structure `(N, 28, 28)`.
   - Split the original training partition into train and validation sets before
     model selection; reserve the official test partition for final evaluation.
   - Convert pixels to `float32`, scale them to `[0, 1]`, and add the channel
     dimension to obtain `(N, 28, 28, 1)`.
   - Assert split sizes, label ranges, and tensor rank.

3. **Fixed-filter exploration**
   - Provide a reusable custom helper for applying a 2-D kernel to one image.
   - Explore horizontal Sobel, vertical Sobel, Laplacian, sharpen, box blur,
     Gaussian-like blur, and emboss kernels.
   - Visualize the source image and filter responses in a compact grid.
   - Explain in one line what each filter emphasizes and why it is useful.

4. **Built-in and learned filters**
   - Demonstrate Keras `Conv2D` and `MaxPooling2D` operations.
   - Compare fixed custom filter responses with built-in convolution output
     without implying that the two are interchangeable.
   - Train the CNN, then visualize kernels from its first convolution layer and
     several resulting feature maps.

5. **CNN shape trace**
   - Build a readable CNN with multiple convolutional filters, pooling,
     dropout, flattening, and dense classification.
   - After every layer, show the output structure and a one-line reason for the
     spatial/channel change.
   - Include an explicit example such as
     `(28, 28, 1) -> (28, 28, 8) -> (14, 14, 8)`.

6. **Complexity analysis**
   - Calculate and tabulate trainable parameters for convolution and dense
     layers.
   - Calculate activation sizes and approximate convolution multiply-accumulate
     counts.
   - Use these formulas:
     - convolution parameters =
       `kernel_height * kernel_width * input_channels * output_channels +
       output_channels`
     - dense parameters = `inputs * outputs + outputs`
     - convolution MACs =
       `output_height * output_width * output_channels * kernel_height *
       kernel_width * input_channels`
   - Explain that channel count increases representational capacity while
     pooling reduces later spatial compute.

7. **Training and evaluation**
   - Train with validation monitoring and early stopping that restores the best
     weights.
   - Plot training and validation loss/accuracy.
   - Evaluate once on the untouched test set.
   - Report accuracy, macro/weighted precision/recall/F1, and a 10-class
     confusion matrix.

8. **Beginner-error audit**
   - Include a concise table mapping each common mistake to the notebook fix:
     data leakage, train/test contamination, missing validation monitoring,
     overfitting, uncontrolled randomness, shape/channel mistakes, relying on
     accuracy alone, and confusing display normalization with model input.
   - Link the TensorFlow/Keras and scikit-learn/Google guidance used for these
     choices.

## Design constraints

- Use Python and a Jupyter notebook; do not create a local environment under
  `lab2`.
- Keep explanations to short one-line “what was done / why” notes rather than
  long prose.
- Prefer existing packages in the parent environment; do not add unrelated
  dependencies.
- Keep test data out of model selection and preprocessing decisions.
- Surface incorrect shapes, labels, or split sizes with assertions rather than
  silently correcting them.

## Validation

Execute the notebook with the parent environment, then validate it with
`nbformat`/`nbclient`. The completed notebook must have populated outputs and
no error cells. Confirm that the complexity table, filter visualizations,
shape trace, metrics, and beginner-error audit are all present.


# Text-to-Latent Representation Notebook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the incomplete reference notebook with a reproducible customer-review pipeline that demonstrates preprocessing, vocabulary construction, token IDs, padding, trained word embeddings, flattened sentence vectors, and PCA visualization.

**Architecture:** Keep the deliverable as one self-contained Jupyter notebook at the existing lab3 path. Organize it into short Markdown explanations and executable code cells, with deterministic preprocessing and seeded Keras training; use the full embedding sequence for inspection/flattening and a masked global-average path for the small sentiment classifier. Generate all tables, assertions, metrics, and plots from fresh notebook execution.

**Tech Stack:** Python, Jupyter/nbformat, NumPy, pandas, Matplotlib, TensorFlow/Keras, scikit-learn PCA, parent `venv` at `/Users/dan/projects/pythonVishal/machine_learning/venv`.

## Global Constraints

- Use exactly the five supplied customer reviews and explicit labels `[1, 1, 1, 0, 0]`; explain that this is a teaching example, not a meaningful benchmark.
- Use deterministic preprocessing and tokenizer behavior so token IDs are reproducible.
- Reserve token ID `0` for padding and set `mask_zero=True` on the embedding layer.
- Use embedding dimension `8`, derive `max_length` from the data, and post-pad with zeros.
- Use global average pooling for the classifier and separately flatten the full `(sequence_length, embedding_dimension)` embedding sequence.
- Exclude the padding row from PCA and report the two explained-variance ratios.
- Avoid external downloads and hidden files; the notebook must run offline in the parent `venv`.
- Preserve unrelated dirty worktree changes in `applied_machine_oil/lab2`.
- Include the Copilot co-author trailer in any commit created during implementation.

---

## File map

- **Modify:** `applied_machine_oil/lab3/Text_to_Latent_Representation_.ipynb` — the sole executable teaching artifact; replace stale reference data, unfinished training flow, and stale outputs with the complete pipeline.
- **Create:** `docs/superpowers/plans/2026-09-30-text-latent-representation.md` — this implementation plan.
- **Do not modify:** dependencies, datasets, or neighboring lab notebooks; required NumPy, TensorFlow, pandas, Matplotlib, and scikit-learn packages are already available in the parent environment.

### Task 1: Rebuild the notebook introduction and deterministic setup

**Files:**
- Modify: `applied_machine_oil/lab3/Text_to_Latent_Representation_.ipynb` — replace the reference cells with the new top-level narrative and setup.

**Interfaces:**
- Consumes: no earlier notebook variables.
- Produces: `SEED`, `texts`, `labels`, `tokenizer`-ready cleaned reviews, and imported `np`, `pd`, `plt`, `tf`, `PCA`, `Tokenizer`, and `pad_sequences`.

- [ ] **Step 1: Create the notebook cell structure**

Create fresh Markdown and code cells in this order:

```python
# imports, reproducibility, and runtime information
import os
import random
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.decomposition import PCA
from tensorflow.keras import Model, layers
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
tf.keras.utils.set_random_seed(SEED)
```

Add a Markdown heading and a top “Additional learning included in this notebook” checklist covering: transparent normalization, deterministic IDs, padding mechanics, masked embeddings, tensor-shape tracing, learned-vs-initial embedding distinction, token lookup inspection, flattening versus pooling, PCA, assertions, and limitations. Follow the neighboring lab pattern by giving each major section a one-line **What** and **Why** explanation.

- [ ] **Step 2: Add the exact assignment dataset and labels**

Use:

```python
texts = [
    "The product quality is excellent",
    "I really love this product",
    "The quality of this item is amazing",
    "The product performance is poor",
    "I dislike this product completely",
]
labels = np.array([1, 1, 1, 0, 0], dtype=np.float32)
```

Create a `review_df` with `review_id`, `text`, and `sentiment`, where sentiment is mapped from labels to `"positive"`/`"negative"`, and display it.

- [ ] **Step 3: Verify the setup cell**

Run from the lab directory:

```bash
/Users/dan/projects/pythonVishal/machine_learning/venv/bin/jupyter nbconvert \
  --to notebook --execute --inplace --ExecutePreprocessor.timeout=180 \
  Text_to_Latent_Representation_.ipynb
```

Expected: the notebook executes its setup and dataset cells without import or seed errors. Do not treat this as the final validation until the later cells exist.

### Task 2: Implement preprocessing, vocabulary, token sequences, and padding

**Files:**
- Modify: `applied_machine_oil/lab3/Text_to_Latent_Representation_.ipynb` — add preprocessing and tokenization sections after the dataset.

**Interfaces:**
- Consumes: `texts`, `review_df`.
- Produces: `cleaned_texts`, `tokenizer`, `word_index`, `vocab_table`, `sequences`, `max_length`, `padded_sequences`.

- [ ] **Step 1: Add a transparent normalization helper**

Use a narrow helper that lowercases, replaces non-alphanumeric characters with spaces, and collapses whitespace:

```python
def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

cleaned_texts = [normalize_text(text) for text in texts]
preprocessing_df = pd.DataFrame(
    {"original_text": texts, "cleaned_text": cleaned_texts}
)
preprocessing_df["token_count"] = preprocessing_df["cleaned_text"].str.split().str.len()
display(preprocessing_df)
```

- [ ] **Step 2: Fit the tokenizer and display the vocabulary**

Use `Tokenizer(num_words=None, oov_token=None, filters="")`, fit it on `cleaned_texts`, and build a table that explicitly includes padding:

```python
tokenizer = Tokenizer(num_words=None, oov_token=None, filters="")
tokenizer.fit_on_texts(cleaned_texts)
word_index = tokenizer.word_index

vocab_table = pd.DataFrame(
    [{"token": "<PAD>", "token_id": 0}]
    + [{"token": word, "token_id": token_id}
       for word, token_id in sorted(word_index.items(), key=lambda item: item[1])]
)
display(vocab_table)
assert vocab_table["token_id"].is_unique
assert set(vocab_table["token_id"]) == set(range(len(word_index) + 1))
```

Explain that IDs are lookup keys, not ordinal measurements.

- [ ] **Step 3: Convert reviews to variable-length sequences**

Create and display a table with each review and its sequence:

```python
sequences = tokenizer.texts_to_sequences(cleaned_texts)
sequence_df = pd.DataFrame(
    {"review_id": np.arange(1, len(texts) + 1), "cleaned_text": cleaned_texts,
     "token_sequence": sequences}
)
display(sequence_df)
assert all(len(sequence) == len(text.split()) for sequence, text in zip(sequences, cleaned_texts))
```

- [ ] **Step 4: Derive and apply post-padding**

Use the longest observed sequence rather than a magic number:

```python
max_length = max(map(len, sequences))
padded_sequences = pad_sequences(
    sequences, maxlen=max_length, padding="post", truncating="post", value=0
)
display(pd.DataFrame(padded_sequences, index=review_df["review_id"]))
assert padded_sequences.shape == (len(texts), max_length)
assert np.all(padded_sequences[:, :max_length] >= 0)
```

Add a short explanation of post-padding and mark which trailing zero positions are padding.

- [ ] **Step 5: Commit the tokenization milestone**

Run the notebook through the padding section and inspect the displayed vocabulary and sequence tables. Commit only the notebook changes for this milestone:

```bash
git add applied_machine_oil/lab3/Text_to_Latent_Representation_.ipynb
git -c commit.gpgsign=false commit -m "feat: add review tokenization and padding pipeline" \
  -m "Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

### Task 3: Train and extract the word embeddings

**Files:**
- Modify: `applied_machine_oil/lab3/Text_to_Latent_Representation_.ipynb` — add the embedding model, training, and extraction cells.

**Interfaces:**
- Consumes: `padded_sequences`, `labels`, `word_index`, `max_length`.
- Produces: `embedding_dimension`, `embedding_model`, `embedding_layer`, `embedding_tensor`, `embedding_matrix`, `sentence_embeddings`.

- [ ] **Step 1: Define the masked embedding classifier**

Use a model whose embedding matrix is learned through the sentiment objective:

```python
vocab_size = len(word_index) + 1
embedding_dimension = 8

inputs = layers.Input(shape=(max_length,), dtype="int32", name="token_ids")
embedding_layer = layers.Embedding(
    input_dim=vocab_size,
    output_dim=embedding_dimension,
    mask_zero=True,
    name="word_embedding",
)
embedded = embedding_layer(inputs)
pooled = layers.GlobalAveragePooling1D(name="masked_average")(embedded)
outputs = layers.Dense(1, activation="sigmoid", name="sentiment_output")(pooled)
embedding_model = Model(inputs, outputs, name="review_embedding_classifier")
embedding_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.03),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
embedding_model.summary()
```

Explain that `mask_zero=True` prevents padded IDs from contributing to the average, while the separate embedding output retains the padded positions for the assignment’s matrix/flattening view.

- [ ] **Step 2: Train with bounded, deterministic settings**

Train without validation because there are only five examples, and state that training accuracy is descriptive only:

```python
history = embedding_model.fit(
    padded_sequences,
    labels,
    epochs=150,
    batch_size=len(texts),
    shuffle=False,
    verbose=0,
)
training_metrics = embedding_model.evaluate(
    padded_sequences, labels, verbose=0, return_dict=True
)
display(pd.DataFrame(history.history).tail())
display(pd.DataFrame([training_metrics]))
```

Plot loss and accuracy from `history.history`, with a note that no generalization claim is justified.

- [ ] **Step 3: Extract the embedding tensor and matrix**

Create extraction models and assert exact shapes:

```python
embedding_extractor = Model(inputs, embedded, name="embedding_extractor")
embedding_tensor = embedding_extractor.predict(padded_sequences, verbose=0)
embedding_matrix = embedding_layer.get_weights()[0]
sentence_embeddings = embedding_tensor.reshape(
    len(texts), max_length, embedding_dimension
)

assert embedding_tensor.shape == (len(texts), max_length, embedding_dimension)
assert embedding_matrix.shape == (vocab_size, embedding_dimension)
assert np.allclose(embedding_matrix[0], 0.0)
```

Display the tensor shape and a clearly labeled per-review embedding matrix. Include a lookup proof for one non-padding token:

```python
first_non_padding_id = int(padded_sequences[0, 0])
assert np.allclose(
    embedding_tensor[0, 0], embedding_matrix[first_non_padding_id]
)
```

- [ ] **Step 4: Display the word-to-vector lookup table**

Build a formatted table from `word_index` and `embedding_matrix`, excluding `<PAD>` from the word table but display the padding row separately:

```python
word_vectors = pd.DataFrame(
    embedding_matrix[1:],
    index=[word for word, _ in sorted(word_index.items(), key=lambda item: item[1])],
    columns=[f"dim_{i + 1}" for i in range(embedding_dimension)],
)
display(word_vectors.round(4))
```

Add prose explaining that every word has one eight-dimensional learned vector and that vector coordinates have no standalone human meaning.

### Task 4: Flatten sentence embedding matrices and add PCA

**Files:**
- Modify: `applied_machine_oil/lab3/Text_to_Latent_Representation_.ipynb` — add flattening, PCA, interpretation, and conclusion sections.

**Interfaces:**
- Consumes: `sentence_embeddings`, `embedding_matrix`, `word_index`, `embedding_dimension`, `max_length`.
- Produces: `flattened_vectors`, `flattened_df`, `pca_coordinates`, and the final notebook conclusions.

- [ ] **Step 1: Flatten each sentence embedding matrix**

Use row-major flattening so token order and position are preserved:

```python
flattened_vectors = sentence_embeddings.reshape(len(texts), -1)
flattened_columns = [
    f"position_{position + 1}_dim_{dimension + 1}"
    for position in range(max_length)
    for dimension in range(embedding_dimension)
]
flattened_df = pd.DataFrame(flattened_vectors, columns=flattened_columns)
flattened_df.insert(0, "review_id", np.arange(1, len(texts) + 1))
display(flattened_df.round(4))
assert flattened_vectors.shape == (len(texts), max_length * embedding_dimension)
```

Add a compact sentence explaining that flattening produces one vector per review but differs from the masked average used by the classifier.

- [ ] **Step 2: Run PCA on vocabulary words only**

Exclude the padding row and create labeled coordinates:

```python
words = [word for word, _ in sorted(word_index.items(), key=lambda item: item[1])]
word_vectors_for_pca = embedding_matrix[1:]
pca = PCA(n_components=2, random_state=SEED)
pca_coordinates = pca.fit_transform(word_vectors_for_pca)
pca_df = pd.DataFrame(
    {"word": words, "PC1": pca_coordinates[:, 0], "PC2": pca_coordinates[:, 1]}
)
display(pca_df.round(4))
print("Explained variance ratio:", np.round(pca.explained_variance_ratio_, 4))
assert pca_coordinates.shape == (len(words), 2)
```

- [ ] **Step 3: Plot and interpret the PCA projection**

Create a readable scatter plot with one annotation per word, axes labeled with explained variance, a zero reference grid, and a title explaining that points are learned word embeddings. Add prose stating that nearby words have similar coordinates in this learned space, but this tiny dataset cannot establish reliable semantic relationships.

- [ ] **Step 4: Add the final audit/conclusion section**

Add a Markdown checklist mapping each assignment requirement to its notebook output:

```text
raw review → normalized text → vocabulary/token IDs → padded IDs
→ learned embedding tensor → flattened sentence vectors → PCA word plot
```

Include limitations: five examples, labels supplied for demonstration, no held-out evaluation, random initialization despite fixed seed, and PCA’s variance-preserving projection rather than a semantic truth measure.

- [ ] **Step 5: Commit the completed notebook**

After running the full notebook once and checking all required outputs:

```bash
git add applied_machine_oil/lab3/Text_to_Latent_Representation_.ipynb
git -c commit.gpgsign=false commit -m "feat: complete text latent representation lab" \
  -m "Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

### Task 5: Validate the final artifact

**Files:**
- Test: `applied_machine_oil/lab3/Text_to_Latent_Representation_.ipynb` via Jupyter execution and notebook inspection.

**Interfaces:**
- Consumes: the completed notebook and parent `venv`.
- Produces: a cleanly executed notebook with all required outputs persisted.

- [ ] **Step 1: Execute in the requested parent environment**

Run:

```bash
cd /Users/dan/projects/pythonVishal/machine_learning/applied_machine_oil/lab3
/Users/dan/projects/pythonVishal/machine_learning/venv/bin/jupyter nbconvert \
  --to notebook --execute --inplace --ExecutePreprocessor.timeout=180 \
  Text_to_Latent_Representation_.ipynb
```

Expected: exit code `0`, no cell errors, and the output notebook contains vocabulary, padded sequences, embedding tables, flattened vectors, and the PCA figure.

- [ ] **Step 2: Check notebook structure and outputs**

Run:

```bash
/Users/dan/projects/pythonVishal/machine_learning/venv/bin/python - <<'PY'
import nbformat
from pathlib import Path

path = Path("Text_to_Latent_Representation_.ipynb")
nb = nbformat.read(path, as_version=4)
assert any(cell.cell_type == "markdown" and "Additional learning included" in cell.source
           for cell in nb.cells)
assert any(cell.cell_type == "markdown" and "PCA" in cell.source
           for cell in nb.cells)
assert all("error" not in [output.output_type for output in cell.get("outputs", [])]
           for cell in nb.cells if cell.cell_type == "code")
print(f"validated {len(nb.cells)} cells with no error outputs")
PY
```

- [ ] **Step 3: Run a final diff check**

Run:

```bash
cd /Users/dan/projects/pythonVishal/machine_learning
git --no-pager diff --check HEAD~1..HEAD
git --no-pager status --short
```

Expected: no whitespace errors. Report any pre-existing lab2 modifications without altering them.


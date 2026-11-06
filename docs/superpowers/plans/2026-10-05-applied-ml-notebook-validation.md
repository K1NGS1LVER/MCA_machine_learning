# Applied ML Notebook Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Validate the required CNN, LeNet-5, and Q-Learning notebooks in the existing uv environment and leave a reproducible offline run guide.

**Architecture:** Use the existing notebooks as the source of truth. A small validation script will discover imports and execute the five scoped notebooks with `MPLBACKEND=Agg`, recording pass/fail and output counts; notebook edits remain surgical and are made only for observed compatibility failures. The run guide will describe the fixed environment path, commands, local dataset/cache requirements, and error-correction practice.

**Tech Stack:** Python 3.14, uv virtual environment, Jupyter nbconvert/nbclient, TensorFlow/Keras, Keras Tuner, Gymnasium, NumPy, Matplotlib.

## Global Constraints

- Use `/Users/dan/projects/pythonVishal/machine_learning/.venv/bin/python` and its Jupyter executable.
- Validate only `lab2_cnn_filters.ipynb`, `cnn_full_visualization.ipynb`, `lab2_cnn_unified.ipynb`, `part_b_lenet5_mnist.ipynb`, and `samples/Q_Learning.ipynb`.
- Apply the smallest compatibility fix that preserves the notebook’s intended algorithm and teaching code.
- Do not claim offline readiness when a required dataset or asset is not locally cached.
- Do not add generated model caches, temporary files, or large execution artifacts to version control.

---

### Task 1: Add a deterministic notebook validation runner

**Files:**
- Create: `applied_machine_oil/validate_practice_notebooks.py`
- Test: command-line execution of the runner against the scoped notebooks

**Interfaces:**
- Produces a command-line report with one result per notebook, cell execution counts, and visualization-output counts.
- Accepts no required arguments; paths are resolved relative to the script’s repository location.

- [ ] **Step 1: Write the runner with explicit notebook scope**

Use `pathlib.Path` to resolve the repository root from `__file__`. Store the five notebook paths in a tuple. For each notebook, run `jupyter nbconvert --to notebook --execute --output <temporary-output> --ExecutePreprocessor.timeout=<timeout>` through `subprocess.run`, with `MPLBACKEND=Agg` in the environment and the uv Jupyter executable selected explicitly. Capture stdout/stderr and return a nonzero exit code if any notebook fails.

- [ ] **Step 2: Count successful cells and visual outputs**

Load each executed temporary notebook with `nbformat`. Count code cells and code cells with `outputs`. Count display outputs whose MIME bundle contains `image/png`, `image/svg+xml`, or `text/plain` from `print`/table output. Print a compact line such as:

```text
PASS applied_machine_oil/lab2/lab2_cnn_filters.ipynb cells=12 outputs=12 visual=4
```

If execution fails, print the notebook path and captured traceback, then continue to report the remaining notebooks so one failure does not hide independent results.

- [ ] **Step 3: Run the runner’s syntax/import check**

Run:

```bash
/Users/dan/projects/pythonVishal/machine_learning/.venv/bin/python -m py_compile applied_machine_oil/validate_practice_notebooks.py
```

Expected: exit code 0 and no generated `.pyc` committed.

- [ ] **Step 4: Commit the runner**

```bash
git add applied_machine_oil/validate_practice_notebooks.py
git commit -m "chore: add practice notebook validation runner"
```

### Task 2: Verify the uv environment and local offline assets

**Files:**
- Modify: `applied_machine_oil/validate_practice_notebooks.py` only if the check exposes a runner defect
- Test: uv environment import smoke check

**Interfaces:**
- The environment check reports Python version, Jupyter kernel executable, and imports for `numpy`, `matplotlib`, `tensorflow`, `keras_tuner`, and `gymnasium`.

- [ ] **Step 1: Run the import smoke check**

Run:

```bash
/Users/dan/projects/pythonVishal/machine_learning/.venv/bin/python -c "import sys, numpy, matplotlib, tensorflow, keras_tuner, gymnasium; print(sys.executable); print('imports ok')"
```

Expected: the executable is under `.venv` and the final line is `imports ok`.

- [ ] **Step 2: Check assets without network access**

Inspect the notebooks for dataset and image paths, then verify those paths or the relevant framework cache directories exist locally. Do not start a download. Record any missing asset as a blocker in the run guide.

- [ ] **Step 3: Commit only if a runner correction was required**

If no source correction was needed, leave the runner commit unchanged. If a correction was required, run the syntax check again and commit it with a focused message.

### Task 3: Execute the CNN and Q-Learning notebooks

**Files:**
- Modify: `applied_machine_oil/lab2/lab2_cnn_filters.ipynb` only if its failing cell identifies a compatibility defect
- Modify: `applied_machine_oil/lab2/cnn_full_visualization.ipynb` only if its failing cell identifies a compatibility defect
- Modify: `applied_machine_oil/lab2/lab2_cnn_unified.ipynb` only if its failing cell identifies a compatibility defect
- Modify: `applied_machine_oil/samples/Q_Learning.ipynb` only if its failing cell identifies a compatibility defect

**Interfaces:**
- Each notebook must execute from a fresh kernel with no error.
- Successful outputs remain embedded only when they are useful for the practical-test setup.

- [ ] **Step 1: Run the three CNN notebooks**

Run the validation runner with the LeNet notebook temporarily excluded only if it would prevent independent CNN results; otherwise run the complete scoped set. For each CNN failure, inspect the exact failing cell and traceback before editing.

- [ ] **Step 2: Apply only evidence-based fixes**

Preserve imports, model structure, training objectives, and visualization intent. Fix only version/API incompatibilities such as renamed arguments or an unbuilt Keras graph. Do not replace the notebook with a rewritten script.

- [ ] **Step 3: Run Q-Learning and inspect its plot**

Confirm the FrozenLake training completes, the Q-table is printed, and the reward-history visualization is present. A missing local Gymnasium environment is an installation failure, not a reason to silently skip the notebook.

- [ ] **Step 4: Re-run changed notebooks from clean kernels**

Run the runner again after each notebook edit. Require a successful process exit and at least one expected output for Q-Learning; require image/display output for visualization notebooks.

- [ ] **Step 5: Commit successful CNN/Q-Learning fixes**

```bash
git add applied_machine_oil/lab2/*.ipynb applied_machine_oil/samples/Q_Learning.ipynb
git commit -m "fix: make practice notebooks execute in current environment"
```

Only stage files that were actually changed for this task.

### Task 4: Finish and verify LeNet-5 execution

**Files:**
- Modify: `applied_machine_oil/lab2/part_b_lenet5_mnist.ipynb`
- Test: full LeNet-5 notebook execution through the validation runner

**Interfaces:**
- The visualization extractor must use a tensor that exists after model construction/training.
- The notebook must complete without the previous `sequential_1 ... has never been called` error.

- [ ] **Step 1: Preserve the known compatibility correction**

Keep the existing `tuned_model.inputs[0]` correction in the extractor construction. Keep `TF_DISABLE_PREFETCH=1` only if the notebook requires it for this environment and the setting is established before TensorFlow is imported.

- [ ] **Step 2: Run LeNet-5 with an extended timeout**

Run the notebook through the uv Jupyter executable with `MPLBACKEND=Agg` and an execution timeout long enough for the Keras Tuner search and final training. Monitor for a real traceback rather than treating TensorFlow starvation warnings as an execution error.

- [ ] **Step 3: Verify model and visualization outputs**

Confirm baseline/tuned training output, accuracy/loss plots, and the traced activation visualization are present. If the notebook fails, patch only the failing API use and rerun from a clean kernel.

- [ ] **Step 4: Commit the LeNet fix**

```bash
git add applied_machine_oil/lab2/part_b_lenet5_mnist.ipynb
git commit -m "fix: support Keras model tracing in LeNet notebook"
```

### Task 5: Document offline setup and final verification

**Files:**
- Create: `applied_machine_oil/PRACTICE_NOTEBOOKS.md`
- Test: final complete validation-runner execution

**Interfaces:**
- The guide gives copy-paste commands using the absolute uv environment path.
- The guide lists the five validated notebooks and explains how to identify and correct a failing cell without rewriting the program.

- [ ] **Step 1: Write the setup guide**

Document the environment path, Python/Jupyter version checks, import smoke check, validation command, expected success indicators, local dataset/cache assumptions, and the known LeNet Keras input correction. State clearly that missing assets must be downloaded before the internet-free test.

- [ ] **Step 2: Run the complete final validation**

Run:

```bash
cd /Users/dan/projects/pythonVishal/machine_learning
MPLBACKEND=Agg /Users/dan/projects/pythonVishal/machine_learning/.venv/bin/python applied_machine_oil/validate_practice_notebooks.py
```

Expected: five `PASS` lines, zero notebook failures, and nonzero output/visual counts appropriate to each notebook.

- [ ] **Step 3: Check repository hygiene**

Run:

```bash
git status --short
git diff --check
```

Remove only temporary files created by the validation runner, and confirm no model caches or execution scratch files are staged.

- [ ] **Step 4: Commit the guide**

```bash
git add applied_machine_oil/PRACTICE_NOTEBOOKS.md
git commit -m "docs: add offline practice notebook guide"
```

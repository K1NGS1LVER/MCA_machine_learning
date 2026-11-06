# Applied ML Notebook Validation Design

## Goal

Make the existing uv-managed environment ready for the practical test by
validating the required CNN, LeNet-5, and Q-Learning notebooks offline. The
notebooks remain teaching material: fixes are limited to execution blockers
revealed by the validation run, not rewrites or algorithm changes.

## Scope

Validate these notebooks with
`/Users/dan/projects/pythonVishal/machine_learning/.venv/bin/jupyter`:

- `applied_machine_oil/lab2/lab2_cnn_filters.ipynb`
- `applied_machine_oil/lab2/cnn_full_visualization.ipynb`
- `applied_machine_oil/lab2/lab2_cnn_unified.ipynb`
- `applied_machine_oil/lab2/part_b_lenet5_mnist.ipynb`
- `applied_machine_oil/samples/Q_Learning.ipynb`

The existing `samples/Text_to_Latent_Representation_.ipynb` is outside this
test scope because the requested practical set is CNN, LeNet-5, and
Q-Learning.

## Execution and fixes

1. Verify the environment imports the libraries used by the scoped notebooks
   and that the Jupyter kernel points at the uv environment.
2. Execute each notebook with a non-interactive plotting backend and a
   sufficiently long per-notebook timeout.
3. If a notebook fails, identify the exact failing cell and apply the
   smallest compatible fix while preserving its intended code and outputs.
   The known LeNet-5 Keras compatibility fix uses the model's built input
   tensor (`tuned_model.inputs[0]`) instead of the fragile `.input` property.
4. Re-run every changed notebook from a clean kernel and retain successful
   outputs in the notebook only when they are useful for the practical-test
   setup.

## Success criteria

- Every scoped notebook completes without an execution error.
- Every code cell has a successful execution result; expected plots or image
  displays are present in the executed notebook.
- The required imports work in the uv environment.
- Execution does not depend on a new internet download after setup. If a
  dataset is missing from the local cache, report that explicitly rather than
  claiming offline readiness.
- A short run guide records the environment path, commands, cache assumptions,
  and the known correction workflow for practicing error diagnosis.

## Deliverables

- Surgical notebook changes required for compatibility.
- A concise offline setup and execution guide under
  `applied_machine_oil/`.
- No generated model caches, temporary files, or large execution artifacts
  added to version control.

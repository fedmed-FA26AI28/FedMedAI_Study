# Repository Guidelines

## Project Structure & Module Organization

FedMedAI compares centralized learning, FedAvg, and FedProx on MedMNIST using PyTorch and Flower. ResNet-18 is the default; BloodCNN is an optional ablation.

- `src/fedmedai/`: models, datasets, partitions, training, evaluation, and experiment orchestration. `monitoring/` contains classification metrics and resource sampling.
- `scripts/`: CLI entry points for unified, centralized, IID, and non-IID experiments.
- `configs/`: default configuration (`config.yaml`) and small sequential runs (`smoke.yaml`).
- `tests/test_experiments.py`: offline regression tests using synthetic data and temporary directories.
- `results/`, `checkpoints/`, and `runs/`: generated reports, model weights, and TensorBoard events. Dataset files use the MedMNIST cache or configured root.

## Build, Test, and Development Commands

Run commands from the repository root with Python 3.10+ in a virtual environment.

- `python -m pip install -e .`: install the package and dependencies for development; packaging uses Hatchling.
- `python -m unittest discover -s tests -v`: run the offline test suite.
- `python scripts/run_experiment.py --config configs/smoke.yaml`: run a small sequential federated experiment.
- `python scripts/run_centralized.py --config configs/smoke.yaml`: check the centralized pipeline.
- `python scripts/run_experiment.py --config configs/smoke.yaml --backend flower --rounds 1`: check Flower/Ray integration.
- `tensorboard --logdir runs`: inspect experiment curves.

Smoke runs require cached BloodMNIST data because downloads are disabled. Prepare it with `python -c "import medmnist; medmnist.BloodMNIST(split='train', download=True)"`.

## Coding Style & Naming Conventions

Use four-space Python indentation, `snake_case` functions and variables, and `PascalCase` classes. Follow existing type hints and docstrings. Keep CLI parsing in `scripts/` and reusable logic in `src/fedmedai/`. Use two-space YAML indentation. No formatter or linter is configured.

## Testing Guidelines

Use `unittest.TestCase`, `test_*.py` files, and descriptive `test_*` methods. Add regression coverage for changed behavior, especially partition reproducibility, metrics, checkpoint selection, and artifact schemas. Keep unit tests offline. No coverage threshold is configured; test Flower/Ray separately when changing that backend.

## Commit & Pull Request Guidelines

History contains only `Initial commit`; no message convention is established. Use concise, imperative subjects. PRs should explain the change, link relevant issues, list validation commands and outcomes, and identify configuration or artifact-schema changes. Avoid adding generated datasets, checkpoints, run outputs, or bytecode.

## Experiment Integrity

Current research uses the complete BloodMNIST splits, ResNet-18, 8 rounds,
3 clients, one local epoch, seed 2026, and Dirichlet alpha=[0.1, 0.3, 1.0] only.
Keep the uncompressed FedAvg control fixed; compare genuine FedProx mu=0.01
and FedProx+FP16. `research/non_iid_baseline.lock.json` is the design authority.
The completed suite is `results/non_iid_study/suite_20260926T103613Z` (9 cells
plus a same-seed FedAvg replay). Read STATUS.md before launching any new work.
Legacy IID/BloodCNN result sets were removed at the user's request; do not
recreate them as part of this study. IID code remains for general pipeline and
offline synthetic regression checks, not the current research matrix.
Both report entry points generate current non-IID and communication artifacts.
Preserve executed source hashes and immutable snapshots when refreshing docs.

Select checkpoints using validation data; evaluate test data only at run completion. Preserve seeds, sample IDs, and partition hashes. Represent unavailable measurements as `null`, and distinguish smoke checks from research benchmarks.

## Definition of Done

A task is NOT complete when code has merely been written.

A task is complete only when:

1. Required deliverables exist.
2. Acceptance criteria are satisfied.
3. Relevant tests pass.
4. No known critical issue remains.
5. Reviewer has approved the task when review is required.
6. STATUS.md reflects the current state.
7. Important decisions are documented.

# Agent Execution Loop

For every task:

1. Read task definition.
2. Read dependencies.
3. Inspect existing implementation.
4. Create a short execution plan.
5. Perform the work.
6. Validate the result.
7. If validation fails:
   - diagnose
   - fix
   - validate again
8. Do not bypass failed validation.
9. Report evidence.
10. Update task status only when acceptance criteria pass.

Never claim success without verification.

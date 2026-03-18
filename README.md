# Delta Framework (Delta-Only Model Training)

A PyTorch-based library for class-incremental learning that benchmarks a
delta-only update (new data + replay + Weight Alignment) against a full
retrain baseline, with optional distribution-shift monitoring and a Streamlit
dashboard.

Note on "equivalence": this repo reports (1) an empirical equivalence gap
versus a full retrain baseline and (2) a simple PAC-style bound on the
accuracy gap. It does not prove that the learned weights are identical to full
retraining.

## What's included

- Incremental trainer (delta-only) with replay, KD, and Weight Alignment
- Full-retrain baseline per task for side-by-side comparison
- Shift detection via KL divergence on penultimate embeddings
- Shift-gated deployment policy that can recommend full retraining
- Equivalence summary with gap, calibration, and worst-class robustness metrics
- Optional ablation suite for naive new-data, replay-only, and replay+KD variants
- Optional Streamlit UI that reads `results.json`

## Project structure

```text
delta_framework/
  api.py
  core/
    trainer.py
    benchmarker.py
    equivalence.py
    shift_detector.py
    bounds.py
  experiments/
    run_experiment.py
  resnet.py
ui/
  app.py
examples/
  template.py
  utils.py
results.json
pyproject.toml
requirements.txt
```

## Install

Supported Python versions: `3.10` and `3.11`.

This project is not currently set up for Python `3.12+`. That limitation comes
mainly from the PyTorch ecosystem compatibility matrix, not from the
pure-Python parts of this repo.

Recommended:

```bash
pip install -e .
```

If you prefer Poetry:

```bash
poetry install
```

Tip: for GPU/CUDA builds, install `torch` and `torchvision` using the official
PyTorch instructions for your system first, then install this project.

## Run the benchmark

After install:

```bash
delta-benchmark --dataset CIFAR-100 --num-tasks 5 --classes-per-task 20 --epochs 3 --batch-size 128 --prefer-cuda --results-path results.json
```

### Run from a JSON config

Dump a config you can edit:

```bash
delta-benchmark --dump-config config.json
```

Run using a config:

```bash
delta-benchmark --config config.json --results-path results.json
```

Show richer runtime logs while the benchmark runs:

```bash
delta-benchmark --config config.json --log-level DEBUG --results-path results.json
```

Run the optional ablation suite and deployment policy thresholds:

```bash
delta-benchmark --dataset CIFAR-100 --run-ablations --equivalence-threshold 0.005 --policy-max-bound-epsilon 0.01 --results-path results.json
```

Or without the installed script:

```bash
python -m delta_framework.experiments.run_experiment --dataset CIFAR-100 --num-tasks 5 --classes-per-task 20 --epochs 3 --batch-size 128 --prefer-cuda --results-path results.json
```

## Dashboard

Install UI extras:

```bash
pip install -e ".[ui]"
```

Then run:

```bash
streamlit run ui/app.py
```

## Dev

```bash
python -m ruff check .
python -m pytest
```

## Acknowledgments

- Weight Alignment: https://arxiv.org/abs/1512.02325
- continuum: https://github.com/Continvvm/continuum

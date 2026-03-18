# Using Delta Framework

This guide shows how another user can install and use **Delta Framework** without changing the internal code.

## 1. Install

### Full framework

```bash
pip install -e ".[app]"
```

### ML runtime only

```bash
pip install -e ".[ml]"
```

### Optional legacy Streamlit UI

```bash
pip install -e ".[ui]"
```

## 2. Use it as a Python library

The clean public API is in `delta_framework.api`.

### Minimal example

```python
from delta_framework.api import BenchmarkConfig, TrainConfig, run

config = BenchmarkConfig(
    dataset="CIFAR-10",
    num_tasks=2,
    classes_per_task=5,
    prefer_cuda=False,
    train=TrainConfig(
        backbone="resnet32",
        epochs=1,
        batch_size=64,
        num_workers=0,
    ),
)

results = run(config, results_path="results.json")
print(results["final_summary"])
```

### More controlled example

```python
from delta_framework.api import BenchmarkConfig, TrainConfig, run

config = BenchmarkConfig(
    dataset="CIFAR-100",
    data_path="./data",
    num_tasks=5,
    classes_per_task=20,
    prefer_cuda=True,
    shift_threshold=0.3,
    equivalence_threshold=0.005,
    policy_max_bound_epsilon=0.01,
    run_ablations=True,
    train=TrainConfig(
        backbone="resnet32",
        epochs=3,
        batch_size=128,
        num_workers=0,
        old_fraction=0.2,
        memory_size=2000,
        use_replay=True,
        use_kd=True,
        use_weight_align=True,
    ),
)

results = run(config, results_path="results.json")
print(results["timeline"][-1]["deployment"])
```

## 3. Use it from the CLI

### Quick demo command

```bash
python -m delta_framework.experiments.run_experiment \
  --dataset CIFAR-10 \
  --num-tasks 2 \
  --classes-per-task 5 \
  --epochs 1 \
  --batch-size 64 \
  --num-workers 0 \
  --results-path results.json
```

### Larger benchmark command

```bash
python -m delta_framework.experiments.run_experiment \
  --dataset CIFAR-100 \
  --num-tasks 5 \
  --classes-per-task 20 \
  --epochs 3 \
  --batch-size 128 \
  --num-workers 0 \
  --run-ablations \
  --equivalence-threshold 0.005 \
  --policy-max-bound-epsilon 0.01 \
  --results-path results.json
```

### Dump a config file first

```bash
python -m delta_framework.experiments.run_experiment --dump-config config.json
```

Then:

```bash
python -m delta_framework.experiments.run_experiment --config config.json --results-path results.json
```

## 4. Use the web app

Start the server:

```bash
python -m delta_framework.web.server
```

Open:

```text
http://127.0.0.1:8080
```

Pages:
- `/` -> setup page
- `/monitor.html` -> live run page
- `/results.html` -> results page

## 5. Understand the output files

### `results.json`

Contains:
- run status
- partial task info while a task is running
- task-by-task delta metrics
- task-by-task full retrain metrics
- equivalence summaries
- shift scores
- deployment decisions
- final summary

### `experiment.log`

Contains:
- benchmark start info
- task start / finish logs
- failure details if a run crashes

## 6. Recommended presets

### Fast demo

Use this if you want something responsive:

- dataset: `CIFAR-10`
- tasks: `2`
- epochs: `1`
- workers: `0`
- ablations: `off`

### Balanced

- dataset: `CIFAR-100`
- tasks: `5`
- epochs: `2`

### Full benchmark

- dataset: `CIFAR-100`
- tasks: `5`
- epochs: `3`
- ablations: `on`

## 7. What the framework is actually doing

Delta Framework runs:

1. an **incremental update path**
2. a **full retrain baseline**
3. an **equivalence comparison**
4. a **shift detection step**
5. a **deployment policy**

So users are not just training a model - they are evaluating whether an incremental update is safe compared to retraining.

## 8. Troubleshooting

### The first results take time to appear

That is expected.

The first task only becomes visible after:
- delta training finishes
- full retrain finishes

### CPU runs are slow

For quick demos:
- use `CIFAR-10`
- use `2` tasks
- use `1` epoch
- keep `num_workers=0`
- disable ablations

### I only see logs and no charts yet

That usually means the first task is still running and has not yet produced a completed task record.

## 9. Best way to present it

If you are showing Delta Framework to others:

1. start from the setup page
2. run a fast demo preset
3. open the live page
4. then finish on the results page

That sequence makes the project much easier to understand.

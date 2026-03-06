"""Single-process CIL trainer (delta-only + full-retrain baselines).

This wraps the repo’s core CIL + Weight Alignment idea into reusable functions
that can run on Windows without `torchrun`/DDP. The WA scaling rule is kept
identical in spirit and computation to the original tutorial implementation.
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

try:
    from continuum import ClassIncremental
    from continuum import rehearsal
    from continuum.datasets import CIFAR100 as ContinuumCIFAR100
    from continuum.datasets import ImageFolderDataset

    try:
        from continuum.datasets import CIFAR10 as ContinuumCIFAR10  # type: ignore
    except Exception:  # pragma: no cover
        ContinuumCIFAR10 = None  # type: ignore
except Exception as e:  # pragma: no cover
    raise ImportError(
        "Missing dependency `continuum`. Install requirements.txt first."
    ) from e

from resnet import resnet20, resnet32, resnet44, resnet56


# Fixed class order from the original tutorial script (CIFAR-100).
CIFAR100_CLASS_ORDER: List[int] = [
    68,
    56,
    78,
    8,
    23,
    84,
    90,
    65,
    74,
    76,
    40,
    89,
    3,
    92,
    55,
    9,
    26,
    80,
    43,
    38,
    58,
    70,
    77,
    1,
    85,
    19,
    17,
    50,
    28,
    53,
    13,
    81,
    45,
    82,
    6,
    59,
    83,
    16,
    15,
    44,
    91,
    41,
    72,
    60,
    79,
    52,
    20,
    10,
    31,
    54,
    37,
    95,
    14,
    71,
    96,
    98,
    97,
    2,
    64,
    66,
    42,
    22,
    35,
    86,
    24,
    34,
    87,
    21,
    99,
    0,
    88,
    27,
    18,
    94,
    11,
    12,
    47,
    25,
    30,
    46,
    62,
    69,
    36,
    61,
    7,
    63,
    75,
    5,
    32,
    4,
    51,
    48,
    73,
    93,
    39,
    67,
    29,
    49,
    57,
    33,
]


def get_device(prefer_cuda: bool = True) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _cifar_normalize() -> transforms.Normalize:
    # CIFAR-100 mean/std (commonly used; aligns with tutorial’s constants).
    return transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))


def build_transforms(dataset: str, is_train: bool, input_size: int = 32) -> transforms.Compose:
    dataset_l = dataset.lower()
    if dataset_l in {"cifar-100", "cifar100", "cifar-10", "cifar10"}:
        norm = _cifar_normalize()
        if is_train:
            return transforms.Compose(
                [
                    transforms.RandomCrop(input_size, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    norm,
                ]
            )
        return transforms.Compose([transforms.ToTensor(), norm])

    # TinyImageNet / generic ImageFolder: use ImageNet normalization.
    imagenet_norm = transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    if is_train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(input_size, scale=(0.6, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                imagenet_norm,
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(int(input_size * 1.15)),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            imagenet_norm,
        ]
    )


class TinyImageNet(ImageFolderDataset):
    """Local-path-only TinyImageNet wrapper.

    Expected layout is ImageFolder-compatible:
      <data_path>/train/<class>/...
      <data_path>/val/<class>/...
    """

    def __init__(self, data_path: str, train: bool = True) -> None:
        super().__init__(data_path=data_path, train=train, download=False)

    def get_data(self):
        import os

        self.data_path = os.path.join(self.data_path, "train" if self.train else "val")
        return super().get_data()


def build_scenarios(
    dataset: str,
    data_path: str,
    classes_per_task: int,
    seed: int,
    num_tasks: int,
    input_size: int = 32,
    class_order: Optional[Sequence[int]] = None,
) -> Tuple[Any, Any, int, List[int]]:
    """Return (scenario_train, scenario_val, nb_classes, class_order_used)."""
    dataset_l = dataset.lower()
    train_tf = build_transforms(dataset, is_train=True, input_size=input_size)
    val_tf = build_transforms(dataset, is_train=False, input_size=input_size)

    if dataset_l in {"cifar-100", "cifar100"}:
        base_train = ContinuumCIFAR100(data_path, train=True, download=True)
        base_val = ContinuumCIFAR100(data_path, train=False, download=True)
        if class_order is None:
            class_order_used = CIFAR100_CLASS_ORDER
        else:
            class_order_used = list(class_order)
    elif dataset_l in {"cifar-10", "cifar10"}:
        if ContinuumCIFAR10 is None:  # pragma: no cover
            raise RuntimeError(
                "Your installed `continuum` build does not expose CIFAR10. "
                "Use CIFAR-100 or upgrade continuum."
            )
        base_train = ContinuumCIFAR10(data_path, train=True, download=True)  # type: ignore[misc]
        base_val = ContinuumCIFAR10(data_path, train=False, download=True)  # type: ignore[misc]
        if class_order is None:
            rng = np.random.default_rng(seed)
            class_order_used = list(rng.permutation(10).tolist())
        else:
            class_order_used = list(class_order)
    elif dataset_l in {"tinyimagenet", "tiny-imagenet", "tinyimagenet-200"}:
        base_train = TinyImageNet(data_path, train=True)
        base_val = TinyImageNet(data_path, train=False)
        if class_order is None:
            # Deterministic but arbitrary ordering based on seed.
            # Actual class count inferred by continuum once data is indexed.
            class_order_used = []
        else:
            class_order_used = list(class_order)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    scenario_train = ClassIncremental(
        base_train,
        initial_increment=classes_per_task,
        increment=classes_per_task,
        transformations=train_tf.transforms,
        class_order=class_order_used if class_order_used else None,
    )
    scenario_val = ClassIncremental(
        base_val,
        initial_increment=classes_per_task,
        increment=classes_per_task,
        transformations=val_tf.transforms,
        class_order=class_order_used if class_order_used else None,
    )
    nb_classes = scenario_train.nb_classes

    max_tasks = math.ceil(nb_classes / classes_per_task)
    if num_tasks > max_tasks:
        raise ValueError(
            f"num_tasks={num_tasks} is too large for nb_classes={nb_classes} "
            f"with classes_per_task={classes_per_task} (max {max_tasks})."
        )

    return scenario_train, scenario_val, nb_classes, class_order_used


def get_backbone(backbone: str) -> nn.Module:
    if backbone == "resnet32":
        return resnet32()
    if backbone == "resnet20":
        return resnet20()
    if backbone == "resnet44":
        return resnet44()
    if backbone == "resnet56":
        return resnet56()
    raise ValueError(f"Unknown backbone: {backbone}")


def _freeze_(m: nn.Module) -> None:
    for p in m.parameters():
        p.requires_grad = False
    m.eval()


class CilClassifier(nn.Module):
    def __init__(self, embed_dim: int, nb_classes: int, device: torch.device):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.heads = nn.ModuleList([nn.Linear(embed_dim, nb_classes).to(device)])

    def __getitem__(self, index: int) -> nn.Linear:
        return self.heads[index]

    def __len__(self) -> int:
        return len(self.heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([head(x) for head in self.heads], dim=1)

    def adaption(self, nb_classes: int, device: torch.device) -> None:
        self.heads.append(nn.Linear(self.embed_dim, nb_classes).to(device))


class CilModel(nn.Module):
    def __init__(self, backbone: str, device: torch.device):
        super().__init__()
        self.backbone = get_backbone(backbone)
        self.fc: Optional[CilClassifier] = None
        self._device = device

    @property
    def feature_dim(self) -> int:
        # `resnet.py` backbones expose `.out_dim`
        return int(getattr(self.backbone, "out_dim"))

    def extract_vector(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feats = self.backbone(x)
        if self.fc is None:
            raise RuntimeError("Classifier head not initialized; call prev_model_adaption() first.")
        logits = self.fc(feats)
        return logits, feats

    def copy(self) -> "CilModel":
        return copy.deepcopy(self)

    def freeze(self) -> "CilModel":
        _freeze_(self)
        return self

    def prev_model_adaption(self, nb_classes: int) -> None:
        if self.fc is None:
            self.fc = CilClassifier(self.feature_dim, nb_classes, device=self._device)
        else:
            self.fc.adaption(nb_classes, device=self._device)

    def after_model_adaption(self, nb_new_classes: int, task_id: int) -> None:
        if task_id > 0:
            self.weight_align(nb_new_classes)

    @torch.no_grad()
    def weight_align(self, nb_new_classes: int) -> None:
        # Same computation as the tutorial: scale newest head’s weights by norm ratio.
        assert self.fc is not None
        w = torch.cat([head.weight.data for head in self.fc], dim=0)
        norms = torch.norm(w, dim=1)

        norm_old = norms[:-nb_new_classes]
        norm_new = norms[-nb_new_classes:]

        gamma = torch.mean(norm_old) / torch.mean(norm_new)
        self.fc[-1].weight.data = gamma * w[-nb_new_classes:]


class SoftTarget(nn.Module):
    def __init__(self, T: float = 2.0):
        super().__init__()
        self.T = float(T)

    def forward(self, out_s: torch.Tensor, out_t: torch.Tensor) -> torch.Tensor:
        return (
            F.kl_div(
                F.log_softmax(out_s / self.T, dim=1),
                F.softmax(out_t / self.T, dim=1),
                reduction="batchmean",
            )
            * (self.T * self.T)
        )


@dataclass(frozen=True)
class TrainConfig:
    backbone: str = "resnet32"
    batch_size: int = 128
    num_workers: int = 2
    epochs: int = 5
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    label_smoothing: float = 0.0
    lambda_kd: float = 0.5
    kd_temperature: float = 2.0
    old_fraction: float = 0.2  # fraction of old (memory) samples in delta training mix
    memory_size: int = 2000
    herding_method: str = "barycenter"
    fixed_memory: bool = False


def _accuracy_topk(logits: torch.Tensor, targets: torch.Tensor, k: int) -> float:
    k = min(int(k), int(logits.shape[1]))
    _, pred = logits.topk(k, dim=1, largest=True, sorted=True)
    correct = pred.eq(targets.view(-1, 1)).any(dim=1)
    return float(correct.float().mean().item())


def expected_calibration_error(
    probs: np.ndarray,
    targets: np.ndarray,
    num_bins: int = 15,
) -> float:
    conf = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    correct = (preds == targets).astype(np.float32)

    bins = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    n = len(targets)
    if n == 0:
        return 0.0

    for i in range(num_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == num_bins - 1:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        if not np.any(mask):
            continue
        acc_bin = float(correct[mask].mean())
        conf_bin = float(conf[mask].mean())
        ece += (mask.sum() / n) * abs(acc_bin - conf_bin)
    return float(ece)


@torch.no_grad()
def evaluate(
    model: CilModel,
    data_loader: DataLoader,
    device: torch.device,
    num_classes: int,
    ece_bins: int = 15,
) -> Dict[str, Any]:
    model.eval()

    all_targets: List[int] = []
    all_preds: List[int] = []
    all_probs: List[np.ndarray] = []

    top1_sum = 0.0
    top5_sum = 0.0
    n_sum = 0

    for images, targets, _task_ids in data_loader:
        images = images.to(device, non_blocking=True)
        targets_t = targets.to(device, non_blocking=True)
        logits, _ = model(images)

        bs = int(images.shape[0])
        n_sum += bs

        top1_sum += _accuracy_topk(logits, targets_t, 1) * bs
        top5_sum += _accuracy_topk(logits, targets_t, 5) * bs

        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
        preds = probs.argmax(axis=1).astype(np.int64)
        t = targets.detach().cpu().numpy().astype(np.int64)

        all_probs.append(probs)
        all_targets.extend(t.tolist())
        all_preds.extend(preds.tolist())

    if n_sum == 0:
        confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
        per_class_acc = [0.0 for _ in range(num_classes)]
        return {
            "top1": 0.0,
            "top5": 0.0,
            "ece": 0.0,
            "per_class_acc": per_class_acc,
            "confusion_matrix": confusion.tolist(),
            "n_samples": 0,
        }

    probs_all = np.concatenate(all_probs, axis=0)
    targets_all = np.asarray(all_targets, dtype=np.int64)
    preds_all = np.asarray(all_preds, dtype=np.int64)

    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(confusion, (targets_all, preds_all), 1)

    row_sums = confusion.sum(axis=1)
    diag = np.diag(confusion).astype(np.float32)
    per_class_acc = np.divide(diag, np.maximum(row_sums, 1), dtype=np.float32)

    ece = expected_calibration_error(probs_all, targets_all, num_bins=ece_bins)

    return {
        "top1": float(top1_sum / n_sum),
        "top5": float(top5_sum / n_sum),
        "ece": float(ece),
        "per_class_acc": per_class_acc.tolist(),
        "confusion_matrix": confusion.tolist(),
        "n_samples": int(n_sum),
    }


@torch.no_grad()
def extract_embeddings_by_class(
    model: CilModel,
    data_loader: DataLoader,
    device: torch.device,
    class_ids: Optional[Sequence[int]] = None,
    max_per_class: int = 256,
) -> Dict[int, np.ndarray]:
    """Return {class_id: embeddings[np_samples, dim]} from the loader."""
    model.eval()
    selected = set(class_ids) if class_ids is not None else None

    out: Dict[int, List[np.ndarray]] = {}
    counts: Dict[int, int] = {}

    for images, targets, _task_ids in data_loader:
        images = images.to(device, non_blocking=True)
        feats = model.extract_vector(images).detach().cpu().numpy()
        t = targets.detach().cpu().numpy().astype(np.int64)

        for i in range(len(t)):
            c = int(t[i])
            if selected is not None and c not in selected:
                continue
            cur = counts.get(c, 0)
            if cur >= max_per_class:
                continue
            out.setdefault(c, []).append(feats[i : i + 1])
            counts[c] = cur + 1

    return {c: np.concatenate(v, axis=0) for c, v in out.items()}


def _make_loader(
    dataset: Any,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def _maybe_subsample_memory(
    memory: "rehearsal.RehearsalMemory",
    rng: np.random.Generator,
    max_samples: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, y, t = memory.get()
    y = np.asarray(y)
    if max_samples <= 0 or len(y) <= max_samples:
        return x, y, t
    idx = rng.choice(len(y), size=max_samples, replace=False)
    return x[idx], y[idx], t[idx]


def train_one_task_delta(
    *,
    model: CilModel,
    teacher_model: Optional[CilModel],
    memory: "rehearsal.RehearsalMemory",
    dataset_train: Any,
    dataset_val: Any,
    task_id: int,
    nb_new_classes: int,
    known_classes: int,
    device: torch.device,
    config: TrainConfig,
    seed: int,
) -> Tuple[CilModel, CilModel, Dict[str, Any], Dict[str, Any]]:
    """Train incrementally on one task and evaluate on seen classes."""
    rng = np.random.default_rng(seed + task_id)

    model.prev_model_adaption(nb_new_classes)
    model.to(device)

    if task_id > 0 and config.memory_size > 0:
        new_n = len(dataset_train)
        old_frac = float(np.clip(config.old_fraction, 0.0, 0.95))
        if old_frac > 0.0:
            max_old = int((old_frac / max(1e-6, (1.0 - old_frac))) * new_n)
            if max_old > 0:
                mem_x, mem_y, mem_t = _maybe_subsample_memory(
                    memory, rng=rng, max_samples=max_old
                )
                dataset_train.add_samples(mem_x, mem_y, mem_t)

    train_loader = _make_loader(
        dataset_train,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=True,
    )
    val_loader = _make_loader(
        dataset_val,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=False,
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    kd_criterion = SoftTarget(T=config.kd_temperature)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )

    model.train()
    for _epoch in range(config.epochs):
        for images, targets, _task_ids in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            logits, _ = model(images)
            loss_ce = criterion(logits, targets)

            if teacher_model is not None and known_classes > 0:
                with torch.no_grad():
                    t_logits, _ = teacher_model(images)
                loss_kd = float(config.lambda_kd) * kd_criterion(
                    logits[:, :known_classes], t_logits
                )
            else:
                loss_kd = torch.tensor(0.0, device=device)

            loss = loss_ce + loss_kd

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    model.after_model_adaption(nb_new_classes, task_id=task_id)

    eval_metrics = evaluate(model, val_loader, device=device, num_classes=known_classes + nb_new_classes)

    # Update teacher (frozen copy).
    new_teacher = model.copy().freeze().to(device)

    # Update rehearsal memory using embeddings (same approach as tutorial).
    unshuffle_loader = _make_loader(
        dataset_train,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=False,
    )
    features: List[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for images, _targets, _task_ids in unshuffle_loader:
            images = images.to(device, non_blocking=True)
            feats = model.extract_vector(images).detach().cpu().numpy()
            features.append(feats)
    if features:
        feats_all = np.concatenate(features, axis=0)
        memory.add(*dataset_train.get_raw_samples(), feats_all)

    train_artifacts = {
        "known_classes": int(known_classes),
        "nb_new_classes": int(nb_new_classes),
        "total_classes": int(known_classes + nb_new_classes),
    }

    return model, new_teacher, eval_metrics, train_artifacts


def train_one_task_full_retrain(
    *,
    backbone: str,
    dataset_train_full: Any,
    dataset_val: Any,
    task_id: int,
    classes_per_task: int,
    total_tasks_seen: int,
    device: torch.device,
    config: TrainConfig,
    seed: int,
) -> Tuple[CilModel, Dict[str, Any]]:
    """Train from scratch on all seen data after each task."""
    _ = seed

    model = CilModel(backbone, device=device).to(device)
    # Build the same multi-head structure as incremental training would have by now.
    for _k in range(total_tasks_seen):
        model.prev_model_adaption(classes_per_task)

    train_loader = _make_loader(
        dataset_train_full,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=True,
    )
    val_loader = _make_loader(
        dataset_val,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=False,
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )

    model.train()
    for _epoch in range(config.epochs):
        for images, targets, _task_ids in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits, _ = model(images)
            loss = criterion(logits, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    total_classes = classes_per_task * total_tasks_seen
    eval_metrics = evaluate(model, val_loader, device=device, num_classes=total_classes)
    eval_metrics["task_id"] = int(task_id)
    return model, eval_metrics


def measure_peak_memory_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0))


def reset_peak_memory_stats(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def timed_call(fn, *args, device: torch.device, **kwargs):
    reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    wall = time.perf_counter() - t0
    peak = measure_peak_memory_mb(device)
    return out, wall, peak



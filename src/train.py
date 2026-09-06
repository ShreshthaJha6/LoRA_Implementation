"""Reproducible training loop and result persistence."""

import csv
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from transformers import get_scheduler

from .config import ExperimentConfig
from .evaluate import evaluate
from .model import parameter_counts


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for repeatable runs."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_experiment(
    model: nn.Module,
    train_loader: Any,
    validation_loader: Any,
    config: ExperimentConfig,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Train one configuration and return its measured, non-fabricated metrics."""

    set_seed(config.seed)
    device = device or get_device()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    model.to(device)
    optimizer = AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    steps_per_epoch = len(train_loader)
    if config.max_train_steps is not None:
        steps_per_epoch = min(steps_per_epoch, config.max_train_steps)
    total_steps = steps_per_epoch * config.num_epochs
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=int(total_steps * config.warmup_ratio),
        num_training_steps=total_steps,
    )

    started_at = time.perf_counter()
    model.train()
    for _ in range(config.num_epochs):
        for step, batch in enumerate(train_loader):
            batch = {name: value.to(device) for name, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            if config.max_train_steps is not None and step + 1 >= config.max_train_steps:
                break
        model.train()
    training_time = time.perf_counter() - started_at

    metrics = evaluate(model, validation_loader, device)
    metrics.update(parameter_counts(model))
    metrics.update(
        {
            "experiment": config.name,
            "rank": config.lora_rank if config.lora_rank is not None else 0,
            "training_time_seconds": training_time,
            "peak_gpu_memory_mb": (
                torch.cuda.max_memory_allocated(device) / (1024**2)
                if device.type == "cuda"
                else None
            ),
        }
    )
    return metrics


RESULT_COLUMNS = [
    "experiment",
    "rank",
    "validation_loss",
    "accuracy",
    "f1",
    "total_parameters",
    "trainable_parameters",
    "trainable_parameter_percentage",
    "training_time_seconds",
    "peak_gpu_memory_mb",
]


def append_result(result: dict[str, Any], results_path: Path) -> None:
    """Append one measured experiment row, creating the CSV when necessary."""

    results_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not results_path.exists() or results_path.stat().st_size == 0
    with results_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=RESULT_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow({column: result.get(column) for column in RESULT_COLUMNS})

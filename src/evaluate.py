"""Validation metrics for binary sequence classification."""

from typing import Iterable

import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn


@torch.no_grad()
def evaluate(model: nn.Module, validation_loader: Iterable, device: torch.device) -> dict[str, float]:
    """Evaluate a model and return mean loss, accuracy, and binary F1."""

    model.eval()
    losses: list[float] = []
    predictions: list[int] = []
    labels: list[int] = []

    for batch in validation_loader:
        batch = {name: value.to(device) for name, value in batch.items()}
        outputs = model(**batch)
        losses.append(outputs.loss.item() * batch["labels"].size(0))
        predictions.extend(outputs.logits.argmax(dim=-1).cpu().tolist())
        labels.extend(batch["labels"].cpu().tolist())

    example_count = len(labels)
    if not example_count:
        raise ValueError("The validation loader produced no examples.")
    return {
        "validation_loss": sum(losses) / example_count,
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, zero_division=0),
    }

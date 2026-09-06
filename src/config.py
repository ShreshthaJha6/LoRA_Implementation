"""Experiment configuration objects."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class ExperimentConfig:
    """All settings needed to run one SST-2 experiment."""

    name: str
    lora_rank: Optional[int] = None
    model_name: str = "distilbert-base-uncased"
    dataset_name: str = "glue"
    dataset_config: str = "sst2"
    max_length: int = 128
    train_batch_size: int = 32
    eval_batch_size: int = 64
    learning_rate: float = 2e-4
    num_epochs: int = 3
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    seed: int = 42
    # None preserves the full experiment; a small value is used only for smoke tests.
    max_train_steps: Optional[int] = None
    lora_alpha: Optional[int] = None
    lora_dropout: float = 0.1
    results_path: Path = Path("results/experiment_results.csv")

    @property
    def is_lora(self) -> bool:
        return self.lora_rank is not None

    @property
    def resolved_lora_alpha(self) -> int:
        if self.lora_rank is None:
            raise ValueError("A LoRA alpha is only defined for a LoRA experiment.")
        return self.lora_alpha if self.lora_alpha is not None else self.lora_rank * 2


def default_experiments() -> list[ExperimentConfig]:
    """Return the requested frozen baseline and two LoRA configurations."""

    return [
        ExperimentConfig(name="E0_frozen_baseline"),
        ExperimentConfig(name="E1_lora_r4", lora_rank=4),
        ExperimentConfig(name="E2_lora_r8", lora_rank=8),
    ]

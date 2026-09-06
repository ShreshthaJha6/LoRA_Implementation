"""Model construction and parameter accounting."""

from typing import Any

from peft import LoraConfig, TaskType, get_peft_model
from torch import nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .config import ExperimentConfig


def build_tokenizer(config: ExperimentConfig):
    return AutoTokenizer.from_pretrained(config.model_name, use_fast=True)


def build_model(config: ExperimentConfig) -> nn.Module:
    """Create a frozen classifier baseline or a PEFT LoRA classifier."""

    model = AutoModelForSequenceClassification.from_pretrained(config.model_name, num_labels=2)
    if not config.is_lora:
        for parameter in model.parameters():
            parameter.requires_grad = False
        # A frozen encoder still needs a trainable classification head to make predictions.
        for name, parameter in model.named_parameters():
            if name.startswith("pre_classifier") or name.startswith("classifier"):
                parameter.requires_grad = True
        return model

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=config.lora_rank,
        lora_alpha=config.resolved_lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=["q_lin", "v_lin"],
        bias="none",
    )
    return get_peft_model(model, peft_config)


def parameter_counts(model: nn.Module) -> dict[str, Any]:
    """Return parameter totals suitable for the results table."""

    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "trainable_parameter_percentage": 100 * trainable / total if total else 0.0,
    }

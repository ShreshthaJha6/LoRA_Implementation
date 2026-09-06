"""SST-2 loading and tokenization."""

from typing import Tuple

from datasets import Dataset
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding, PreTrainedTokenizerBase

from .config import ExperimentConfig


def load_sst2(tokenizer: PreTrainedTokenizerBase, config: ExperimentConfig) -> Tuple[Dataset, Dataset]:
    """Download SST-2 and return tokenized train and validation splits."""

    dataset = load_dataset(config.dataset_name, config.dataset_config)

    def tokenize(batch: dict) -> dict:
        return tokenizer(batch["sentence"], truncation=True, max_length=config.max_length)

    # Keep labels so the model can compute the training and validation loss.
    columns_to_remove = [column for column in dataset["train"].column_names if column != "label"]
    tokenized = dataset.map(tokenize, batched=True, remove_columns=columns_to_remove)
    tokenized = tokenized.rename_column("label", "labels")
    return tokenized["train"], tokenized["validation"]


def make_dataloaders(
    train_dataset: Dataset,
    validation_dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    config: ExperimentConfig,
) -> tuple[DataLoader, DataLoader]:
    """Build dynamically padded data loaders for a single experiment."""

    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(config.seed),
        collate_fn=collator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        collate_fn=collator,
    )
    return train_loader, validation_loader

"""Small plotting helpers for completed experiment results."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt


def save_comparison_figure(results_path: Path, figures_dir: Path) -> Path:
    """Save an accuracy-versus-trainable-parameters chart from measured CSV rows."""

    with results_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError("No experiment rows are available to plot.")

    figures_dir.mkdir(parents=True, exist_ok=True)
    figure_path = figures_dir / "accuracy_vs_trainable_parameters.png"
    labels = [row["experiment"] for row in rows]
    parameters = [int(row["trainable_parameters"]) for row in rows]
    accuracy = [float(row["accuracy"]) for row in rows]

    figure, axis = plt.subplots(figsize=(7, 4))
    axis.scatter(parameters, accuracy, s=70)
    for label, x_value, y_value in zip(labels, parameters, accuracy):
        axis.annotate(label, (x_value, y_value), xytext=(5, 5), textcoords="offset points")
    axis.set_xscale("log")
    axis.set_xlabel("Trainable parameters (log scale)")
    axis.set_ylabel("Validation accuracy")
    axis.set_title("LoRA parameter efficiency on SST-2")
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(figure_path, dpi=160)
    plt.close(figure)
    return figure_path


def save_performance_by_rank_figure(results_path: Path, figures_dir: Path) -> Path:
    """Save a grouped bar chart comparing validation accuracy and F1."""

    with results_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError("No experiment rows are available to plot.")

    figures_dir.mkdir(parents=True, exist_ok=True)
    figure_path = figures_dir / "performance_by_configuration.png"

    labels = []
    accuracy = []
    f1 = []

    for row in rows:
        experiment = row["experiment"]
        if experiment == "E0_frozen_baseline":
            label = "Frozen baseline"
        elif experiment == "E1_lora_r4":
            label = "LoRA (r=4)"
        elif experiment == "E2_lora_r8":
            label = "LoRA (r=8)"
        else:
            label = experiment

        labels.append(label)
        accuracy.append(float(row["accuracy"]))
        f1.append(float(row["f1"]))

    positions = range(len(labels))
    width = 0.35

    figure, axis = plt.subplots(figsize=(8, 5))

    accuracy_positions = [position - width / 2 for position in positions]
    f1_positions = [position + width / 2 for position in positions]

    axis.bar(accuracy_positions, accuracy, width=width, label="Accuracy")
    axis.bar(f1_positions, f1, width=width, label="F1")

    axis.set_xticks(list(positions))
    axis.set_xticklabels(labels)
    axis.set_ylabel("Validation score")
    axis.set_ylim(0.80, 0.92)
    axis.set_title("Validation performance across configurations")
    axis.legend()
    axis.grid(axis="y", alpha=0.3)

    figure.tight_layout()
    figure.savefig(figure_path, dpi=160)
    plt.close(figure)

    return figure_path

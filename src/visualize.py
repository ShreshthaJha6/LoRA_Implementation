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

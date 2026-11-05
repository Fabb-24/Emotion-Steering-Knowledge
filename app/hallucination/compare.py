import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import EMOTIONS, HALLUC_LAYERS_BY_MODEL, MODELS, ROOT, STEERING_MODE
from hallucination.analyze import draw_accuracy

OUT_DIR = ROOT / "results" / "general"
MODEL_ORDER = ["qwen", "llama", "qwen38"]  # order of the panels and of the heatmap columns


def load_metrics(model_name, dataset):
    """
    Load the metrics saved by `halluc analyze` for one model and dataset.

    Args:
        model_name: key of MODELS
        dataset: "bbf" or "bbc"

    Returns:
        DataFrame with the baseline row (emotion "none") and one row per layer and emotion
    """

    path = ROOT / "results" / model_name / "hallucination" / STEERING_MODE / f"{dataset}_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run `python app/cli.py halluc analyze` with MODEL_NAME = {model_name!r}")
    return pd.read_csv(path)


def plot_accuracy_all(dataset, hallucination=False, out_file=None):
    """
    Plot the accuracy bar chart of every model (see draw_accuracy) in one figure, one panel per model.
    With hallucination=True, plot the hallucination rate instead (wrong answers, "unknown" included: 1 - accuracy).

    Args:
        dataset: "bbf" or "bbc"
        hallucination: plot the hallucination rate instead of the accuracy
        out_file: path of the saved image (default: results/general/{dataset}_accuracy_all.png
                  or {dataset}_hallucination_all.png)
    """

    name = "hallucination_all" if hallucination else "accuracy_all"
    out_file = out_file or OUT_DIR / f"{dataset}_{name}.png"
    fig, axes = plt.subplots(len(MODEL_ORDER), 1, figsize=(14, 6 * len(MODEL_ORDER)))

    for ax, model_name in zip(axes, MODEL_ORDER):
        metrics = load_metrics(model_name, dataset)
        if hallucination:
            metrics["accuracy"] = 1 - metrics["accuracy"]  # draw_accuracy plots the "accuracy" column
        draw_accuracy(ax, metrics, f"{MODELS[model_name].split('/')[-1]} - {dataset.upper()}")
        if hallucination:
            ax.set_ylabel("hallucination rate (%)")

    fig.tight_layout()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=150)
    plt.close(fig)
    print(f"saved {out_file}")


def plot_yes_rate(dataset, out_file=None):
    """
    Plot the yes rate of every emotion (rows) at every model and layer (columns) as a heatmap:
    blue = always "no", white = 50% "yes", red = always "yes".

    Args:
        dataset: "bbf" or "bbc"
        out_file: path of the saved image (default: results/general/{dataset}_yes_rate.png)
    """

    out_file = out_file or OUT_DIR / f"{dataset}_yes_rate.png"
    columns, labels = [], []
    for model_name in MODEL_ORDER:
        metrics = load_metrics(model_name, dataset)
        for layer in HALLUC_LAYERS_BY_MODEL[model_name]:
            yes = metrics[metrics["layer"] == layer].set_index("emotion")["yes_rate"]
            columns.append(yes.reindex(EMOTIONS).values)
            labels.append(f"{MODELS[model_name].split('/')[-1].split('-')[0]}\nlayer {layer}")
    matrix = np.column_stack(columns)

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=0, vmax=1, aspect="auto")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(matrix[i, j] - 0.5) > 0.35 else "black")

    ax.set_xticks(range(len(labels)), labels, fontsize=8)
    ax.set_yticks(range(len(EMOTIONS)), EMOTIONS)
    ax.set_title(f"Yes rate - {dataset.upper()}")
    fig.colorbar(im, ax=ax, label="yes rate", fraction=0.03)
    fig.tight_layout()

    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=150)
    plt.close(fig)
    print(f"saved {out_file}")

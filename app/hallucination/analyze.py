import matplotlib.pyplot as plt
import pandas as pd

from config import EMOTIONS, HALLUC_RESULTS_DIR, MODEL_ID, STEERING_MODE
from hallucination.evaluate import answers_path

# meaning of the steering strength in the labels of the plots
STRENGTH_LABEL = {"fixed": "α", "std": "σ"}


def compute_metrics(dataset):
    """
    Compare every steering condition with the baseline, question by question:
    accuracy ("unknown" answers count as wrong), flip rate (split into correct -> wrong and wrong -> correct),
    rate of "yes" answers and rate of "unknown" answers (the model leaves the yes/no format).
    Save the result to results/{model}/hallucination/{mode}/{dataset}_metrics.csv.

    Args:
        dataset: "bbf" or "bbc"

    Returns:
        DataFrame with one row per condition (the baseline first)
    """

    # load the answers of the dataset and check that there is a baseline
    path = answers_path(dataset)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run `python app/cli.py halluc eval` first")
    answers = pd.read_csv(path)

    # obtain the baseline answers
    base = answers[answers["emotion"] == "none"].set_index("id")
    if base.empty:
        raise ValueError(f"no baseline in {path.name}: run `python app/cli.py halluc eval --baseline-only`")
    base_ok = base["pred"] == base["expected"]
    base_yes = (base["pred"] == "yes").mean()
    base_unknown = (base["pred"] == "unknown").mean()

    rows = [{
        "layer": None,
        "emotion": "none",
        "alpha": 0.0,
        "accuracy": base_ok.mean(),
        "delta_accuracy": 0.0,
        "flip_rate": 0.0,
        "correct_to_wrong": 0.0,
        "wrong_to_correct": 0.0,
        "yes_rate": base_yes,
        "delta_yes_rate": 0.0,
        "unknown_rate": base_unknown,
        "delta_unknown_rate": 0.0,
    }]

    steered = answers[answers["emotion"] != "none"]

    # compare every condition with the baseline, question by question
    for (layer, emotion, alpha), group in steered.groupby(["layer", "emotion", "alpha"]):
        group = group.set_index("id").loc[base.index]
        ok = group["pred"] == group["expected"]
        flip = group["pred"] != base["pred"]
        yes = (group["pred"] == "yes").mean()
        unknown = (group["pred"] == "unknown").mean()

        rows.append({
            "layer": int(layer),
            "emotion": emotion,
            "alpha": alpha,
            "accuracy": ok.mean(),
            "delta_accuracy": ok.mean() - base_ok.mean(),
            "flip_rate": flip.mean(),
            "correct_to_wrong": (flip & base_ok).mean(),
            "wrong_to_correct": (flip & ~base_ok).mean(),
            "yes_rate": yes,
            "delta_yes_rate": yes - base_yes,
            "unknown_rate": unknown,  # high = the model leaves the yes/no format
            "delta_unknown_rate": unknown - base_unknown,
        })

    # save the metrics to a CSV file
    metrics = pd.DataFrame(rows).round(4)
    metrics["layer"] = metrics["layer"].astype("Int64")  # integers, empty for the baseline
    metrics.to_csv(HALLUC_RESULTS_DIR / f"{dataset}_metrics.csv", index=False)
    return metrics


def plot_metrics(dataset, metrics):
    """
    One figure per layer with four bar charts over the emotions: change of accuracy,
    flip rate (correct -> wrong and wrong -> correct stacked), rate of "yes" and rate of "unknown" (baseline as a line).
    Save the plots to results/{model}/hallucination/{mode}/{dataset}_layer{layer}.png.

    Args:
        dataset: "bbf" or "bbc"
        metrics: DataFrame returned by compute_metrics
    """

    base = metrics[metrics["emotion"] == "none"].iloc[0]
    steered = metrics[metrics["emotion"] != "none"]

    # plot one figure per layer, with four bar charts over the emotions
    for layer, group in steered.groupby("layer"):
        group = group.set_index("emotion")
        group = group.loc[[e for e in EMOTIONS if e in group.index]]  # same order as EMOTIONS
        x = range(len(group))
        labels = [f"{e}\n({STRENGTH_LABEL[STEERING_MODE]}={a})" for e, a in zip(group.index, group["alpha"])]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()  # [top left, top right, bottom left, bottom right]

        axes[0].bar(x, 100 * group["delta_accuracy"])
        axes[0].axhline(0, linewidth=0.8, color="black")
        axes[0].set_title(f"Accuracy change (baseline {100 * base['accuracy']:.1f}%)")
        axes[0].set_ylabel("percentage points")

        axes[1].bar(x, 100 * group["correct_to_wrong"], label="correct → wrong")
        axes[1].bar(x, 100 * group["wrong_to_correct"], bottom=100 * group["correct_to_wrong"], label="wrong → correct")
        axes[1].set_title("Answers changed by steering")
        axes[1].set_ylabel("% of questions")
        axes[1].legend()

        axes[2].bar(x, 100 * group["yes_rate"])
        axes[2].axhline(100 * base["yes_rate"], linestyle="--", color="black", label="baseline")
        axes[2].set_title('Answers "yes"')
        axes[2].set_ylabel("% of questions")
        axes[2].legend()

        axes[3].bar(x, 100 * group["unknown_rate"])
        axes[3].axhline(100 * base["unknown_rate"], linestyle="--", color="black", label="baseline")
        axes[3].set_title('Answers neither "yes" nor "no"')
        axes[3].set_ylabel("% of questions")
        axes[3].legend()

        for ax in axes:
            ax.set_xticks(list(x), labels, rotation=45, ha="right", fontsize=8)
            ax.grid(axis="y", linewidth=0.5)
            ax.set_axisbelow(True)
        fig.suptitle(f"{dataset.upper()}, steering at layer {layer} ({STEERING_MODE} mode)")
        fig.tight_layout()
        fig.savefig(HALLUC_RESULTS_DIR / f"{dataset}_layer{layer}.png", dpi=150)
        plt.close(fig)


def draw_accuracy(ax, metrics, title):
    """
    Draw on ax a grouped bar chart over the emotions: for each emotion, the accuracy of the baseline
    and the accuracy with steering at each analyzed layer.

    Args:
        ax: matplotlib Axes to draw on
        metrics: DataFrame returned by compute_metrics (or read from {dataset}_metrics.csv)
        title: title of the chart
    """

    base = metrics[metrics["emotion"] == "none"].iloc[0]
    steered = metrics[metrics["emotion"] != "none"]
    layers = sorted(steered["layer"].unique())
    emotions = [e for e in EMOTIONS if e in steered["emotion"].values]

    n_bars = 1 + len(layers)
    width = 0.8 / n_bars
    colors = plt.cm.Blues([0.35 + 0.6 * i / max(len(layers) - 1, 1) for i in range(len(layers))])

    ax.bar([i - 0.4 + width / 2 for i in range(len(emotions))], [100 * base["accuracy"]] * len(emotions),
           width, color="gray", label="baseline")
    for j, (layer, color) in enumerate(zip(layers, colors), start=1):
        acc = steered[steered["layer"] == layer].set_index("emotion")["accuracy"].reindex(emotions)
        # int(): the layer is read back as float from the csv, because the baseline row has no layer
        ax.bar([i - 0.4 + width / 2 + j * width for i in range(len(emotions))], 100 * acc,
               width, color=color, label=f"layer {int(layer)}")

    ax.set_xticks(range(len(emotions)), emotions)
    ax.set_ylabel("accuracy (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(ncol=n_bars, loc="upper center", bbox_to_anchor=(0.5, -0.08), frameon=False)
    ax.set_title(title)


def plot_accuracy(dataset, metrics):
    """
    Plot the grouped bar chart of the accuracy of the active model (see draw_accuracy).
    Save the plot to results/{model}/hallucination/{mode}/{dataset}_accuracy.png.

    Args:
        dataset: "bbf" or "bbc"
        metrics: DataFrame returned by compute_metrics
    """

    fig, ax = plt.subplots(figsize=(14, 6))
    draw_accuracy(ax, metrics, f"{MODEL_ID.split('/')[-1]} - {dataset.upper()}")
    fig.tight_layout()
    fig.savefig(HALLUC_RESULTS_DIR / f"{dataset}_accuracy.png", dpi=150)
    plt.close(fig)


def main(datasets):
    """
    Compute, print and plot the metrics of each dataset.

    Args:
        datasets: list of "bbf" / "bbc"
    """

    for dataset in datasets:
        metrics = compute_metrics(dataset)
        print(f"\n=== {dataset.upper()}")
        print(metrics.to_string(index=False))
        plot_metrics(dataset, metrics)
        plot_accuracy(dataset, metrics)

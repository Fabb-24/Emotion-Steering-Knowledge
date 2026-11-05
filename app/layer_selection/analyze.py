import json
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay

from config import EMOTIONS, LAYER_RESULTS_DIR, MODEL_NAME, MODELS


def global_layer(df):
    """
    Return the layer with the highest macro F1 (ties go to the lowest layer).
    Also return the macro F1 and per-emotion F1 at that layer.

    Args:
        df: probing_metrics.csv as a DataFrame

    Returns:
        per_layer: DataFrame with macro F1 per layer
        best: dict with the best layer, its macro F1 and per-emotion F1
    """

    # Compute macro F1 per layer
    per_layer = df.groupby("layer")["macro_f1"].first().sort_index().to_frame()

    # idxmax returns the first occurrence of the max, which is the lowest layer in case of ties
    layer = int(per_layer["macro_f1"].idxmax())

    # Get the per-emotion F1 at that layer
    at_layer = df[df["layer"] == layer].set_index("emotion")["f1"]

    return per_layer, {
        "layer": layer,
        "macro_f1": round(float(per_layer.loc[layer, "macro_f1"]), 4),
        "f1_per_emotion": {e: round(float(at_layer[e]), 4) for e in EMOTIONS},
    }


def layer_scores(df, per_layer):
    """
    Return a dict with macro F1 per layer and per-emotion F1 per layer.
    This is used to save layer_f1.json.

    Args:
        df: probing_metrics.csv as a DataFrame
        per_layer: DataFrame with macro F1 per layer

    Returns:
        dict with macro F1 per layer and per-emotion F1 per layer
    """

    # Compute per-emotion F1 per layer
    pivot = df.pivot(index="emotion", columns="layer", values="f1").loc[EMOTIONS]

    return {
        "macro_f1": {int(l): round(float(v), 4) for l, v in per_layer["macro_f1"].items()},
        "f1_per_emotion": {e: {int(l): round(float(v), 4) for l, v in pivot.loc[e].items()} for e in EMOTIONS},
    }


def plot_macro_f1(per_layer, best):
    """
    Plot macro F1 per layer and highlight the best layer.
    Save the plot to results/{model}/layer_selection/global_layer_macro_f1.png.

    Args:
        per_layer: DataFrame with macro F1 per layer
        best: dict with the best layer, its macro F1 and per-emotion F1
    """

    layers = per_layer.index
    values = per_layer["macro_f1"]

    fig, ax = plt.subplots(figsize=(12, 5))

    (line,) = ax.plot(layers, values, linewidth=2, marker="o", markersize=4, zorder=2)

    ax.plot(best["layer"], best["macro_f1"], "o", markersize=10, color=line.get_color(), zorder=3)
    ax.annotate(
        f"layer {best['layer']}\nmacro F1 {best['macro_f1']:.3f}",
        (best["layer"], best["macro_f1"]),
        xytext=(0, -38),
        textcoords="offset points",
        ha="center",
        fontsize=9
    )

    ax.set_xticks(layers[::2] if len(layers) > 40 else layers)
    ax.set_xlim(layers.min() - 0.5, layers.max() + 0.5)
    ax.set_ylim(0, values.max() + 0.1)
    ax.set_xlabel("layer")
    ax.set_ylabel("macro F1 (test)")
    ax.set_title("Linear probe macro F1 per layer", loc="left")
    ax.grid(axis="y", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    fig.tight_layout()
    fig.savefig(LAYER_RESULTS_DIR / "global_layer_macro_f1.png", dpi=150)


def confusion_at_layer(layer=None):
    """
    Plot the confusion matrix at a given layer.
    If layer is None, use the layer with the highest macro F1.
    Save the plot to results/{model}/layer_selection/confusion_layer{layer}.png.

    Args:
        layer: int or None, layer to inspect (default: highest macro F1)
    """

    if layer is None:
        _, best = global_layer(pd.read_csv(LAYER_RESULTS_DIR / "probing_metrics.csv"))
        layer = best["layer"]

    preds = pd.read_csv(LAYER_RESULTS_DIR / "probing_predictions.csv")
    preds = preds[preds["layer"] == layer]

    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay.from_predictions(
        preds["label"],
        preds["pred"],
        labels=EMOTIONS,
        cmap="Blues",
        xticks_rotation=45,
        ax=ax
    )
    
    model = MODELS[MODEL_NAME].split("/")[-1].split("-")[0]
    ax.set_title(f"{model} - layer {layer}")
    fig.tight_layout()
    fig.savefig(LAYER_RESULTS_DIR / f"confusion_layer{layer}.png", dpi=150)


def main():
    """
    Run the analysis: best layer by macro F1, layer_f1.json, macro F1 plot and confusion matrix.
    """

    # Load probing metrics and find the best layer
    df = pd.read_csv(LAYER_RESULTS_DIR / "probing_metrics.csv")
    per_layer, best = global_layer(df)

    # Save layer_f1.json with macro F1 per layer and per-emotion F1 per layer
    with open(LAYER_RESULTS_DIR / "layer_f1.json", "w") as f:
        json.dump(layer_scores(df, per_layer), f, indent=2)
    print(f"global layer {best['layer']:2d}  macro_f1={best['macro_f1']:.3f}")

    # Print the F1 of every emotion at the best layer
    for emotion, f1 in best["f1_per_emotion"].items():
        print(f"  {emotion:12s} f1={f1:.3f}")

    # Plot macro F1 per layer and highlight the best layer
    plot_macro_f1(per_layer, best)

    # Plot confusion matrix at the best layer
    confusion_at_layer(best["layer"])


if __name__ == "__main__":
    main()

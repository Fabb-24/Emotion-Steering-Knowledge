import json
import matplotlib.pyplot as plt

from config import HALLUC_LAYERS_BY_MODEL, MODELS, ROOT

OUT_FILE = ROOT / "results" / "general" / "probing_macro_f1.png"

STYLES = {
    "qwen": {"color": "#2a78d6", "marker": "o", "label_dy": 6},
    "llama": {"color": "#eb6834", "marker": "s", "label_dy": 6},
    "qwen38": {"color": "#1baf7a", "marker": "^", "label_dy": -12},
}


def load_macro_f1(model_name):
    """
    Load the macro F1 per layer saved by `layer analyze` for one model.

    Args:
        model_name: key of MODELS

    Returns:
        dict layer -> macro F1
    """

    path = ROOT / "results" / model_name / "layer_selection" / "layer_f1.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run `python app/cli.py layer analyze` with MODEL_NAME = {model_name!r}")
    with open(path) as f:
        return {int(l): v for l, v in json.load(f)["macro_f1"].items()}


def plot_comparison(out_file=OUT_FILE):
    """
    Plot the macro F1 of every model against the relative depth of the layer (layer / number of layers),
    mark the layers used for steering (HALLUC_LAYERS_BY_MODEL) and save the figure.

    Args:
        out_file: path of the saved image
    """

    fig, ax = plt.subplots(figsize=(7, 3.6))

    for model_name, style in STYLES.items():
        scores = load_macro_f1(model_name)
        n_layers = max(scores)
        depth = [l / n_layers for l in scores]
        
        ax.plot(depth, list(scores.values()), color=style["color"], linewidth=1.5, marker=style["marker"],
                markevery=[], label=MODELS[model_name].split("/")[-1])

        for layer in HALLUC_LAYERS_BY_MODEL[model_name]:
            x, y = layer / n_layers, scores[layer]
            ax.plot(x, y, linestyle="none", marker=style["marker"], markersize=6,
                    color=style["color"], markeredgecolor="white", markeredgewidth=1)
            ax.annotate(str(layer), (x, y), textcoords="offset points", xytext=(0, style["label_dy"]),
                        ha="center", fontsize=7, color="#52514e")

    ax.set_xlabel("relative depth (layer / number of layers)")
    ax.set_ylabel("macro F1")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.7)
    ax.grid(axis="y", color="#e6e5e0", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()

    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_file}")


if __name__ == "__main__":
    plot_comparison()

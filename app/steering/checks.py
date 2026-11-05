import json
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, f1_score

from config import (EMOTIONS, LAYER_RESULTS_DIR, MODEL_NAME, MODELS, STEERING_METHOD, STEERING_RESULTS_DIR)
from data_processing.activations import load_acts
from steering.vectors import load_steering_split, load_vectors


def cosine_matrix(vectors):
    """
    Cosine similarity between every pair of steering vectors.

    Args:
        vectors: Tensor of shape [n_emotions, d]

    Returns:
        Tensor of shape [n_emotions, n_emotions]
    """

    v = vectors / vectors.norm(dim=-1, keepdim=True)
    return v @ v.T


def plot_cosine(sim, layer, method=STEERING_METHOD):
    """
    Plot the cosine similarity matrix as a heatmap (fixed scale -1..1, comparable across layers).
    Save the plot to results/{model}/steering/{method}/cosine_layer{layer}.png.

    Args:
        sim: Tensor of shape [n_emotions, n_emotions]
        layer: layer of the vectors
        method: method used to compute the vectors
    """

    fig, ax = plt.subplots(figsize=(8, 7))
    disp = ConfusionMatrixDisplay(sim.numpy(), display_labels=EMOTIONS)
    disp.plot(cmap="RdBu_r", values_format=".2f", xticks_rotation=45, ax=ax)
    disp.im_.set_clim(-1, 1)
    ax.set_xlabel("")
    ax.set_ylabel("")
    
    model = MODELS[MODEL_NAME].split("/")[-1].split("-")[0]
    ax.set_title(f"{model} - layer {layer}")
    fig.tight_layout()

    out_dir = STEERING_RESULTS_DIR / method
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"cosine_layer{layer}.png", dpi=150)
    plt.close(fig)


def cosine_check(layer, method=STEERING_METHOD, top_pairs=4):
    """
    Cosine similarity between the saved steering vectors of a layer: plot + most similar pairs.

    Args:
        layer: layer of the vectors
        method: method used to compute the vectors
        top_pairs: number of most similar pairs to print

    Returns:
        sim: Tensor of shape [n_emotions, n_emotions]
    """

    # Load the saved vectors and compute the cosine similarity matrix
    saved = load_vectors(layer, method)
    sim = cosine_matrix(saved["vectors"])
    plot_cosine(sim, layer, method)

    # Print the most similar pairs of emotions
    emotions = saved["emotions"]
    pairs = sorted(
        ((sim[i, j].item(), emotions[i], emotions[j])
            for i in range(len(emotions)) for j in range(i + 1, len(emotions))),
        reverse=True
    )
    print(f"most similar vectors at layer {layer}:")
    for value, a, b in pairs[:top_pairs]:
        print(f"  {a:12s} {b:12s} {value:.2f}")

    return sim


def projection_check(layer, method=STEERING_METHOD):
    """
    Classify the held-out sentences of the steering split by projecting them on the saved steering vectors
    (computed only on the steering train split).
    Save the result to results/{model}/steering/{method}/projection_layer{layer}.json.

    Args:
        layer: layer to check
        method: method used to compute the vectors

    Returns:
        dict with accuracy, macro F1, probe macro F1 (if available, as a reference) and F1 per emotion
    """

    # Load the saved vectors and the held-out test split, and check that they match
    saved = load_vectors(layer, method)
    test_ids = load_steering_split("test")["id"].tolist()
    if test_ids != saved["test_ids"]:
        raise ValueError(f"the steering split changed after the vectors of layer {layer} were computed: re-run `python app/cli.py steer vectors --layer {layer}`")

    # Load the activations of the test split and classify them by projection on the vectors
    x_test, y_test = load_acts(test_ids, layer=layer)
    v = saved["vectors"] / saved["vectors"].norm(dim=-1, keepdim=True)
    pred = ((x_test - saved["mean"]) @ v.T).argmax(dim=-1)

    # Compute metrics and save them to a JSON file
    f1s = f1_score(y_test, pred, labels=range(len(EMOTIONS)), average=None)
    result = {
        "layer": layer,
        "method": method,
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "macro_f1": round(float(f1s.mean()), 4),
        "probe_macro_f1": None,
        "f1_per_emotion": {e: round(float(f), 4) for e, f in zip(EMOTIONS, f1s)},
    }
    metrics_file = LAYER_RESULTS_DIR / "probing_metrics.csv"
    if metrics_file.exists():
        probe = pd.read_csv(metrics_file)
        probe = probe[probe["layer"] == layer]["macro_f1"]
        if len(probe):
            result["probe_macro_f1"] = round(float(probe.iloc[0]), 4)

    out_dir = STEERING_RESULTS_DIR / method
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"projection_layer{layer}.json", "w") as f:
        json.dump(result, f, indent=2)

    probe_txt = (f"  (probe {result['probe_macro_f1']:.3f})" if result["probe_macro_f1"] is not None else "")
    print(f"projection on unseen sentences at layer {layer}: acc={result['accuracy']:.3f}  macro_f1={result['macro_f1']:.3f}{probe_txt}")
    for emotion, f1 in result["f1_per_emotion"].items():
        print(f"  {emotion:12s} f1={f1:.3f}")
    return result

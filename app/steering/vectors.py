import torch

from config import EMOTIONS, MODEL_ID, STEERING_DIR, STEERING_METHOD, STEERING_SPLITS_DIR
from data_processing.activations import load_acts
from data_processing.splits import load_split


def diff_means(x, y, n_classes):
    """
    Difference of means: for each emotion, mean of its sentences minus mean of all the other sentences.

    Args:
        x: Tensor of shape [N, d], activations of one layer
        y: Tensor of shape [N], integer labels
        n_classes: number of emotions

    Returns:
        Tensor of shape [n_classes, d], one steering vector per emotion
    """

    return torch.stack([x[y == k].mean(dim=0) - x[y != k].mean(dim=0) for k in range(n_classes)])


METHODS = {"diff_means": diff_means}


def vectors_path(layer, method=STEERING_METHOD):
    """
    Path to the saved steering vectors of a layer.

    Args:
        layer: layer of the vectors
        method: method used to compute them

    Returns:
        Path to the saved vectors
    """

    return STEERING_DIR / method / f"layer{layer}.pt"


def load_steering_split(name):
    """
    Load the steering split (train/test) built by `steer dataset`.

    Args:
        name: "train" or "test"

    Returns:
        DataFrame with the split
    """

    if not (STEERING_SPLITS_DIR / f"{name}.csv").exists():
        raise FileNotFoundError(f"{STEERING_SPLITS_DIR / f'{name}.csv'} not found: run `python app/cli.py steer dataset` first")
    return load_split(name, STEERING_SPLITS_DIR)


def compute_vectors(layer, method=STEERING_METHOD):
    """
    Compute the steering vectors of one layer from the train sentences of the steering split
    and save them to data/steering/{model}/{method}/layer{layer}.pt.

    Args:
        layer: layer to compute the vectors for
        method: key of METHODS

    Returns:
        dict with vectors [n_emotions, d], emotions, layer, method, act_norm, mean, proj_std, test_ids, model
    """

    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}, available: {list(METHODS)}")

    # Load activations and labels of the train split
    x, y = load_acts(load_steering_split("train")["id"].tolist(), layer=layer)
    vectors = METHODS[method](x, y, len(EMOTIONS))
    mean = x.mean(dim=0)

    # Coordinate of every train sentence along every emotion direction: [N, n_emotions]
    v_hat = vectors / vectors.norm(dim=-1, keepdim=True)
    proj = (x - mean) @ v_hat.T

    # Save the vectors with the statistics used by the steering modes and the checks
    result = {
        "vectors": vectors,
        "emotions": EMOTIONS,
        "layer": layer,
        "method": method,
        "act_norm": x.norm(dim=-1).mean().item(),
        "mean": mean,
        "proj_std": 1.4826 * (proj - proj.median(dim=0).values).abs().median(dim=0).values,
        "test_ids": load_steering_split("test")["id"].tolist(),
        "model": MODEL_ID,
    }
    path = vectors_path(layer, method)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, path)

    return result


def load_vectors(layer, method=STEERING_METHOD):
    """
    Load the steering vectors saved by compute_vectors.

    Args:
        layer: layer of the vectors
        method: method used to compute them

    Returns:
        dict with vectors [n_emotions, d], emotions, layer, method, act_norm, mean, proj_std, test_ids, model
    """

    path = vectors_path(layer, method)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run `python app/cli.py steer vectors --layer {layer}` first")
    saved = torch.load(path)

    # files computed before the model was saved in them have no "model": they are accepted
    if saved.get("model", MODEL_ID) != MODEL_ID:
        raise ValueError(f"{path} was computed with {saved['model']}, not {MODEL_ID}: re-run `python app/cli.py steer vectors --layer {layer}`")
    return saved

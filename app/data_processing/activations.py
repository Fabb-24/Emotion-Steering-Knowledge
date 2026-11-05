import torch

from config import ACTIVATIONS_FILE, MODEL_ID


def _open_acts():
    """
    Open data/activations/{model}/acts_all.pt (memory-mapped) and check that it belongs to the active model.

    Returns:
        dict saved by extract_all
    """

    if not ACTIVATIONS_FILE.exists():
        raise FileNotFoundError(f"{ACTIVATIONS_FILE} not found: run `python app/cli.py extract` first")

    data = torch.load(ACTIVATIONS_FILE, mmap=True)  # mmap: read from disk only the layers/sentences we select
    if data.get("model") != MODEL_ID:
        raise ValueError(f"{ACTIVATIONS_FILE} was extracted with {data.get('model')}, not {MODEL_ID}: re-run `python app/cli.py extract`")
    return data


def load_acts(ids=None, layer=None):
    """
    Load the last-token residual stream from data/activations/{model}/acts_all.pt.
    If ids is given, return only those sentences, in the same order as ids.
    If layer is given, return only that layer.

    Args:
        ids: list of sentence ids ("<emotion>-<row>") or None for every sentence
        layer: int or None for every layer

    Returns:
        acts: Tensor (float32) of shape [L+1, N, d] if layer is None, else [N, d],
              where L is the number of layers, N is the number of sentences, and d is the hidden size.
        labels: Tensor of shape [N], containing the integer labels for each sentence.
    """

    data = _open_acts()
    acts, labels = data["acts"], data["labels"]

    if layer is not None:
        acts = acts[layer]

    if ids is None:
        return acts.float(), labels

    position = {id_: i for i, id_ in enumerate(data["ids"])}
    missing = [id_ for id_ in ids if id_ not in position]
    if missing:
        raise KeyError(f"{len(missing)} ids not in {ACTIVATIONS_FILE.name} (e.g. {missing[:3]}): re-run `python app/cli.py extract`")
    idx = torch.tensor([position[id_] for id_ in ids])
    return acts[..., idx, :].float(), labels[idx]


def n_layers():
    """
    Number of hidden states saved per sentence (embeddings + every layer).

    Returns:
        int, e.g. 29 for Qwen2.5-7B (embeddings + 28 layers), 33 for Llama-3.1-8B (embeddings + 32 layers)
    """

    return _open_acts()["acts"].shape[0]

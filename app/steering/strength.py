from config import STEERING_MODE
from models.steering import steer

MODES = ["fixed", "std"]


def check_vectors(saved, layer, mode=STEERING_MODE):
    """
    Check that the mode exists and that the saved vectors have what the mode needs
    (call it before any GPU work, so that a missing statistic is found immediately).

    Args:
        saved: steering vectors of the layer, as returned by load_vectors
        layer: layer of the vectors
        mode: one of MODES
    """

    if mode not in MODES:
        raise ValueError(f"unknown steering mode {mode!r}, available: {MODES}")
    if mode == "std" and "proj_std" not in saved:
        raise ValueError(f"the vectors of layer {layer} have no projection statistics, needed by the 'std' mode: re-run `python app/cli.py steer vectors --layer {layer}`")


def steering(model, layer, saved, emotion, strength, mode=STEERING_MODE):
    """
    Context manager that steers one layer towards an emotion with the given mode:
        "fixed": add strength * act_norm along the emotion
        "std":   add strength * std of the projections on the emotion direction

    Args:
        model: the model returned by load_model
        layer: layer to steer
        saved: steering vectors of the layer, as returned by load_vectors
        emotion: emotion to steer towards
        strength: multiple of act_norm ("fixed") or number of standard deviations ("std")
        mode: one of MODES

    Returns:
        context manager: the model is steered inside the `with` block
    """

    check_vectors(saved, layer, mode)
    k = saved["emotions"].index(emotion)
    vector = saved["vectors"][k]

    if mode == "fixed":
        return steer(model, layer, vector, strength, saved["act_norm"])
    return steer(model, layer, vector, strength, saved["proj_std"][k].item())

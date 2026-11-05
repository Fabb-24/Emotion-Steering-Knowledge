import pandas as pd

from config import (EMOTIONS, GEN_PROMPTS, STEERING_ALPHAS, STEERING_METHOD, STEERING_MODE, STEERING_RESULTS_DIR)
from models.generation import generate
from steering.strength import check_vectors, steering
from steering.vectors import load_vectors


def run_generation(layer, tokenizer, model, emotions=EMOTIONS, alphas=STEERING_ALPHAS, method=STEERING_METHOD, prompts=GEN_PROMPTS):
    """
    Generate answers to neutral prompts steering one layer with each emotion vector and strength.
    The baseline (no steering) is generated once.
    The meaning of alpha depends on STEERING_MODE (multiple of act_norm for "fixed", standard deviations for "std").
    Save the texts to results/{model}/steering/{method}/{mode}/generation_layer{layer}.csv.

    Args:
        layer: layer to steer
        tokenizer: the tokenizer for the model
        model: the model itself
        emotions: emotions to steer towards
        alphas: steering strengths, with the meaning of the active STEERING_MODE (0 = baseline)
        method: method used to compute the vectors
        prompts: neutral user messages

    Returns:
        DataFrame with columns layer, emotion, alpha, prompt, text
    """

    # Check that the given emotions are valid
    unknown = [e for e in emotions if e not in EMOTIONS]
    if unknown:
        raise ValueError(f"unknown emotions {unknown}, available: {EMOTIONS}")

    saved = load_vectors(layer, method)
    check_vectors(saved, layer)
    rows = []

    # add the texts of one condition to the rows and print the first one
    def add(emotion, alpha, texts):
        rows.extend({"layer": layer, "mode": STEERING_MODE, "emotion": emotion, "alpha": alpha, "prompt": p, "text": t} for p, t in zip(prompts, texts))
        print(f"[layer {layer} | {STEERING_MODE} | {emotion} | alpha={alpha}] {texts[0][:250]!r}")

    # Generate the baseline (no steering) and add it to the rows
    add("none", 0.0, generate(prompts, tokenizer, model))

    # Generate texts for each emotion and alpha, steering the given layer with the corresponding vector
    for emotion in emotions:
        for alpha in alphas:
            if alpha == 0:
                continue
            with steering(model, layer, saved, emotion, alpha):
                add(emotion, alpha, generate(prompts, tokenizer, model))

    # Save the generated texts to a CSV file
    df = pd.DataFrame(rows)
    out_dir = STEERING_RESULTS_DIR / method / STEERING_MODE
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"generation_layer{layer}.csv", index=False)
    return df

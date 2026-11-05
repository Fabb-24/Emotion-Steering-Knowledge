import numpy as np
import pandas as pd

from config import (BB_PROMPTS, EMOTIONS, HALLUC_ALPHAS, HALLUC_RESULTS_DIR, STD_ALPHA, STEERING_METHOD, STEERING_MODE)
from data_processing.beliefbank import load_bb_sample
from models.qa import ask_yes_no, parse_answer
from steering.strength import check_vectors, steering
from steering.vectors import load_vectors


def answers_path(dataset):
    """
    Path to the raw answers of a BeliefBank dataset: results/{model}/hallucination/{mode}/{dataset}_answers.csv.

    Args:
        dataset: "bbf" or "bbc"

    Returns:
        Path to the answers file
    """

    return HALLUC_RESULTS_DIR / f"{dataset}_answers.csv"


def _answer(sample, tokenizer, model, prompt, layer, emotion, alpha):
    """
    Answer every question of the sample with the current model (steered or not).

    Args:
        sample: DataFrame with the questions to answer
        tokenizer: the tokenizer for the model
        model: the model itself
        prompt: question template of the dataset; {statement} = the question
        layer: layer to steer, or np.nan for the baseline
        emotion: emotion to steer towards, or "none" for the baseline
        alpha: steering strength, or 0.0 for the baseline

    Returns:
        DataFrame with one row per question: id, question, expected, layer, emotion, mode, alpha, prompt, answer_text, pred
    """

    # generate the answer to every question and read it as yes / no / unknown
    texts = ask_yes_no(sample["question"].tolist(), tokenizer, model, prompt)

    return pd.DataFrame({
        "id": sample["id"],
        "question": sample["question"],
        "expected": sample["answer"],
        "layer": layer,  # empty for the baseline
        "emotion": emotion,  # "none" for the baseline
        "mode": STEERING_MODE,  # meaning of alpha: multiple of act_norm ("fixed") or standard deviations ("std")
        "alpha": alpha,
        "prompt": prompt,  # to check that every condition used the same prompt
        "answer_text": texts,  # raw generated answer, to check the parsing
        "pred": [parse_answer(t) for t in texts],  # "yes", "no" or "unknown"
    })


def _save(dataset, new):
    """
    Add the answers of a condition to the answers file, replacing the old answers of the same condition.

    Args:
        dataset: "bbf" or "bbc"
        new: DataFrame with the answers of the condition
    """

    path = answers_path(dataset)
    if path.exists():
        old = pd.read_csv(path)

        # condition of the new answers (the same for every row)
        emotion = new["emotion"].iloc[0]
        layer = new["layer"].iloc[0]

        same_emotion = old["emotion"] == emotion

        # if the layer is NaN (baseline), check that the old layer is also NaN; otherwise, check equality
        if pd.isna(layer):
            same_layer = old["layer"].isna()
        else:
            same_layer = old["layer"] == layer

        # rows of the same condition: same emotion and same layer
        same = same_emotion & same_layer

        # keep the old answers of the other conditions and add the new ones
        if (~same).any():  # concatenating an empty table is deprecated in pandas
            new = pd.concat([old[~same], new], ignore_index=True)

    # save the answers to the file, creating the directory if needed
    path.parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(path, index=False)


def _strength(layer, emotion, alpha):
    """
    Steering strength of one condition, with the meaning of the active STEERING_MODE:
    --alpha if given, otherwise HALLUC_ALPHAS[layer][emotion] ("fixed") or STD_ALPHA ("std").

    Args:
        layer: layer to steer
        emotion: emotion to steer towards
        alpha: one strength for every condition, or None to use the default of the active STEERING_MODE

    Returns:
        steering strength of the condition
    """

    if alpha is not None:
        return alpha
    if STEERING_MODE == "std":
        return STD_ALPHA

    # "fixed": the alphas chosen by hand in free generation
    if layer not in HALLUC_ALPHAS:
        raise ValueError(f"no alphas for layer {layer} in HALLUC_ALPHAS: add them to the config or pass --alpha")
    if emotion not in HALLUC_ALPHAS[layer]:
        raise ValueError(f"no alpha for {emotion} at layer {layer} in HALLUC_ALPHAS: add it or pass --alpha")
    return HALLUC_ALPHAS[layer][emotion]


def _plan(layers, emotions, alpha, method):
    """
    Steering conditions to run: (layer, emotion, strength, saved vectors of the layer).
    Every check is done here, before any GPU work.

    Args:
        layers: layers to steer
        emotions: emotions to steer towards
        alpha: one strength for every condition, or None to use the default of the active STEERING_MODE
        method: method used to compute the vectors

    Returns:
        list of (layer, emotion, strength, saved) for every condition
    """

    # check that the requested emotions are available
    unknown = [e for e in emotions if e not in EMOTIONS]
    if unknown:
        raise ValueError(f"unknown emotions {unknown}, available: {EMOTIONS}")

    plan = []
    for layer in layers:
        # load the vectors once per layer and check that they have what the steering mode needs
        saved = load_vectors(layer, method)
        check_vectors(saved, layer)
        for emotion in emotions:
            plan.append((layer, emotion, _strength(layer, emotion, alpha), saved))
    return plan


def run_eval(dataset, layers, emotions, tokenizer, model, alpha=None, baseline_only=False, method=STEERING_METHOD):
    """
    Answer the fixed BeliefBank sample without steering (baseline) and with each steering condition,
    generating the answers and reading them as yes / no / unknown.
    The answers are saved after every condition to results/{model}/hallucination/{mode}/{dataset}_answers.csv;
    a condition that was already in the file is replaced, the others are kept.
    If the file was computed on another sample or with another prompt, it is replaced.

    Args:
        dataset: "bbf" or "bbc"
        layers: layers to steer
        emotions: emotions to steer towards
        tokenizer: the tokenizer for the model
        model: the model itself
        alpha: one strength for every condition, or None to use the default of the active STEERING_MODE
        baseline_only: only answer without steering (to check the prompt)
        method: method used to compute the vectors
    """

    sample = load_bb_sample(dataset)
    plan = [] if baseline_only else _plan(layers, emotions, alpha, method)

    # answers computed on another sample or with another prompt cannot be compared: start a new file
    path = answers_path(dataset)
    if path.exists():
        old = pd.read_csv(path)
        if set(old["id"]) != set(sample["id"]):
            print(f"{path.name} was computed on a different sample: it is replaced")
            path.unlink()
        elif "prompt" not in old or not old["prompt"].eq(BB_PROMPTS[dataset]).all():
            print(f"{path.name} was computed with a different prompt: it is replaced")
            path.unlink()

    # report the accuracy and rates of "yes" and "unknown" of a condition
    def report(df, name):
        acc = (df["pred"] == df["expected"]).mean()
        yes = (df["pred"] == "yes").mean()
        unknown = (df["pred"] == "unknown").mean()
        print(f"[{dataset} | {name}] accuracy={acc:.3f}  yes={yes:.3f}  unknown={unknown:.3f}")

    # the baseline does not depend on the layer: computed once, or again if asked for
    has_baseline = path.exists() and pd.read_csv(path)["emotion"].eq("none").any()
    if baseline_only or not has_baseline:
        base = _answer(sample, tokenizer, model, BB_PROMPTS[dataset], np.nan, "none", 0.0)
        _save(dataset, base)
        report(base, "baseline")

    # run every steering condition: steer the model, answer the questions and save the answers
    for layer, emotion, a, saved in plan:
        with steering(model, layer, saved, emotion, a):
            steered = _answer(sample, tokenizer, model, BB_PROMPTS[dataset], layer, emotion, a)
        _save(dataset, steered)
        report(steered, f"layer {layer} | {STEERING_MODE} | {emotion} | alpha={a}")

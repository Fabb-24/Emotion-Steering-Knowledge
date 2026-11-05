from typing import List, Optional
import typer

from config import HALLUC_LAYERS

app = typer.Typer(help="Compare the answers of the model on BeliefBank with and without steering.")

DATASETS = ["bbf", "bbc"]


@app.command()
def dataset():
    """
    Sample the fixed BeliefBank questions (50% yes, 50% no) into data/splits/beliefbank/.
    """

    from data_processing.beliefbank import build_bb_sample

    for name in DATASETS:
        sample = build_bb_sample(name)
        print(f"{name}: {len(sample)} questions, {(sample['answer'] == 'yes').mean():.0%} yes")


@app.command(name="eval")
def evaluate(
    dataset: Optional[List[str]] = typer.Option(None, help="BeliefBank dataset: bbf or bbc (repeatable, default: both)."),
    layer: Optional[List[int]] = typer.Option(None, help="Layer to steer (repeatable, default: HALLUC_LAYERS)."),
    emotion: Optional[List[str]] = typer.Option(None, help="Emotion to steer towards (repeatable, default: all)."),
    alpha: Optional[float] = typer.Option(None, help="One strength for every condition, with the meaning of STEERING_MODE (default: from the config)."),
    baseline_only: bool = typer.Option(False, "--baseline-only", help="Only answer without steering (to check the prompt)."),
):
    """
    Answer the BeliefBank questions without steering and with each steering condition.
    Answers go to results/{model}/hallucination/{mode}/{dataset}_answers.csv.
    """

    from config import EMOTIONS
    from hallucination.evaluate import run_eval
    from models.llm import load_model

    tokenizer, model = load_model()
    for name in dataset or DATASETS:
        run_eval(name, layer or HALLUC_LAYERS, emotion or EMOTIONS, tokenizer, model,
                 alpha=alpha, baseline_only=baseline_only)


@app.command()
def analyze(
    dataset: Optional[List[str]] = typer.Option(None, help="BeliefBank dataset: bbf or bbc (repeatable, default: both)."),
):
    """
    Accuracy, flip rate and rate of "yes" of every condition against the baseline, with plots.
    Results go to results/{model}/hallucination/{mode}/.
    """

    from hallucination.analyze import main

    main(dataset or DATASETS)


@app.command()
def accuracy(
    dataset: Optional[List[str]] = typer.Option(None, help="BeliefBank dataset: bbf or bbc (repeatable, default: both)."),
    hallucination: bool = typer.Option(False, "--hallucination", help="Plot the hallucination rate (1 - accuracy) instead."),
):
    """
    Accuracy (or hallucination rate) bar charts of every model in one figure per dataset.
    Results go to results/general/.
    """

    from hallucination.compare import plot_accuracy_all

    for name in dataset or DATASETS:
        plot_accuracy_all(name, hallucination)


@app.command(name="yes-rate")
def yes_rate(
    dataset: Optional[List[str]] = typer.Option(None, help="BeliefBank dataset: bbf or bbc (repeatable, default: both)."),
):
    """
    Heatmap of the yes rate of every emotion at every model and layer, one per dataset.
    Results go to results/general/.
    """

    from hallucination.compare import plot_yes_rate

    for name in dataset or DATASETS:
        plot_yes_rate(name)

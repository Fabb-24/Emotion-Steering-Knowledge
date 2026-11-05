from typing import List, Optional
import typer

from config import STEERING_METHOD

app = typer.Typer(help="Compute and apply emotion steering vectors.")


def _layers(layer, all_layers):
    """
    Layers chosen on the command line: --layer N (repeatable) or --all.
    """

    from data_processing.activations import n_layers

    layers = list(range(n_layers())) if all_layers else layer
    if not layers:
        raise typer.BadParameter("pass --layer N (one or more times) or --all")
    return layers


@app.command()
def dataset():
    """
    Sample STEERING_N_TRAIN sentences per emotion to compute the vectors and STEERING_N_TEST for the checks.
    """

    from config import STEERING_N_TEST, STEERING_N_TRAIN, STEERING_SPLITS_DIR
    from data_processing.splits import build_splits

    train, test = build_splits(STEERING_N_TRAIN, STEERING_N_TEST, STEERING_SPLITS_DIR)
    print(f"train: {len(train)} rows, test: {len(test)} rows -> {STEERING_SPLITS_DIR}")


@app.command()
def vectors(
    layer: Optional[List[int]] = typer.Option(None, help="Layer to compute (repeatable)."),
    all_layers: bool = typer.Option(False, "--all", help="Compute every layer."),
    method: str = typer.Option(STEERING_METHOD, help="Method used to compute the vectors."),
):
    """
    Compute the steering vectors of the given layers and save them to data/steering/{model}/{method}/.
    """

    from steering.vectors import compute_vectors, vectors_path

    for l in _layers(layer, all_layers):
        result = compute_vectors(l, method)
        norms = result["vectors"].norm(dim=-1)
        print(f"layer {l:2d}  act_norm={result['act_norm']:7.1f}  mean ||v||={norms.mean():6.2f}  -> {vectors_path(l, method)}")


@app.command()
def check(
    layer: Optional[List[int]] = typer.Option(None, help="Layer to check (repeatable)."),
    all_layers: bool = typer.Option(False, "--all", help="Check every layer."),
    method: str = typer.Option(STEERING_METHOD, help="Method used to compute the vectors."),
):
    """
    Cosine similarity between the vectors and classification by projection on unseen sentences.
    Results go to results/{model}/steering/{method}/.
    """

    from steering.checks import cosine_check, projection_check

    for l in _layers(layer, all_layers):
        cosine_check(l, method)
        projection_check(l, method)
        print()


@app.command()
def generate(
    layer: List[int] = typer.Option(..., help="Layer to steer (repeatable)."),
    emotion: Optional[List[str]] = typer.Option(None, help="Emotion to steer towards (repeatable, default: all)."),
    alpha: Optional[List[float]] = typer.Option(None, help="Steering strength with the meaning of STEERING_MODE (repeatable, default: STEERING_ALPHAS)."),
    method: str = typer.Option(STEERING_METHOD, help="Method used to compute the vectors."),
):
    """
    Free generation on neutral prompts with each steering vector (needs the GPU).
    Texts go to results/{model}/steering/{method}/{mode}/generation_layer{layer}.csv.
    """

    from config import EMOTIONS, STEERING_ALPHAS
    from models.llm import load_model
    from steering.generate import run_generation

    tokenizer, model = load_model()
    for l in layer:
        run_generation(l, tokenizer, model, emotion or EMOTIONS, alpha or STEERING_ALPHAS, method)


@app.command(name="all")
def run_all(
    layer: Optional[List[int]] = typer.Option(None, help="Layer to compute and check (repeatable)."),
    all_layers: bool = typer.Option(False, "--all", help="Compute and check every layer."),
    method: str = typer.Option(STEERING_METHOD, help="Method used to compute the vectors."),
):
    """
    Run dataset -> vectors -> check on the given layers. Generation is left out (GPU, needs choices).
    """

    _layers(layer, all_layers)
    dataset()
    vectors(layer=layer, all_layers=all_layers, method=method)
    check(layer=layer, all_layers=all_layers, method=method)

import typer

from hallucination.cli import app as halluc_app
from layer_selection.cli import app as layer_app
from steering.cli import app as steer_app

app = typer.Typer(help="Emotion steering pipeline.")
app.add_typer(layer_app, name="layer")
app.add_typer(steer_app, name="steer")
app.add_typer(halluc_app, name="halluc")


@app.command()
def download():
    """
    Download EmpatheticDialogues and save one CSV per emotion into data/emotions.
    """

    from data_processing.download import Emotions
    Emotions().save_emotions()


@app.command()
def extract():
    """
    Extract the last-token residual stream of every layer for every sentence.
    """

    from data_processing.extract import extract_all
    extract_all()


@app.command()
def stats():
    """
    Print statistics of the emotion data, of the splits and of BeliefBank.
    """

    from data_processing.stats import main
    main()


if __name__ == "__main__":
    app()

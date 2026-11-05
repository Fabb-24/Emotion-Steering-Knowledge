from typing import Optional
import typer

app = typer.Typer(help="Choose the steering layer with linear probes.")


@app.command()
def dataset():
    """
    Sample the train/test splits from data/emotions into data/splits/layer_selection/.
    """

    from layer_selection import dataset as ds
    ds.main()


@app.command()
def probing():
    """
    Train one linear softmax probe per layer and save metrics + predictions.
    """

    from layer_selection import probing as prb
    prb.run_probing()


@app.command()
def analyze():
    """
    Best layer by macro F1, layer_f1.json, macro F1 plot and confusion matrix.
    """

    from layer_selection import analyze as anl
    anl.main()


@app.command()
def confusion(
    layer: Optional[int] = typer.Option(None, help="Layer to inspect (default: highest macro F1)."),
):
    """
    Plot the confusion matrix at a given layer.
    """

    from layer_selection import analyze as anl
    anl.confusion_at_layer(layer)


@app.command()
def compare():
    """
    Plot the macro F1 of every model against the relative depth of the layer, with the steering layers marked.
    """

    from layer_selection import compare as cmp
    cmp.plot_comparison()


@app.command(name="all")
def run_all():
    """
    Run dataset -> probing -> analyze (needs `extract` to have been run once).
    """

    dataset()
    probing()
    analyze()

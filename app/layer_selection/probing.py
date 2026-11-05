import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from config import (EMOTIONS, LAYER_RESULTS_DIR, LAYER_SPLITS_DIR, PROBE_EPOCHS, PROBE_L2, PROBE_LR)
from data_processing.activations import load_acts
from data_processing.splits import load_split
from models.linear_probe import DEVICE, standardize, train_probe


def run_probing():
    """
    Train one linear softmax probe per layer and save metrics + predictions.

    Returns:
        df: DataFrame with the probing metrics (layer, emotion, f1, accuracy, macro_f1)
    """

    # Load activations and labels for train and test splits
    X_train, y_train = load_acts(load_split("train", LAYER_SPLITS_DIR)["id"].tolist())
    X_test, y_test = load_acts(load_split("test", LAYER_SPLITS_DIR)["id"].tolist())
    y_train = y_train.to(DEVICE)
    y_test = y_test.numpy()

    rows, pred_rows = [], []
    # Train one probe per layer and compute metrics
    for layer in range(X_train.shape[0]):
        x_train, x_test = standardize(X_train[layer].to(DEVICE), X_test[layer].to(DEVICE))

        probe = train_probe(x_train, y_train, len(EMOTIONS), PROBE_EPOCHS, PROBE_LR, PROBE_L2)
        pred = probe.predict(x_test).cpu().numpy()

        f1s = f1_score(y_test, pred, labels=range(len(EMOTIONS)), average=None)
        acc = accuracy_score(y_test, pred)
        for k, emotion in enumerate(EMOTIONS):
            rows.append({"layer": layer, "emotion": emotion, "f1": f1s[k], "accuracy": acc, "macro_f1": f1s.mean()})
        for sample, (true, p) in enumerate(zip(y_test, pred)):
            pred_rows.append({"layer": layer, "sample": sample, "label": EMOTIONS[true], "pred": EMOTIONS[p]})
        print(f"layer {layer:2d}  acc={acc:.3f}  macro_f1={f1s.mean():.3f}")

    # Save metrics and predictions to CSV files
    LAYER_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(LAYER_RESULTS_DIR / "probing_metrics.csv", index=False)
    pd.DataFrame(pred_rows).to_csv(LAYER_RESULTS_DIR / "probing_predictions.csv", index=False)
    return df


if __name__ == "__main__":
    run_probing()

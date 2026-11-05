import pandas as pd

from config import BB_SPLITS_DIR, EMOTIONS, EMOTIONS_DIR, LAYER_SPLITS_DIR, MIN_WORDS, STEERING_SPLITS_DIR
from data_processing.beliefbank import load_beliefbank

# subject of a BeliefBank statement: "a frog is a salad." -> "frog", "If an ape is a house, then ..." -> "ape"
BBF_ENTITY = r"^(?:an?|the)\s+(.+?)\s+(?:is|are|can|cannot|has|have|does|do)\b"
BBC_ENTITY = r"^If (?:an?|the)\s+(.+?)\s+(?:is|are)\b"
NEGATION = r"\b(?:not|cannot)\b"


def _words(texts):
    """
    Number of words of every text.

    Args:
        texts: Series of strings

    Returns:
        Series of ints
    """

    return texts.str.split().str.len()


def _length_row(texts):
    """
    Summary of the length in words of a set of texts.

    Args:
        texts: Series of strings

    Returns:
        dict with mean, median, min and max length
    """

    n = _words(texts)
    return {"words_mean": round(n.mean(), 1), "words_median": n.median(), "words_min": n.min(), "words_max": n.max()}


def emotion_stats():
    """
    Rows of every emotion before and after each cleaning step of load_emotion, and their length in words.

    Returns:
        DataFrame with one row per emotion and a total row
    """

    rows = []
    for emotion in EMOTIONS:
        # same steps as data_processing.splits.load_emotion, counting the rows removed by each one
        df = pd.read_csv(EMOTIONS_DIR / f"{emotion}.csv")
        raw = len(df)
        df = df.dropna(subset=["situation"])
        missing = raw - len(df)
        df["situation"] = df["situation"].str.strip()
        n = len(df)
        df = df.drop_duplicates(subset="situation")
        duplicates = n - len(df)
        n = len(df)
        df = df[_words(df["situation"]) >= MIN_WORDS]
        short = n - len(df)

        rows.append({"emotion": emotion, "raw": raw, "missing": missing, "duplicates": duplicates,
                     "short": short, "clean": len(df), **_length_row(df["situation"])})

    table = pd.DataFrame(rows)
    total = table[["raw", "missing", "duplicates", "short", "clean"]].sum()
    return pd.concat([table, pd.DataFrame([{"emotion": "total", **total}])], ignore_index=True)


def split_stats():
    """
    Size of the layer selection and steering splits and how many sentences they share.

    Returns:
        sizes: DataFrame with the rows per emotion of every split
        overlap: DataFrame with the number of shared sentences between every pair of splits
    """

    splits = {}
    for phase, folder in [("layer", LAYER_SPLITS_DIR), ("steer", STEERING_SPLITS_DIR)]:
        for name in ["train", "test"]:
            path = folder / f"{name}.csv"
            if path.exists():
                splits[f"{phase}_{name}"] = pd.read_csv(path)

    sizes = pd.DataFrame({name: df["emotion"].value_counts() for name, df in splits.items()}).reindex(EMOTIONS)
    ids = {name: set(df["id"]) for name, df in splits.items()}
    overlap = pd.DataFrame({a: {b: len(ids[a] & ids[b]) for b in ids} for a in ids})
    return sizes, overlap


def beliefbank_stats(name):
    """
    Composition of a BeliefBank dataset and of its fixed sample.

    Args:
        name: "bbf" or "bbc"

    Returns:
        DataFrame with one row for the whole dataset and one for the sample (if it exists)
    """

    entity = BBF_ENTITY if name == "bbf" else BBC_ENTITY
    data = {"full": load_beliefbank(name)}
    if (BB_SPLITS_DIR / f"{name}.csv").exists():
        data["sample"] = pd.read_csv(BB_SPLITS_DIR / f"{name}.csv")

    rows = []
    for part, df in data.items():
        q = df["question"]
        rows.append({
            "dataset": name, "part": part, "rows": len(df),
            "yes": int((df["answer"] == "yes").sum()), "no": int((df["answer"] == "no").sum()),
            "duplicates": int(q.duplicated().sum()),
            "negated": int(q.str.contains(NEGATION).sum()),
            "entities": q.str.extract(entity)[0].nunique(),
            **_length_row(q),
        })
    return pd.DataFrame(rows)


def top_entities(name, n=10):
    """
    Most frequent entities of a BeliefBank dataset.

    Args:
        name: "bbf" or "bbc"
        n: number of entities to show

    Returns:
        Series with the number of questions of each entity
    """

    entity = BBF_ENTITY if name == "bbf" else BBC_ENTITY
    return load_beliefbank(name)["question"].str.extract(entity)[0].value_counts().head(n)


def main():
    """
    Print the statistics of the emotion data, of the splits and of BeliefBank.
    """

    pd.set_option("display.width", 200)

    print("=== EmpatheticDialogues (data/emotions) ===")
    print(emotion_stats().to_string(index=False))

    sizes, overlap = split_stats()
    print("\n=== Splits: rows per emotion ===")
    print(sizes.to_string())
    print("\n=== Splits: shared sentences ===")
    print(overlap.to_string())

    print("\n=== BeliefBank ===")
    print(pd.concat([beliefbank_stats(name) for name in ["bbf", "bbc"]]).to_string(index=False))
    for name in ["bbf", "bbc"]:
        print(f"\n{name}: most frequent entities")
        print(top_entities(name).to_string())


if __name__ == "__main__":
    main()

import pandas as pd

from config import BB_N_SAMPLES, BB_SPLITS_DIR, BELIEFBANK_DIR, SEED

BB_FILES = {"bbf": "BBF.csv", "bbc": "BBC.csv"}


def load_beliefbank(name):
    """
    Load a BeliefBank dataset. Each question gets an id "<name>-<row>", where row is its row number in the CSV.

    Args:
        name: "bbf" (factual) or "bbc" (logical)

    Returns:
        DataFrame with columns question, answer, type, id
    """

    if name not in BB_FILES:
        raise ValueError(f"unknown BeliefBank dataset {name!r}, available: {list(BB_FILES)}")
    df = pd.read_csv(BELIEFBANK_DIR / BB_FILES[name])
    df["id"] = name + "-" + df.index.astype(str)
    return df


def build_bb_sample(name, n=BB_N_SAMPLES, seed=SEED):
    """
    Sample n questions with 50% yes and 50% no answers and save them to data/splits/beliefbank/{name}.csv.
    BBF: n/2 yes and n/2 no questions. BBC: n/2 whole pairs (a statement and its negation, opposite answers),
    so that the consistency of the answers can be checked later.

    Args:
        name: "bbf" or "bbc"
        n: number of questions
        seed: random seed for the sampling

    Returns:
        DataFrame with the sample
    """

    df = load_beliefbank(name)

    if name == "bbf":
        df = df.drop_duplicates(subset="question")
        sample = pd.concat([group.sample(n=n // 2, random_state=seed) for _, group in df.groupby("answer")])
    else:
        # consecutive rows are a statement and its negation, with opposite answers
        df["pair"] = df.index // 2
        pairs = df.groupby("pair")["answer"]
        if not (pairs.size().eq(2) & pairs.nunique().eq(2)).all():
            raise ValueError("BBC rows are not pairs of opposite answers")

        # sample n/2 pairs, which gives n questions in total
        chosen = pd.Series(df["pair"].unique()).sample(n=n // 2, random_state=seed)
        sample = df[df["pair"].isin(chosen)]

    # sort by id to have a consistent order for the same seed and save to CSV
    sample = sample.sort_index()
    BB_SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    sample.to_csv(BB_SPLITS_DIR / f"{name}.csv", index=False)
    return sample


def load_bb_sample(name):
    """
    Load the fixed sample built by `halluc dataset`.

    Args:
        name: "bbf" or "bbc"

    Returns:
        DataFrame with the sample
    """

    path = BB_SPLITS_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run `python app/cli.py halluc dataset` first")
    return pd.read_csv(path)

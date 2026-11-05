import pandas as pd

from config import EMOTIONS, EMOTIONS_DIR, MIN_WORDS, SEED


def load_emotion(emotion):
    """
    Load the sentences of one emotion from data/emotions and clean them.
    Each sentence gets an id "<emotion>-<row>", where row is its row number in the emotion CSV.

    Args:
        emotion: name of the emotion

    Returns:
        DataFrame with columns situation, emotion, id
    """

    # Load the CSV file for the emotion and create an id column
    df = pd.read_csv(EMOTIONS_DIR / f"{emotion}.csv")
    df["id"] = emotion + "-" + df.index.astype(str)

    # Clean the sentences: remove rows with missing situations, strip whitespace, and remove duplicates
    df = df.dropna(subset=["situation"])
    df["situation"] = df["situation"].str.strip()
    df = df.drop_duplicates(subset="situation")

    # Remove situations that are too short to describe anything (e.g. "1")
    df = df[df["situation"].str.split().str.len() >= MIN_WORDS]

    return df


def load_all():
    """
    Load the cleaned sentences of every emotion.

    Returns:
        DataFrame with columns situation, emotion, id
    """

    return pd.concat([load_emotion(e) for e in EMOTIONS], ignore_index=True)


def build_splits(n_train, n_test, folder, seed=SEED):
    """
    Build train/test splits for each emotion and save them to folder/train.csv and folder/test.csv.
    Each split has n_train/n_test rows per emotion, sampled randomly with a fixed seed.

    Args:
        n_train: rows per emotion in the train split
        n_test: rows per emotion in the test split
        folder: directory where the two CSV files are saved
        seed: random seed for the sampling

    Returns:
        train: DataFrame with the train split
        test: DataFrame with the test split
    """

    train_parts, test_parts = [], []

    # Sample n_train + n_test rows for each emotion and split them into train/test
    for emotion in EMOTIONS:
        df = load_emotion(emotion)
        if n_train + n_test > len(df):
            raise ValueError(f"{emotion} has {len(df)} rows, but n_train + n_test = {n_train + n_test}")
        sampled = df.sample(n=n_train + n_test, random_state=seed)
        train_parts.append(sampled.iloc[:n_train])
        test_parts.append(sampled.iloc[n_train:])

    train = pd.concat(train_parts, ignore_index=True)
    test = pd.concat(test_parts, ignore_index=True)

    # Save the splits to CSV files
    folder.mkdir(parents=True, exist_ok=True)
    train.to_csv(folder / "train.csv", index=False)
    test.to_csv(folder / "test.csv", index=False)
    return train, test


def load_split(name, folder):
    """
    Load a split (train/test) from CSV.

    Args:
        name: "train" or "test"
        folder: directory of the split (e.g. LAYER_SPLITS_DIR or STEERING_SPLITS_DIR)

    Returns:
        DataFrame with the split
    """
    return pd.read_csv(folder / f"{name}.csv")

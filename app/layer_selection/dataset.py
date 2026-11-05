from config import LAYER_SPLITS_DIR, N_TEST, N_TRAIN
from data_processing.splits import build_splits


def main():
    train, test = build_splits(N_TRAIN, N_TEST, LAYER_SPLITS_DIR)
    print(f"train: {len(train)} rows, test: {len(test)} rows")


if __name__ == "__main__":
    main()

import pandas as pd

from config import EMOTIONS, EMOTIONS_DIR


class Emotions:
    def __init__(self):
        self.emotions = EMOTIONS

    def _download_data(self):
        splits = {
            'train': 'train.csv',
            'validation': 'valid.csv',
            'test': 'test.csv'
        }

        base_url = "hf://datasets/bdotloh/empathetic-dialogues-contexts/"

        df_train = pd.read_csv(base_url + splits['train'])
        df_valid = pd.read_csv(base_url + splits['validation'])
        df_test = pd.read_csv(base_url + splits['test'])

        return df_train, df_valid, df_test

    def save_emotions(self):
        train, valid, test = self._download_data()

        unified_emotions = pd.concat(
            [train, valid, test],
            axis=0,
            ignore_index=True
        )

        EMOTIONS_DIR.mkdir(parents=True, exist_ok=True)
        min_emotion_count = unified_emotions['emotion'].value_counts().min()
        for emotion in self.emotions:
            emotion_df = unified_emotions[
                unified_emotions['emotion'] == emotion
            ]
            emotion_df = emotion_df.drop(columns=emotion_df.columns[0])

            #save the first min_emotion_count rows of the emotion_df to a csv file
            emotion_df.head(min_emotion_count).to_csv(
                EMOTIONS_DIR / f"{emotion}.csv",
                index=False
            )


if __name__ == "__main__":
    Emotions().save_emotions()

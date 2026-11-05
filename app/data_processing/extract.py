import torch

from config import ACTIVATIONS_FILE, EMOTIONS, MODEL_ID, PROMPT_TEMPLATE
from data_processing.splits import load_all
from models.hidden_states import extract_last_token
from models.llm import load_model


def extract_all():
    """
    Extract the last-token residual stream of every layer for every sentence of every emotion
    and save it to data/activations/{model}/acts_all.pt.
    """

    df = load_all()
    tokenizer, model = load_model()
    acts = extract_last_token(df["situation"].tolist(), tokenizer, model, prompt_template=PROMPT_TEMPLATE)

    ACTIVATIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "acts": acts.half(),
            "ids": df["id"].tolist(),
            "labels": torch.tensor([EMOTIONS.index(e) for e in df["emotion"]]),
            "emotions": EMOTIONS,
            "model": MODEL_ID,
            "prompt_template": PROMPT_TEMPLATE,
        },
        ACTIVATIONS_FILE,
    )
    print(f"saved {tuple(acts.shape)} to {ACTIVATIONS_FILE}")


if __name__ == "__main__":
    extract_all()

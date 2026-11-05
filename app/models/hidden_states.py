import torch
from tqdm import tqdm

from config import BATCH_SIZE, MAX_LENGTH


@torch.no_grad()
def extract_last_token(texts, tokenizer, model, prompt_template="{text}", batch_size=BATCH_SIZE, max_length=MAX_LENGTH):
    """
    Extract the last-token residual stream of every layer for a list of texts.

    Args:
        texts: list of strings to process
        tokenizer: the tokenizer for the model
        model: the model itself
        prompt_template: template applied to each text; {text} = the text
        batch_size: number of texts per forward pass
        max_length: max number of tokens per text (longer texts are truncated)

    Returns:
        Tensor of shape [L+1, N, d], where L is the number of layers, N is the number of texts, and d is the hidden size.
    """

    feats = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = [prompt_template.format(text=t) for t in texts[i:i + batch_size]]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(model.device)
        out = model(**enc, output_hidden_states=True)
        # out.hidden_states: tuple of [B, T, d] for each layer (B=batch size, T=sequence length, d=hidden size)
        last = torch.stack([h[:, -1, :] for h in out.hidden_states])  # [L+1, B, d]
        feats.append(last.float().cpu())
    return torch.cat(feats, dim=1)  # [L+1, N, d]

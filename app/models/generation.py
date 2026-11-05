import torch
from tqdm import tqdm

from config import BATCH_SIZE, GEN_MAX_NEW_TOKENS


@torch.no_grad()
def generate(prompts, tokenizer, model, max_new_tokens=GEN_MAX_NEW_TOKENS, batch_size=BATCH_SIZE):
    """
    Answer a list of prompts with the chat template of the model, using greedy decoding
    (deterministic: with the same prompt, only the steering changes the output).

    Args:
        prompts: list of user messages
        tokenizer: the tokenizer for the model (left padding)
        model: the model itself
        max_new_tokens: max number of generated tokens per answer
        batch_size: number of prompts generated together

    Returns:
        list of generated answers (prompt excluded)
    """

    answers = []
    for i in tqdm(range(0, len(prompts), batch_size)):
        # Apply the chat template to each prompt and tokenize the batch
        chats = [tokenizer.apply_chat_template(
                    [{"role": "user", "content": p}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                ) for p in prompts[i:i + batch_size]]

        # the chat template already contains the special start tokens (e.g. <|begin_of_text|> of Llama): must not add them a second time
        enc = tokenizer(chats, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)

        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            pad_token_id=tokenizer.pad_token_id,
        )

        # Decode the generated tokens to text, skipping the prompt tokens
        new_tokens = out[:, enc["input_ids"].shape[1]:]
        answers.extend(tokenizer.batch_decode(new_tokens, skip_special_tokens=True))

    return answers

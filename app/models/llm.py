import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from config import LOAD_IN_4BIT, MODEL_ID


def load_model(model_id=MODEL_ID, load_in_4bit=LOAD_IN_4BIT):
    """
    Load the active model (MODEL_ID in the config) and its tokenizer, with optional 4-bit quantization.

    Args:
        model_id: Hugging Face model id
        load_in_4bit: whether to load the weights in 4-bit (NF4)

    Returns:
        tokenizer: the tokenizer for the model
        model: the model itself
    """

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "left"

    # Llama has no padding token: pad with the end-of-text token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = None
    if load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    return tokenizer, model

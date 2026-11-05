import re

from config import BATCH_SIZE, BB_MAX_NEW_TOKENS
from models.generation import generate

NEGATIVE_PATTERN = r"\b(not true|isn't true|false|incorrect|untrue|wrong)\b"
POSITIVE_PATTERN = r"\b(true|correct|right)\b"


def ask_yes_no(statements, tokenizer, model, prompt, max_new_tokens=BB_MAX_NEW_TOKENS, batch_size=BATCH_SIZE):
    """
    Ask the model whether each statement is true and generate its answer.

    Args:
        statements: list of statements
        tokenizer: the tokenizer for the model (left padding)
        model: the model itself
        prompt: question template; {statement} = the statement
        max_new_tokens: max length of each answer
        batch_size: number of questions generated together

    Returns:
        list of generated answers (raw text)
    """

    questions = [prompt.format(statement=s) for s in statements]
    return generate(questions, tokenizer, model, max_new_tokens=max_new_tokens, batch_size=batch_size)


def parse_answer(text):
    """
    Read a generated answer as "yes", "no" or "unknown" with string operations:
    1. the first word is "yes" or "no";
    2. otherwise, only one of the words "yes" / "no" appears in the first sentence;
    3. otherwise, the first sentence says that the statement is false / true;
    4. otherwise the answer cannot be interpreted.

    Args:
        text: generated answer

    Returns:
        "yes", "no" or "unknown"
    """

    # Qwen3.8 can reason before answering even with thinking disabled: read only the answer after the reasoning
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    elif text.lstrip().lower().startswith("the user"):
        return "unknown"

    # lowercase and remove markdown and quotes ("**Yes**", '"No"')
    text = re.sub(r"[*_`#>\"“”‘’]", " ", text.lower()).strip()

    # first word
    words = re.findall(r"[a-z']+", text)
    if words and words[0] in ("yes", "no"):
        return words[0]

    # whole words "yes" / "no" in the first sentence (\b: "not" and "know" are not "no")
    first = re.split(r"[.!?\n]", text, maxsplit=1)[0]
    has_yes = re.search(r"\byes\b", first) is not None
    has_no = re.search(r"\bno\b", first) is not None
    if has_yes != has_no:
        return "yes" if has_yes else "no"

    # "false", "not true", ... before "true", "correct", ...
    if re.search(NEGATIVE_PATTERN, first):
        return "no"
    if re.search(POSITIVE_PATTERN, first):
        return "yes"

    return "unknown"

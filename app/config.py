import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Active model and steering mode (change these lines to switch)
# ---------------------------------------------------------------------------
MODEL_NAME = "llama"
MODELS = {
    "qwen": "Qwen/Qwen2.5-7B-Instruct",
    "llama": "meta-llama/Llama-3.1-8B-Instruct",  # gated
    "qwen38": "Qwen/Qwen3.8-27B",  # run it on Modal
}
STEERING_MODE = "fixed"  # "fixed" or "std"

# ---------------------------------------------------------------------------
# Paths (data and splits are shared, the outputs of the model have one directory per model)
# ---------------------------------------------------------------------------
ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parent.parent))  # root of data/ and results/: the project folder, or PROJECT_ROOT if set (e.g. the Modal volume)
DATA_DIR = ROOT / "data"
EMOTIONS_DIR = DATA_DIR / "emotions"  # directory containing the raw emotion data
SPLITS_DIR = DATA_DIR / "splits"
LAYER_SPLITS_DIR = SPLITS_DIR / "layer_selection"  # train/test split of the layer selection phase
STEERING_SPLITS_DIR = SPLITS_DIR / "steering"  # train/test split of the steering phase
BB_SPLITS_DIR = SPLITS_DIR / "beliefbank"  # fixed BeliefBank samples of the hallucination phase
BELIEFBANK_DIR = DATA_DIR / "beliefbank"  # original BBF.csv, BBC.csv
ACTIVATIONS_DIR = DATA_DIR / "activations" / MODEL_NAME  # directory containing the activations of the model for each sentence of each emotion
ACTIVATIONS_FILE = ACTIVATIONS_DIR / "acts_all.pt"  # every sentence of every emotion
STEERING_DIR = DATA_DIR / "steering" / MODEL_NAME  # steering vectors, one file per method and layer
RESULTS_DIR = ROOT / "results" / MODEL_NAME
LAYER_RESULTS_DIR = RESULTS_DIR / "layer_selection"  # metrics and plots of the layer selection phase
STEERING_RESULTS_DIR = RESULTS_DIR / "steering"  # plots and checks of the steering vectors (free generation in a subfolder per mode)
HALLUC_RESULTS_DIR = RESULTS_DIR / "hallucination" / STEERING_MODE  # answers, metrics and plots of the hallucination phase

# ---------------------------------------------------------------------------
# General
# ---------------------------------------------------------------------------
EMOTIONS = [
    'afraid', 'joyful', 'sad', 'annoyed', 'ashamed',
    'angry', 'trusting', 'anxious', 'content', 'devastated',
]

MIN_WORDS = 2  # situations with fewer words are dropped

SEED = 42

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
MODEL_ID = MODELS[MODEL_NAME]
LOAD_IN_4BIT_BY_MODEL = {
    "qwen": True,
    "llama": True,
    "qwen38": False,  # on Modal
}
LOAD_IN_4BIT = LOAD_IN_4BIT_BY_MODEL[MODEL_NAME]

BATCH_SIZE_BY_MODEL = {
    "qwen": 8,
    "llama": 8,
    "qwen38": 16,  # on Modal
}
BATCH_SIZE = BATCH_SIZE_BY_MODEL[MODEL_NAME]

MAX_LENGTH = 256  # max number of tokens in the input (the last token is the one whose residual stream is extracted)
PROMPT_TEMPLATE = "{text}"  # text fed to the model

# ---------------------------------------------------------------------------
# Layer selection
# ---------------------------------------------------------------------------
N_TRAIN = 50   # rows per emotion in the train split
N_TEST = 50    # rows per emotion in the test split

# Linear probe
PROBE_EPOCHS = 1000  # number of epochs to train the linear probe
PROBE_LR = 1e-3  # learning rate for the Adam optimizer
PROBE_L2 = 0.01  # weight of the L2 penalty on W added to the loss

# ---------------------------------------------------------------------------
# Steering
# ---------------------------------------------------------------------------
STEERING_METHOD = "diff_means"  # how steering vectors are computed
STEERING_N_TRAIN = 350  # rows per emotion used to compute the vectors
STEERING_N_TEST = 100  # rows per emotion held out for the checks

STD_ALPHA_BY_MODEL = {
    "qwen": 8.0,
    "llama": 8.0,
    "qwen38": 8.0,
}
STD_ALPHA = STD_ALPHA_BY_MODEL[MODEL_NAME]

# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
GEN_PROMPTS = [
    "Write a short story about a man walking home from work.",
    "Describe a rainy afternoon in the city.",
    "Continue the story: When I opened the door, I",
    "Write a short story about a woman waiting at a train station.",
]
GEN_MAX_NEW_TOKENS = 100  # max length of the generated text

STEERING_ALPHAS_BY_MODE = {  # strengths tried by `steer generate`, with the meaning of the active STEERING_MODE (0 = baseline)
    "fixed": [0.0, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8],
    "std": [0.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0],
}
STEERING_ALPHAS = STEERING_ALPHAS_BY_MODE[STEERING_MODE]

# ---------------------------------------------------------------------------
# Hallucination
# ---------------------------------------------------------------------------
BB_N_SAMPLES = 1000  # questions per BeliefBank dataset (50% yes, 50% no)
BB_MAX_NEW_TOKENS = 50  # max length of the generated answers

BB_PROMPTS = {  # prompt templates for the BeliefBank datasets, with {statement} replaced by the statement to evaluate
    "bbf": """You'll be provided with a fact, and your task is to confirm or deny its truthfulness.
        Just answer only with yes or no.
        Is the fact true? Fact: {statement}
        Answer:""",
    "bbc": """You'll be provided with a fact, and your task is to confirm or deny its truthfulness.
        Just answer only with yes or no.
        Is the fact true? Fact: {statement}
        Answer:""",
}

HALLUC_LAYERS_BY_MODEL = {  # layers used for the hallucination phase
    "qwen": [14, 20, 25],
    "llama": [17, 24, 27],
    "qwen38": [27, 39, 50, 54],
}
HALLUC_LAYERS = HALLUC_LAYERS_BY_MODEL[MODEL_NAME]

HALLUC_ALPHAS_BY_MODEL = {  # strengths used for the hallucination phase, with the meaning of the active STEERING_MODE
    "qwen": {
        14: {"afraid": 0.7, "angry": 0.7, "annoyed": 0.6, "anxious": 0.4, "ashamed": 0.8,
             "content": 1.0, "devastated": 0.4, "joyful": 0.8, "sad": 0.3, "trusting": 0.9},
        20: {"afraid": 0.8, "angry": 0.7, "annoyed": 0.6, "anxious": 0.8, "ashamed": 0.8,
             "content": 0.9, "devastated": 0.8, "joyful": 0.8, "sad": 0.6, "trusting": 0.8},
        25: {"afraid": 0.8, "angry": 0.8, "annoyed": 0.9, "anxious": 0.8, "ashamed": 0.7,
             "content": 0.7, "devastated": 0.7, "joyful": 0.8, "sad": 0.6, "trusting": 0.7},
    },
    "llama": {
        17: {"afraid": 0.6, "angry": 0.6, "annoyed": 0.6, "anxious": 0.6, "ashamed": 0.6,
             "content": 0.6, "devastated": 0.6, "joyful": 0.6, "sad": 0.7, "trusting": 0.6},
        24: {"afraid": 0.9, "angry": 0.6, "annoyed": 0.8, "anxious": 0.9, "ashamed": 0.7,
             "content": 0.9, "devastated": 0.8, "joyful": 0.6, "sad": 0.8, "trusting": 0.7},
        27: {"afraid": 1.0, "angry": 0.8, "annoyed": 0.8, "anxious": 0.8, "ashamed": 0.7,
             "content": 0.9, "devastated": 0.8, "joyful": 0.7, "sad": 0.9, "trusting": 0.6},
    },
    "qwen38": {
        27: {"afraid": 0.4, "angry": 0.5, "annoyed": 0.7, "anxious": 0.4, "ashamed": 0.5,
             "content": 0.5, "devastated": 0.5, "joyful": 0.4, "sad": 0.4, "trusting": 0.6},
        39: {"afraid": 0.6, "angry": 0.7, "annoyed": 0.7, "anxious": 0.6, "ashamed": 0.9,
             "content": 0.6, "devastated": 0.6, "joyful": 0.5, "sad": 0.6, "trusting": 0.7},
        50: {"afraid": 0.8, "angry": 0.6, "annoyed": 0.6, "anxious": 0.5, "ashamed": 0.7,
             "content": 0.6, "devastated": 0.6, "joyful": 0.9, "sad": 0.5, "trusting": 0.5},
        54: {"afraid": 1.0, "angry": 0.9, "annoyed": 0.8, "anxious": 0.9, "ashamed": 0.6,
             "content": 0.9, "devastated": 0.7, "joyful": 0.9, "sad": 0.9, "trusting": 0.8},
    },
}
HALLUC_ALPHAS = HALLUC_ALPHAS_BY_MODEL[MODEL_NAME]

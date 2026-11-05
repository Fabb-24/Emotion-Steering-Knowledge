# Emotion-Steering-Memory

**Does steering an LLM's emotions change what it knows?**

This project steers three instruction-tuned LLMs towards ten emotions with activation steering and measures how this affects their answers to factual yes/no questions. Steering vectors are computed as differences of means on the residual stream, at layers chosen with linear probes. For every emotion and layer, the steering strength is calibrated on free generation before the models are evaluated on BeliefBank.

The full write-up is in [docs/report/out/template.pdf](docs/report/out/template.pdf).

## Main findings

- **Where emotions are encoded:** the ten emotions become linearly separable after the first 20–36% of the network and stay so up to the last layer (probe macro-F1 0.50–0.61, chance 0.10).
- **Difference-of-means vectors:** simple vectors classify unseen situations almost as well as the probes. Similar emotions have similar vectors (angry/annoyed, sad/devastated, content/joyful).
- **Steering strength:** the shallowest steered layers break the text at lower strengths than the deepest ones.
- **Effect on factual knowledge:** it depends mainly on the steered layer.
  - At intermediate layers (50–61% of the depth), the mean BBF hallucination rate rises by 20–37 points. Answers often collapse to a constant "yes" (positive emotions) or "no" (negative emotions in the Qwen models).
  - At the deepest selected layers (84–89%), the hallucination rate stays close to the baseline. The emotion there appears only through a few words added to an otherwise unchanged text.

## Pipeline

| Step | Description |
|---|---|
| 1. Activations | EmpatheticDialogues situations are fed to the model as plain text and the last-token residual stream of every layer is stored. |
| 2. Layer selection | One linear softmax probe per layer is trained (macro-F1 on held-out situations). Candidate layers are picked on the plateau at different depths. |
| 3. Steering vectors | One-vs-rest difference of means per emotion and layer. The vector is normalized and added at every position as `h ← h + α · n̄ · v̂`, where `n̄` is the mean activation norm of the layer. |
| 4. Strength calibration | Free generation on 4 neutral prompts over a grid of α. The selected α is 0.2 below the first value that breaks the text, or 0.1 below if the emotion is not yet visible. |
| 5. Evaluation | Yes/no answers on 1,000 BBF facts and 1,000 BBC constraints, with greedy decoding. Every steering condition is compared with the unsteered baseline (hallucination rate, i.e. wrong or unknown answers, yes rate, unknown rate). |

Models and steered layers:

| Model | Layers | Steered layers | Run on |
|---|---|---|---|
| Qwen/Qwen2.5-7B-Instruct | 28 | 14, 20, 25 | local GPU, 4-bit |
| meta-llama/Llama-3.1-8B-Instruct | 32 | 17, 24, 27 | local GPU, 4-bit |
| Qwen/Qwen3.8-27B | 64 | 27, 39, 50, 54 | Modal (A100 80 GB), bf16 |

## Project structure

```
app/
  cli.py                 entry point (download, extract, stats + the layer / steer / halluc groups)
  config.py              all settings: active model, paths, hyperparameters, steered layers and α
  data_processing/       download of EmpatheticDialogues, activation extraction, splits, BeliefBank sampling
  layer_selection/       linear probes, per-layer metrics and plots
  steering/              steering vectors, checks (cosine, projection), free generation
  hallucination/         BeliefBank evaluation, metrics and plots
  models/                model loading, hooks, generation, probe, answer parsing
modal_app.py             runs any CLI command on a Modal GPU
data/                    emotions, BeliefBank, splits (activations and vectors are not versioned)
results/{model}/         outputs of each model, one folder per phase
results/general/         plots comparing the three models
docs/report/             LaTeX report
```

## Setup

```bash
pip install -r requirements.txt
```

You need a CUDA GPU (the two 7–8B models run in 4-bit on 8 GB). Llama 3.1 is gated, so log in to Hugging Face first (`huggingface-cli login`) with an account that has access.

BeliefBank (`BBF.csv`, `BBC.csv`) is already in `data/beliefbank/`. EmpatheticDialogues is downloaded by the `download` command.

## Usage

All commands are run from the project root. The active model and steering mode are set at the top of [app/config.py](app/config.py):

```python
MODEL_NAME = "qwen"       # "qwen", "llama" or "qwen38"
STEERING_MODE = "fixed"   # the report uses "fixed" (α as a fraction of the mean activation norm)
```

Every output goes to `results/<MODEL_NAME>/`. Run `python app/cli.py --help` (or `<group> --help`) for all options.

### 1. Data and activations
```bash
python app/cli.py download      # EmpatheticDialogues -> data/emotions/
python app/cli.py extract       # last-token activations of every layer -> data/activations/<model>/
python app/cli.py stats         # statistics of the datasets and splits
```

### 2. Layer selection
```bash
python app/cli.py layer all                  # splits, one probe per layer, macro-F1 plot
python app/cli.py layer confusion --layer 20 # confusion matrix at a given layer
```

### 3. Steering vectors and strength calibration
```bash
python app/cli.py steer all --layer 14 --layer 20 --layer 25   # vectors + cosine/projection checks
python app/cli.py steer generate --layer 14                    # free generation over the α grid
```
Read the generated texts in `results/<model>/steering/diff_means/fixed/generation_layer<N>.csv`. Then write the steered layers in `HALLUC_LAYERS_BY_MODEL` and the selected α in `HALLUC_ALPHAS_BY_MODEL` in `config.py`.

### 4. Evaluation on BeliefBank
```bash
python app/cli.py halluc dataset   # fixed samples of 1,000 BBF and 1,000 BBC statements (once)
python app/cli.py halluc eval      # baseline + every (layer, emotion) condition
python app/cli.py halluc analyze   # metrics CSV and per-layer plots
```

### 5. Comparison of the three models
After steps 2–4 have been run for every model:
```bash
python app/cli.py layer compare                   # probe macro-F1 vs relative depth
python app/cli.py halluc accuracy --hallucination # hallucination rate per emotion and layer
python app/cli.py halluc yes-rate                 # yes-rate heatmaps
```
The plots go to `results/general/`.

### Running on Modal
Qwen3.8-27B was run on [Modal](https://modal.com). The code is sent with every run, while `data/` and `results/` live on the `emotion-steering` volume. A `huggingface` secret holds `HF_TOKEN`. Any CLI command can be run remotely:
```bash
modal run modal_app.py --args "halluc eval --layer 39"
```
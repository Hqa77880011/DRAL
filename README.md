# DRAL

Implementation of **Disagreement-Resolved Active Learning with Abstaining Annotators**.

## Installation

Use Python 3.11+. Create and activate a virtual environment before installing:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

In PowerShell, replace the activation line with `.venv\Scripts\Activate.ps1`.

## Data and preprocessing

The controlled-source experiment uses real CIFAR-10 or CIFAR-100 images with generated source channels. Obtain images from the [CIFAR dataset](https://www.cs.toronto.edu/~kriz/cifar.html). CIFAR-N noisy fields can support a separate recorded-label replay protocol, but they do not define persistent annotator identities or refusals.

Create a CSV with `path,label,split`. Labels are contiguous zero-based class indices. `split` is `pool`, `audit`, or `test`. Reserve official test images for `test`; split training images into the paid acquisition pool and disjoint trusted audit set before querying. The audit labels are used only by the response-model evaluator.

Preparation uses the frozen torchvision ResNet-18 ImageNet encoder and its prescribed preprocessing; its weights are downloaded on first use. Alternatively provide a numeric archive directly with `x[N,D]`, `y[N]`, and `split[N]`. All policies must use identical frozen features.

`channels.py` creates a broad noisy source, two complementary class specialists, and an unreliable source. The generated JSON contains `probabilities[A,B,K,K+1]` and positive `costs[A]`: source, difficulty stratum, true class, and outcome. Outcome `K` is refusal. Rows sum to one. Edit these declared simulator parameters to study other response mechanisms. `--independent-refusal` removes class dependence while preserving the configured mean rate for balanced classes.

```bash
dral prepare --manifest data/images.csv --output data/features.npz
python channels.py --classes 10 --refusal 0.3 --output data/channels.json
```

## Run and evaluate

```bash
dral run --data data/features.npz --channels data/channels.json --output outputs/full --budget 1000 --anchors-per-class 5 --heads 8 --channels-samples 16 --seed 0
```

## Ablations

Supported controls: `label_only`, `constant_beta`, `new_only`, `random`. Reuse the same inputs, seed and budget. For retrained controls, evaluate the corresponding saved model.

```bash
dral run --data data/features.npz --channels data/channels.json --output outputs/label_only --budget 1000 --anchors-per-class 5 --heads 8 --channels-samples 16 --seed 0 --variant label_only
```

## Outputs

`results.json` records paid queries, posterior information, checkpoints and actual costs. `accuracy_cost.png` plots observed checkpoint accuracy against billed cost.

Source inspection: Python 3.13.5, numpy 2.1.3, torch 2.7.1+cu118, scikit-learn 1.6.1, matplotlib 3.10.0 (installed versions inspected). Checks covered Python syntax, configuration parsing and CLI arguments. No training, full evaluation or CI was run in this update.

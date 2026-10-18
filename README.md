# DRAL

Implementation of **Disagreement-Resolved Active Learning with Abstaining Annotators**.

DRAL keeps a finite joint bank of classifier and response-channel hypotheses. Every class response and refusal updates the same posterior once. Query utility measures predictive label information plus a decreasing source-information term per billed cost.

## Install

Use Python 3.11+ in an activated virtual environment:

```bash
python -m pip install .
```

## Prepare the image pool

The controlled-source experiment uses real CIFAR-10 or CIFAR-100 images with generated source channels. Obtain images from the [CIFAR dataset](https://www.cs.toronto.edu/~kriz/cifar.html). CIFAR-N noisy fields can support a separate recorded-label replay protocol, but they do not define persistent annotator identities or refusals.

Create a CSV with `path,label,split`. Labels are contiguous zero-based class indices. `split` is `pool`, `audit`, or `test`. Reserve official test images for `test`; split training images into the paid acquisition pool and disjoint trusted audit set before querying. The audit labels are used only by the response-model evaluator.

```bash
dral prepare --manifest data/images.csv --output data/features.npz
python channels.py --classes 10 --refusal 0.3 --output data/channels.json
```

Preparation uses the frozen torchvision ResNet-18 ImageNet encoder and its prescribed preprocessing; its weights are downloaded on first use. Alternatively provide a numeric archive directly with `x[N,D]`, `y[N]`, and `split[N]`. All policies must use identical frozen features.

`channels.py` creates a broad noisy source, two complementary class specialists, and an unreliable source. The generated JSON contains `probabilities[A,B,K,K+1]` and positive `costs[A]`: source, difficulty stratum, true class, and outcome. Outcome `K` is refusal. Rows sum to one. Edit these declared simulator parameters to study other response mechanisms. `--independent-refusal` removes class dependence while preserving the configured mean rate for balanced classes.

## Run acquisition and downstream training

```bash
dral run --data data/features.npz --channels data/channels.json --output outputs/full --budget 1000 --anchors-per-class 5 --heads 8 --channels-samples 16
```

Trusted anchors are purchased class by class from the pool. Each costs `--anchor-cost 1`; all source responses on those anchors are also charged. The remaining action set contains each available pool/source pair at most once. Potential source outcomes are sampled once per seed; the policy accesses an outcome only after selecting and paying for the pair.

Classifier hypotheses are bootstrap logistic-regression fits on the trusted anchors. Channel hypotheses are sampled from positive Dirichlet priors conditioned on the paid anchor responses. Difficulty strata are frozen quantiles of initial classifier uncertainty. A separate linear soft-target classifier is fitted at `--checkpoint-cost 100` billed intervals for `--train-epochs 100` optimization steps. Its predictions never re-enter the Bayesian acquisition state.

Important controls:

| Parameter | Meaning |
| --- | --- |
| `--beta 1` | Initial source-information weight |
| `--budget` | Total billed budget including anchors and refusals |
| `--candidates 128` | Maximum randomly filtered candidate images per request |
| `--resolution 0.05` | Resolved if maximum posterior class mass reaches 0.95 |
| `--seed` | Paired pool, bank, channel-outcome and query-filter randomness |

The budget must exceed initial trusted-anchor and source-calibration charges. With ten classes, five anchors per class, four unit-cost sources and unit trusted-label cost, initialization costs 250 units.

## Compare policies

```bash
dral run --data data/features.npz --channels data/channels.json --output outputs/random --variant random
dral run --data data/features.npz --channels data/channels.json --output outputs/label-only --variant label_only
dral run --data data/features.npz --channels data/channels.json --output outputs/constant-beta --variant constant_beta
dral run --data data/features.npz --channels data/channels.json --output outputs/new-only --variant new_only
```

`random` chooses an affordable pair uniformly and retains identical posterior inference. `label_only` sets source utility to zero while keeping Bayesian channel learning. `constant_beta` removes budget decay; `new_only` disallows second opinions. Repeat these commands with the same seeds and billed budgets. Sweep refusal using `channels.py`, including its class-independent arm, before changing bank parameters on a separate development split.

## Results

`results.json` stores every query, response probability, label/source information, role, request cost, and cumulative bill. Checkpoint records include test accuracy, balanced accuracy, minority recall, unresolved fraction, held-out response NLL, normalized source entropy, and cost by query role. `accuracy_cost.png` shows the resulting accuracy curve.

The normalized area under the accuracy-cost curve uses the observed checkpoint interval; report that interval with comparisons. Minority classes have pool frequency at or below the pool median. Source NLL uses trusted audit labels and held-out responses after acquisition decisions. The final checkpoint can contain only final accuracy and balanced accuracy when the remaining budget ends between regular checkpoints. Initialization and subsequent role costs sum to the final billed amount.

## Implementation choices

The shared frozen feature encoder and linear downstream classifier implement the paper's primary finite-bank tier. Initial anchor calibration queries all configured sources, so those anchor/source pairs are exhausted before active selection. The candidate filter is a seeded uniform subset shared in form across controls. Channel matrices and difficulty strata stay fixed during a run; response drift can be studied by preparing a different explicitly time-dependent simulator. The bank update includes refusals directly and never multiplies already-observed responses into a separately refitted channel model.

## Reproduction logic conventions

Source-response NLL is reported in bits per response, using log2 consistently with the manuscript tables and information utilities.

import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score
from torch import nn


def entropy(p):
    return -(p * np.log2(p.clip(1e-300))).sum(-1)


class Bank:
    def __init__(self, predictions, channels, anchors, labels):
        heads, n, k = predictions.shape
        samples = channels.shape[0]
        self.prior = predictions.copy()
        self.factors = {}
        self.k = k
        for i in anchors:
            self.prior[:, i] = np.eye(k)[labels[i]]
        self.channels = channels
        self.w = np.full((heads, samples), 1 / (heads * samples))

    def marginal(self):
        result = np.einsum("h,hnk->nk", self.w.sum(1), self.prior)
        for i, q in self.factors.items():
            result[i] = np.einsum("hs,hsk->k", self.w, q)
        return result

    def conditional(self, i):
        return self.factors[i] if i in self.factors else np.broadcast_to(self.prior[:, None, i], self.w.shape + (self.k,))

    def hypothetical(self, i, a, stratum, outcome):
        likelihood = self.channels[:, a, stratum, :, outcome]
        joint = self.conditional(i) * likelihood[None]
        conditional_probability = joint.sum(-1)
        probability = float((self.w * conditional_probability).sum())
        w = self.w * conditional_probability / max(probability, 1e-300)
        q = joint / conditional_probability[..., None].clip(1e-300)
        return probability, w, q

    def information(self, i, a, stratum):
        label = entropy(np.einsum("hs,hsk->k", self.w, self.conditional(i))); source = entropy(self.w.sum(0))
        for outcome in range(self.k + 1):
            probability, w, q = self.hypothetical(i, a, stratum, outcome)
            label -= probability * entropy(np.einsum("hs,hsk->k", w, q))
            source -= probability * entropy(w.sum(0))
        return max(0., float(label)), max(0., float(source))

    def update(self, i, a, stratum, outcome):
        probability, w, q = self.hypothetical(i, a, stratum, outcome)
        if probability <= 0:
            raise ValueError("Observed outcome outside finite-bank support")
        self.w, self.factors[i] = w, q
        return probability


def prepare(args):
    from torchvision.models import ResNet18_Weights, resnet18
    rows = list(csv.DictReader(Path(args.manifest).open(encoding="utf-8")))
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights); model.fc = nn.Identity(); model.eval()
    features = []
    with torch.no_grad():
        for row in rows:
            image = Image.open(row["path"]).convert("RGB")
            features.append(model(weights.transforms()(image)[None]).numpy()[0])
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, x=np.array(features), y=np.array([int(r["label"]) for r in rows]), split=np.array([r["split"] for r in rows]))


def soft_classifier(x, targets, anchors, labels, selected, test, epochs, seed):
    torch.manual_seed(seed)
    head = nn.Linear(x.shape[1], targets.shape[1]); optimizer = torch.optim.Adam(head.parameters(), lr=.01)
    ids = np.array(sorted(set(selected) | set(anchors)), dtype=int)
    target = targets[ids].copy()
    for j, i in enumerate(ids):
        if i in anchors:
            target[j] = np.eye(target.shape[1])[labels[i]]
    xx, yy = torch.tensor(x[ids], dtype=torch.float32), torch.tensor(target, dtype=torch.float32)
    for _ in range(epochs):
        loss = -(yy * head(xx).log_softmax(-1)).sum(-1).mean()
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    with torch.no_grad():
        return head(torch.tensor(x[test], dtype=torch.float32)).argmax(-1).numpy()


def run(args):
    rng = np.random.default_rng(args.seed)
    data = np.load(args.data, allow_pickle=False)
    x, y, split = data["x"], data["y"], data["split"]
    pool = np.flatnonzero(split == "pool"); test = np.flatnonzero(split == "test"); audit = np.flatnonzero(split == "audit")
    k = int(y.max()) + 1
    if not len(test) or not len(audit) or any(np.sum(y[pool] == c) < args.anchors_per_class for c in range(k)):
        raise ValueError("Require disjoint pool/test/audit and enough paid anchors in every class")
    anchors = np.concatenate([rng.choice(pool[y[pool] == c], args.anchors_per_class, replace=False) for c in range(k)])
    mean, scale = x[anchors].mean(0), x[anchors].std(0).clip(.01)
    x = (x - mean) / scale
    specification = json.loads(Path(args.channels).read_text(encoding="utf-8"))
    true_channels = np.array(specification["probabilities"], dtype=float)
    costs = np.array(specification["costs"], dtype=float)
    a_count, strata, classes, outcomes = true_channels.shape
    if classes != k or outcomes != k + 1 or not np.allclose(true_channels.sum(-1), 1) or np.any(costs <= 0) or np.any(true_channels < 0):
        raise ValueError("Channel tensor must be A x strata x K x (K+1), each row normalized")
    heads = []
    for _ in range(args.heads):
        ids = np.concatenate([rng.choice(anchors[y[anchors] == c], args.anchors_per_class, replace=True) for c in range(k)])
        model = LogisticRegression(C=1., max_iter=500).fit(x[ids], y[ids])
        heads.append(model.predict_proba(x))
    predictions = np.array(heads)
    difficulty = 1 - predictions.mean(0).max(-1)
    boundaries = np.quantile(difficulty[anchors], np.arange(1, strata) / strata)
    b = np.searchsorted(boundaries, difficulty)
    potential = np.empty((len(y), a_count), dtype=int)
    for i in range(len(y)):
        for a in range(a_count):
            potential[i, a] = rng.choice(k + 1, p=true_channels[a, b[i], y[i]])
    concentration = np.ones_like(true_channels)
    available = np.zeros((len(y), a_count), dtype=bool); available[pool] = True
    billed = len(anchors) * args.anchor_cost
    for i in anchors:
        for a in range(a_count):
            concentration[a, b[i], y[i], potential[i, a]] += 1
            billed += costs[a]; available[i, a] = False
    if billed >= args.budget:
        raise ValueError("Budget must exceed trusted anchors plus all anchor-source calibration requests")
    samples = np.empty((args.channels_samples,) + true_channels.shape)
    for s in range(args.channels_samples):
        for index in np.ndindex(true_channels.shape[:-1]):
            samples[(s,) + index] = rng.dirichlet(concentration[index])
    bank = Bank(predictions, samples, anchors, y)
    initial_bill = billed; queried, records, curve = set(), [], []
    role_cost = {"anchor": float(billed), "second_opinion": 0., "new_image": 0.}
    initial_entropy = float(entropy(bank.w.sum(0)))
    next_checkpoint = billed
    start = time.perf_counter()
    while True:
        affordable = available & (costs[None] <= args.budget - billed)
        if args.variant == "new_only":
            affordable[list(queried)] = False
        feasible = np.argwhere(affordable)
        if not len(feasible):
            break
        images = np.unique(feasible[:, 0])
        if len(images) > args.candidates:
            images = rng.choice(images, args.candidates, replace=False)
            feasible = feasible[np.isin(feasible[:, 0], images)]
        beta = args.beta if args.variant == "constant_beta" else args.beta * (args.budget - billed) / (args.budget - initial_bill)
        if args.variant == "label_only":
            beta = 0
        scores = []
        for i, a in feasible:
            iy, ins = bank.information(i, a, b[i])
            scores.append(((iy + beta * ins) / costs[a], iy, ins))
        chosen = int(rng.integers(len(feasible))) if args.variant == "random" else int(np.argmax([s[0] for s in scores]))
        i, a = feasible[chosen]; outcome = int(potential[i, a])
        role = "anchor" if i in anchors else "second_opinion" if i in queried else "new_image"
        probability = bank.update(i, a, b[i], outcome)
        available[i, a] = False; billed += costs[a]; role_cost[role] += float(costs[a]); queried.add(int(i))
        records.append({"image": int(i), "source": int(a), "outcome": outcome, "cost": float(costs[a]), "billed": float(billed), "role": role, "label_information": scores[chosen][1], "source_information": scores[chosen][2], "response_nll": -float(np.log2(max(probability, 1e-300)))})
        if billed >= next_checkpoint or not affordable.any():
            marginal = bank.marginal()
            pred = soft_classifier(x, marginal, anchors, y, queried, test, args.train_epochs, args.seed)
            nll = []
            weights = bank.w.sum(0)
            for j in audit:
                for source in range(a_count):
                    probability = float(weights @ samples[:, source, b[j], y[j], potential[j, source]])
                    nll.append(-np.log2(max(probability, 1e-300)))
            count = np.bincount(y[pool], minlength=k); minority = np.flatnonzero(count <= np.median(count))
            recalls = recall_score(y[test], pred, labels=np.arange(k), average=None, zero_division=0)
            curve.append({"billed": float(billed), "accuracy": accuracy_score(y[test], pred), "balanced_accuracy": balanced_accuracy_score(y[test], pred), "minority_recall": float(recalls[minority].mean()), "unresolved_fraction": float(np.mean(marginal[list(queried)].max(-1) < 1 - args.resolution)), "response_model_nll": float(np.mean(nll)), "source_entropy_fraction": float(entropy(weights)) / max(initial_entropy, 1e-12), "role_cost": role_cost.copy()})
            next_checkpoint = billed + args.checkpoint_cost
    if not curve or curve[-1]["billed"] != billed:
        pred = soft_classifier(x, bank.marginal(), anchors, y, queried, test, args.train_epochs, args.seed)
        curve.append({"billed": float(billed), "accuracy": accuracy_score(y[test], pred), "balanced_accuracy": balanced_accuracy_score(y[test], pred)})
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    costs_curve = np.array([r["billed"] for r in curve]); accuracy = np.array([r["accuracy"] for r in curve])
    area = float(np.sum((accuracy[1:] + accuracy[:-1]) * .5 * np.diff(costs_curve)) / (costs_curve[-1] - costs_curve[0])) if len(curve) > 1 else None
    result = {"curve": curve, "queries": records, "metrics": {"accuracy": float(accuracy[-1]), "normalized_accuracy_cost_area": area, "billed": float(billed), "seconds": time.perf_counter() - start, "role_cost": role_cost}}
    (output / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    plt.plot(costs_curve, accuracy); plt.xlabel("Total billed cost"); plt.ylabel("Test accuracy"); plt.savefig(output / "accuracy_cost.png", dpi=180)


def main():
    p = argparse.ArgumentParser(); sub = p.add_subparsers(required=True)
    q = sub.add_parser("prepare"); q.add_argument("--manifest", required=True); q.add_argument("--output", required=True); q.set_defaults(func=prepare)
    q = sub.add_parser("run")
    for key in ["data", "channels", "output"]:
        q.add_argument("--" + key, required=True)
    q.add_argument("--variant", choices=["full", "label_only", "constant_beta", "new_only", "random"], default="full")
    for key, val in [("budget", 1000.), ("anchor-cost", 1.), ("beta", 1.), ("checkpoint-cost", 100.), ("resolution", .05)]:
        q.add_argument("--" + key, type=float, default=val)
    for key, val in [("heads", 8), ("channels-samples", 16), ("anchors-per-class", 5), ("candidates", 128), ("train-epochs", 100), ("seed", 7)]:
        q.add_argument("--" + key, type=int, default=val)
    q.set_defaults(func=run); args = p.parse_args(); args.func(args)


if __name__ == "__main__":
    main()

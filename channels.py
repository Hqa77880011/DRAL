import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--classes", type=int, required=True); parser.add_argument("--strata", type=int, default=2); parser.add_argument("--refusal", type=float, default=.3); parser.add_argument("--independent-refusal", action="store_true"); parser.add_argument("--output", required=True); args = parser.parse_args()
    if args.classes < 2 or not 0 <= args.refusal <= 1:
        parser.error("Require at least two classes and refusal in [0,1]")
    k = args.classes; probabilities = np.zeros((4, args.strata, k, k + 1))
    for a in range(4):
        for b in range(args.strata):
            for y in range(k):
                specialist = a in [1, 2]
                favored = y % 2 == a - 1
                accuracy = [.75, .95 if favored else .5, .95 if favored else .5, .4][a]
                accuracy = max(1 / k, accuracy - .1 * b)
                refusal = args.refusal
                if specialist and not args.independent_refusal:
                    refusal = np.clip(args.refusal + (-.2 if favored else .2), 0, 1)
                probabilities[a, b, y, :k] = (1 - accuracy) / (k - 1)
                probabilities[a, b, y, y] = accuracy
                probabilities[a, b, y, :k] *= 1 - refusal
                probabilities[a, b, y, k] = refusal
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps({"probabilities": probabilities.tolist(), "costs": [1., 1., 1., 1.]}, indent=2))


if __name__ == "__main__":
    main()

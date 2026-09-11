from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT / "results"
SEED = 20260908
DRAWS = 20000
SEEDS = 5
PUBLISHED = ["BENNER74", "BLOSUM45", "BLOSUM62", "BLOSUM80", "PAM250"]
INTEREST = ["rank-one reconstruction, 41 numbers", "measured matrix, 190 numbers"]


def bootstrap(difference: np.ndarray, groups: np.ndarray, seed: int) -> np.ndarray:
    parts = [difference[groups == g] for g in np.unique(groups)]
    rng = np.random.default_rng(seed)
    index = rng.integers(0, len(parts), (DRAWS, len(parts)))
    return np.array([np.concatenate([parts[j] for j in row]).mean() for row in index])


def main() -> None:
    frame = pd.read_csv(RESULTS / "N6c_per_assay_spearman.csv")
    rows = []
    for block, part in frame.groupby("external_block", sort=False):
        groups = part["resampling_group"].to_numpy()
        means = {name: float(part[name].mean()) for name in PUBLISHED}
        best = max(means, key=means.get)
        print(f"\n{block}: strongest published is {best} at {means[best]:.4f}, "
              f"{len(groups)} assays in {len(np.unique(groups))} groups")
        for name in INTEREST + ["BLOSUM62"]:
            if name == best:
                continue
            difference = (part[name] - part[best]).to_numpy(float)
            low, five, high, below = [], [], [], []
            for step in range(SEEDS):
                draws = bootstrap(difference, groups, SEED + 100 * step)
                low.append(np.percentile(draws, 2.5))
                five.append(np.percentile(draws, 5))
                high.append(np.percentile(draws, 97.5))
                below.append(float((draws <= 0).mean()))
            rows.append({"external_block": block, "scale": name,
                         "strongest_published": best,
                         "published_macro_spearman": means[best],
                         "scale_macro_spearman": float(part[name].mean()),
                         "difference": float(difference.mean()),
                         "ci_low": float(np.mean(low)),
                         "ci_low_seed_spread": float(np.ptp(low)),
                         "one_sided_5pc": float(np.mean(five)),
                         "ci_high": float(np.mean(high)),
                         "share_of_draws_at_or_below_zero": float(np.mean(below))})
            print(f"  {name}: {difference.mean():+.4f}  "
                  f"95% [{np.mean(low):+.4f}, {np.mean(high):+.4f}]  "
                  f"one-sided 5th {np.mean(five):+.4f}  "
                  f"P(draw <= 0) = {np.mean(below):.3f}  "
                  f"(2.5th percentile varies by {np.ptp(low):.4f} across {SEEDS} seeds)")
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / "X8_direct_comparison_vs_strongest_published.csv", index=False)
    print("\nwrote X8_direct_comparison_vs_strongest_published.csv")


if __name__ == "__main__":
    main()

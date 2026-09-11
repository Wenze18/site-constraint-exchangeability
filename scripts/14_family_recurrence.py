from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DATA = PROJECT / "data"
RESULTS = PROJECT / "results"
SEED = 20260908

MIN_FAMILY_VARIANTS = 2000
FAMILY_CELL = 6
POOLED_CELL = 15
DRAWS = 2500
SPLITS = 200


def bins_within_protein(frame: pd.DataFrame, bins: int) -> pd.Series:
    return (frame.groupby("protein_id")["observed_position_intolerance"]
            .transform(lambda s: pd.qcut(s.rank(method="first"), bins,
                                         labels=False, duplicates="drop")))


def leave_one_out(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.dropna(subset=["effect_rank"]).copy()
    grouped = work.groupby(["protein_id", "position"])["effect_rank"]
    total, count = grouped.transform("sum"), grouped.transform("size")
    keep = count >= 4
    work, total, count = work[keep].copy(), total[keep], count[keep]
    work["contrast"] = work["effect_rank"] - (total - work["effect_rank"]) / (count - 1)
    return work


def paired(frame: pd.DataFrame, minimum: int) -> tuple:
    table = (frame.groupby(["reference_residue", "alternate_residue"])["contrast"]
             .agg(mean="mean", var="var", n="size").dropna())
    table = table[table["n"] >= minimum]
    table["se2"] = table["var"] / table["n"]
    index = set(table.index)
    seen, keep = set(), []
    for a, b in table.index:
        if (b, a) in index and (b, a) not in seen:
            seen.add((a, b))
            keep.append((a, b))
    if len(keep) < 20:
        return np.array([]), np.array([]), np.array([])
    forward = np.array([table.at[(a, b), "mean"] for a, b in keep])
    reverse = np.array([table.at[(b, a), "mean"] for a, b in keep])
    se2 = np.array([table.at[(a, b), "se2"] + table.at[(b, a), "se2"] for a, b in keep])
    return forward, reverse, se2


def fraction(forward, reverse, se2, index=None) -> float:
    if index is not None:
        forward, reverse, se2 = forward[index], reverse[index], se2[index]
    floor = se2.mean() / 4
    sym = max(((forward + reverse) / 2).var(ddof=1) - floor, 1e-12)
    anti = max(((forward - reverse) / 2).var(ddof=1) - floor, 1e-12)
    return sym / (sym + anti)


def interval(forward, reverse, se2, rng) -> tuple[float, float]:
    draws = np.array([fraction(forward, reverse, se2,
                               rng.integers(0, len(forward), len(forward)))
                      for _ in range(DRAWS)])
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def per_family(frame: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    sizes = frame.groupby("protein_family")["effect_rank"].size()
    eligible = sizes[sizes >= MIN_FAMILY_VARIANTS].index
    for family in sorted(eligible):
        part = frame[frame["protein_family"] == family].copy()
        part["tertile"] = bins_within_protein(part, 3)
        contrasts = leave_one_out(part)
        low = paired(contrasts[contrasts["tertile"] == 0], FAMILY_CELL)
        high = paired(contrasts[contrasts["tertile"] == 2], FAMILY_CELL)
        if not len(low[0]) or not len(high[0]):
            rows.append({"family": family, "variants": int(sizes[family]),
                         "proteins": int(part["protein_id"].nunique()),
                         "pairs_low": len(low[0]), "pairs_high": len(high[0]),
                         "share_low": np.nan, "share_high": np.nan,
                         "difference": np.nan})
            continue
        share_low, share_high = fraction(*low), fraction(*high)
        rows.append({"family": family, "variants": int(sizes[family]),
                     "proteins": int(part["protein_id"].nunique()),
                     "pairs_low": len(low[0]), "pairs_high": len(high[0]),
                     "share_low": share_low, "share_high": share_high,
                     "difference": share_high - share_low})
    return pd.DataFrame(rows)


def family_halves(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    families = np.array(sorted(frame["protein_family"].unique()))
    rng = np.random.default_rng(SEED + 11)
    collected = {("discovery", label): [] for label in range(5)}
    collected.update({("held out", label): [] for label in range(5)})
    gradients = []
    for step in range(SPLITS):
        order = rng.permutation(len(families))
        cut = len(families) // 2
        assignment = {"discovery": families[order[:cut]],
                      "held out": families[order[cut:]]}
        one = {}
        for side, members in assignment.items():
            part = frame[frame["protein_family"].isin(members)].copy()
            part["quintile"] = bins_within_protein(part, 5)
            contrasts = leave_one_out(part)
            values = []
            for label in range(5):
                arrays = paired(contrasts[contrasts["quintile"] == label], POOLED_CELL)
                value = fraction(*arrays) if len(arrays[0]) else np.nan
                collected[(side, label)].append(value)
                values.append(value)
            one[side] = values
        gradients.append({"split": step,
                          "discovery_gradient": one["discovery"][4] - one["discovery"][0],
                          "held_out_gradient": one["held out"][4] - one["held out"][0],
                          "discovery_monotone":
                              bool(np.all(np.diff(one["discovery"]) > 0)),
                          "held_out_monotone":
                              bool(np.all(np.diff(one["held out"]) > 0))})
    rows = []
    for (side, label), values in collected.items():
        values = np.array([v for v in values if np.isfinite(v)])
        rows.append({"side": side, "constraint_bin": f"C{label + 1}",
                     "splits": len(values), "fraction": float(values.mean()),
                     "ci_low": float(np.percentile(values, 2.5)),
                     "ci_high": float(np.percentile(values, 97.5))})
    curve = pd.DataFrame(rows).sort_values(["side", "constraint_bin"])
    return curve, pd.DataFrame(gradients)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(DATA / "abundance.parquet")
    frame = frame.dropna(subset=["effect_rank", "observed_position_intolerance"])
    print(f"{frame['protein_family'].nunique()} families, "
          f"{frame['protein_id'].nunique()} proteins", flush=True)

    table = per_family(frame)
    table.to_csv(RESULTS / "X2_family_gradient.csv", index=False)
    usable = table.dropna(subset=["difference"])
    positive = int((usable["difference"] > 0).sum())
    print(f"\n-- within-family gradient --")
    print(f"families with at least {MIN_FAMILY_VARIANTS} measurements: {len(table)}; "
          f"estimable in both tertiles: {len(usable)}")
    print(f"positive difference: {positive} of {len(usable)}")
    print(f"median difference {usable['difference'].median():+.3f}, "
          f"quartiles {usable['difference'].quantile(0.25):+.3f} to "
          f"{usable['difference'].quantile(0.75):+.3f}")
    n, k = len(usable), positive
    from math import comb
    tail = sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n
    print(f"one-sided sign test against a fair coin: p = {tail:.3g}")
    print(usable.sort_values("variants", ascending=False)
          [["family", "variants", "proteins", "share_low", "share_high", "difference"]]
          .head(12).to_string(index=False))
    pd.DataFrame([{"families_eligible": len(table), "families_estimable": len(usable),
                   "positive": positive,
                   "median_difference": float(usable["difference"].median()),
                   "q1": float(usable["difference"].quantile(0.25)),
                   "q3": float(usable["difference"].quantile(0.75)),
                   "sign_test_p": float(tail)}]).to_csv(
        RESULTS / "X2_family_sign_test.csv", index=False)

    curve, gradients = family_halves(frame)
    curve.to_csv(RESULTS / "X2_family_holdout_curve.csv", index=False)
    gradients.to_csv(RESULTS / "X2_family_holdout_splits.csv", index=False)
    print("\n-- discovery and held-out halves of the family list --")
    print(curve.to_string(index=False))
    print(f"\nheld-out gradient over {len(gradients)} splits: "
          f"{gradients['held_out_gradient'].mean():+.3f} "
          f"[{gradients['held_out_gradient'].quantile(0.025):+.3f}, "
          f"{gradients['held_out_gradient'].quantile(0.975):+.3f}]")
    print(f"held-out half monotone in "
          f"{int(gradients['held_out_monotone'].sum())} of {len(gradients)} splits")


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT / "results"
SEED = 20260908
GROUPS = {"aliphatic": set("ILVM"), "aromatic": set("FYW"), "small_polar": set("STA"),
          "acidic": set("DE"), "basic": set("KR"), "amide": set("NQ")}
GROUP_OF = {residue: name for name, members in GROUPS.items() for residue in members}


def spreads(block: pd.DataFrame, minimum: int = 15) -> dict:
    work = block.dropna(subset=["effect_rank"]).copy()
    total = work.groupby(["protein_id", "position"])["effect_rank"].transform("sum")
    count = work.groupby(["protein_id", "position"])["effect_rank"].transform("size")
    keep = count >= 4
    work, total, count = work[keep], total[keep], count[keep]
    work["contrast"] = work["effect_rank"] - (total - work["effect_rank"]) / (count - 1)
    table = (work.groupby(["reference_residue", "alternate_residue"])["contrast"]
             .agg(mean="mean", n="size").dropna())
    table = table[table["n"] >= minimum]
    within, between = [], []
    seen = set()
    for a, b in table.index:
        if (b, a) not in table.index or (b, a) in seen:
            continue
        seen.add((a, b))
        if a not in GROUP_OF or b not in GROUP_OF:
            continue
        weights = np.array([table.at[(a, b), "n"], table.at[(b, a), "n"]], float)
        means = np.array([table.at[(a, b), "mean"], table.at[(b, a), "mean"]])
        value = float(np.average(means, weights=weights))
        (within if GROUP_OF[a] == GROUP_OF[b] else between).append(value)
    if len(within) < 8 or len(between) < 40:
        return {}
    return {"within_spread": float(np.std(within, ddof=1)),
            "between_spread": float(np.std(between, ddof=1)),
            "within_n": len(within), "between_n": len(between)}


def main() -> None:
    frame = pd.read_parquet(PROJECT / "data" / "abundance.parquet")
    frame = frame.dropna(subset=["observed_position_intolerance"]).copy()
    frame["constraint_bin"] = (
        frame.groupby("protein_id")["observed_position_intolerance"]
        .transform(lambda s: pd.qcut(s.rank(method="first"), 5,
                                     labels=[f"C{i + 1}" for i in range(5)]).astype(object)))
    proteins = np.array(sorted(frame["protein_id"].unique()))
    rng = np.random.default_rng(SEED)

    point = {label: spreads(frame[frame["constraint_bin"] == label])
             for label in ("C4", "C5")}
    draws = {"between_ratio": [], "within_ratio": []}
    for _ in range(400):
        picked = proteins[rng.integers(0, len(proteins), len(proteins))]
        resampled = frame[frame["protein_id"].isin(set(picked))]
        four = spreads(resampled[resampled["constraint_bin"] == "C4"])
        five = spreads(resampled[resampled["constraint_bin"] == "C5"])
        if not four or not five:
            continue
        draws["between_ratio"].append(five["between_spread"] / four["between_spread"])
        draws["within_ratio"].append(five["within_spread"] / four["within_spread"])

    rows = []
    for name, key in (("between class (105 pairs)", "between"),
                      ("within class (15 pairs)", "within")):
        values = np.array(draws[f"{key}_ratio"])
        rows.append({
            "pair_class": name,
            "C4_spread": point["C4"][f"{key}_spread"],
            "C5_spread": point["C5"][f"{key}_spread"],
            "C5_over_C4": point["C5"][f"{key}_spread"] / point["C4"][f"{key}_spread"],
            "ci_low": float(np.percentile(values, 2.5)),
            "ci_high": float(np.percentile(values, 97.5)),
            "bootstrap_draws": len(values),
            "fraction_below_1": float((values < 1).mean())})
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / "R1_spread_ratio_intervals.csv", index=False)
    print(table.round(4).to_string(index=False))
    print("\nnote: the bootstrap resamples proteins, not residue pairs, because the 190")
    print("pair values are summaries of the same 503 proteins and are not independent.")


if __name__ == "__main__":
    main()

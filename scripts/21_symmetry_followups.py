from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

PROJECT = HERE.parent
DATA = PROJECT / "data"
RESULTS = PROJECT / "results"
SEED = 20260908
MIN_POSITIONS = 25
MIN_PAIRS = 20


def leave_one_out(frame: pd.DataFrame, outcome: str) -> pd.DataFrame:
    work = frame.dropna(subset=[outcome]).copy()
    total = work.groupby(["protein_id", "position"])[outcome].transform("sum")
    count = work.groupby(["protein_id", "position"])[outcome].transform("size")
    keep = count >= 4
    work, total, count = work[keep], total[keep], count[keep]
    work["contrast"] = work[outcome] - (total - work[outcome]) / (count - 1)
    return work


def paired_directions(contrasts: pd.DataFrame, minimum: int = MIN_POSITIONS):
    directed = contrasts.groupby(["reference_residue", "alternate_residue"])["contrast"].agg(
        ["mean", "size"])
    directed = directed[directed["size"] >= minimum]["mean"]
    seen, keep = set(), []
    for a, b in directed.index:
        if (b, a) in directed.index and (b, a) not in seen:
            seen.add((a, b))
            keep.append((a, b))
    if len(keep) < MIN_PAIRS:
        return None, None, keep
    forward = np.array([directed.loc[(a, b)] for a, b in keep])
    reverse = np.array([directed.loc[(b, a)] for a, b in keep])
    return forward, reverse, keep


def symmetric_fraction(forward, reverse, seed: int, replicates: int = 3000) -> dict:
    def fraction(index):
        f, r = forward[index], reverse[index]
        s, a = (f + r) / 2, (f - r) / 2
        return s.var(ddof=1) / (s.var(ddof=1) + a.var(ddof=1))
    rng = np.random.default_rng(seed)
    observed = fraction(np.arange(len(forward)))
    draws = np.array([fraction(rng.integers(0, len(forward), len(forward)))
                      for _ in range(replicates)])
    return {"symmetric_fraction": observed, "ci_low": float(np.percentile(draws, 2.5)),
            "ci_high": float(np.percentile(draws, 97.5)), "pairs": len(forward)}


def q3_packing_curve(frame: pd.DataFrame) -> pd.DataFrame:
    contrasts = leave_one_out(frame, "effect_rank")
    contrasts = contrasts.dropna(subset=["packing_percentile"])
    rows = []
    centres = np.arange(0.10, 0.91, 0.10)
    half_width = 0.15
    for centre in centres:
        window = contrasts[(contrasts["packing_percentile"] >= centre - half_width)
                           & (contrasts["packing_percentile"] <= centre + half_width)]
        forward, reverse, keep = paired_directions(window, minimum=15)
        if forward is None:
            rows.append({"packing_centre": centre, "pairs": len(keep),
                         "symmetric_fraction": np.nan})
            continue
        rows.append({"packing_centre": centre,
                     **symmetric_fraction(forward, reverse, SEED + int(centre * 100))})
    return pd.DataFrame(rows)


def q4_intolerance_mediation(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.dropna(subset=["observed_position_intolerance", "burial_class"]).copy()
    work["intolerance_tertile"] = (
        work.groupby("protein_id")["observed_position_intolerance"]
        .transform(lambda s: pd.qcut(s.rank(method="first"), 3,
                                     labels=["low", "mid", "high"]).astype(object)))
    rows = []
    for tertile in ("all", "low", "mid", "high"):
        subset = work if tertile == "all" else work[work["intolerance_tertile"] == tertile]
        contrasts = leave_one_out(subset, "effect_rank")
        per_level = {}
        for level in ("buried", "exposed"):
            forward, reverse, keep = paired_directions(
                contrasts[contrasts["burial_class"] == level], minimum=15)
            if forward is None:
                continue
            per_level[level] = symmetric_fraction(forward, reverse,
                                                  SEED + len(tertile) + len(level))
        if len(per_level) == 2:
            rows.append({"intolerance_stratum": tertile,
                         "buried": per_level["buried"]["symmetric_fraction"],
                         "exposed": per_level["exposed"]["symmetric_fraction"],
                         "gradient": (per_level["buried"]["symmetric_fraction"]
                                      - per_level["exposed"]["symmetric_fraction"]),
                         "buried_pairs": per_level["buried"]["pairs"],
                         "exposed_pairs": per_level["exposed"]["pairs"]})
    return pd.DataFrame(rows)


def q5_pair_movers(frame: pd.DataFrame) -> pd.DataFrame:
    contrasts = leave_one_out(frame, "effect_rank")
    per_level = {}
    for level in ("buried", "exposed"):
        part = contrasts[contrasts["burial_class"] == level]
        directed = part.groupby(["reference_residue", "alternate_residue"])["contrast"].agg(
            ["mean", "size"])
        directed = directed[directed["size"] >= MIN_POSITIONS]
        symmetric = {}
        for (a, b), row in directed.iterrows():
            if (b, a) not in directed.index:
                continue
            key = tuple(sorted((a, b)))
            if key in symmetric:
                continue
            other = directed.loc[(b, a)]
            weights = np.array([row["size"], other["size"]], dtype=float)
            symmetric[key] = float(np.average([row["mean"], other["mean"]], weights=weights))
        per_level[level] = symmetric
    common = set(per_level["buried"]) & set(per_level["exposed"])
    rows = [{"residue_a": a, "residue_b": b,
             "buried": per_level["buried"][(a, b)],
             "exposed": per_level["exposed"][(a, b)],
             "buried_minus_exposed": per_level["buried"][(a, b)] - per_level["exposed"][(a, b)]}
            for a, b in sorted(common)]
    out = pd.DataFrame(rows)
    out["abs_move"] = out["buried_minus_exposed"].abs()
    return out.sort_values("buried_minus_exposed", ascending=False)


def q6_consensus(frame: pd.DataFrame) -> pd.DataFrame:
    experts = ["structure_sequence_expert_rank", "structure_language_expert_rank",
               "stability_language_expert_rank", "sequence_language_expert_rank",
               "inverse_folding_expert_rank"]
    work = frame.copy()
    work["consensus_rank"] = work[experts].mean(axis=1)
    rows = []
    for name, column in [("experiment", "effect_rank"),
                         ("five-expert consensus", "consensus_rank")] + \
                        [(c.replace("_expert_rank", ""), c) for c in experts]:
        contrasts = leave_one_out(work, column)
        per_level = {}
        for level in ("buried", "exposed"):
            forward, reverse, _ = paired_directions(
                contrasts[contrasts["burial_class"] == level], minimum=MIN_POSITIONS)
            if forward is None:
                continue
            per_level[level] = symmetric_fraction(forward, reverse, SEED + len(name))
        if len(per_level) == 2:
            rows.append({"target": name,
                         "buried": per_level["buried"]["symmetric_fraction"],
                         "exposed": per_level["exposed"]["symmetric_fraction"],
                         "gradient": (per_level["buried"]["symmetric_fraction"]
                                      - per_level["exposed"]["symmetric_fraction"])})
    return pd.DataFrame(rows)


def q7_second_endpoint() -> pd.DataFrame:
    frame = pd.read_parquet(DATA / "biochemical.parquet")
    rows = []
    for endpoint, part in frame.groupby("endpoint_class"):
        positions = part.groupby(["protein_id", "position"]).size()
        eligible = int((positions >= 4).sum())
        contrasts = leave_one_out(part, "effect_rank")
        if contrasts.empty:
            rows.append({"endpoint": endpoint, "positions_total": len(positions),
                         "positions_with_4plus": eligible, "pairs_with_25plus": 0,
                         "pairs_with_10plus": 0, "verdict": "no eligible positions"})
            continue
        directed = contrasts.groupby(["reference_residue",
                                      "alternate_residue"])["contrast"].size()
        rows.append({
            "endpoint": endpoint, "positions_total": len(positions),
            "positions_with_4plus": eligible,
            "pairs_with_25plus": int((directed >= 25).sum()),
            "pairs_with_10plus": int((directed >= 10).sum()),
            "verdict": "estimable" if (directed >= 10).sum() >= 2 * MIN_PAIRS
                       else "underpowered",
        })
    return pd.DataFrame(rows)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(DATA / "abundance.parquet")

    print("Q3  symmetric fraction along continuous packing")
    q3 = q3_packing_curve(frame)
    q3.to_csv(RESULTS / "followup_q3_packing_curve.csv", index=False)
    print(q3.round(4).to_string(index=False))

    print("\nQ4  burial gradient within strata of equal position intolerance")
    q4 = q4_intolerance_mediation(frame)
    q4.to_csv(RESULTS / "followup_q4_mediation.csv", index=False)
    print(q4.round(4).to_string(index=False))

    print("\nQ5  residue pairs that move most between exposed and buried")
    q5 = q5_pair_movers(frame)
    q5.to_csv(RESULTS / "followup_q5_pair_movers.csv", index=False)
    print("  more tolerated when BURIED:")
    print("   " + q5.head(8).round(4).to_string(index=False).replace("\n", "\n   "))
    print("  more tolerated when EXPOSED:")
    print("   " + q5.tail(8).round(4).to_string(index=False).replace("\n", "\n   "))

    print("\nQ6  does the consensus recover the gradient its members miss?")
    q6 = q6_consensus(frame)
    q6.to_csv(RESULTS / "followup_q6_consensus.csv", index=False)
    print(q6.round(4).to_string(index=False))

    print("\nQ7  can the symmetry be tested on a second endpoint?")
    q7 = q7_second_endpoint()
    q7.to_csv(RESULTS / "followup_q7_second_endpoint.csv", index=False)
    print(q7.to_string(index=False))


if __name__ == "__main__":
    main()

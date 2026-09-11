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


def leave_one_out(frame: pd.DataFrame, outcome: str, minimum: int = 4) -> pd.DataFrame:
    work = frame.dropna(subset=[outcome]).copy()
    total = work.groupby(["protein_id", "position"])[outcome].transform("sum")
    count = work.groupby(["protein_id", "position"])[outcome].transform("size")
    keep = count >= minimum
    work, total, count = work[keep], total[keep], count[keep]
    work["contrast"] = work[outcome] - (total - work[outcome]) / (count - 1)
    return work


def pair_estimates(contrasts: pd.DataFrame, minimum: int) -> pd.DataFrame:
    grouped = contrasts.groupby(["reference_residue", "alternate_residue"])["contrast"]
    table = grouped.agg(mean="mean", var="var", n="size").dropna()
    table = table[table["n"] >= minimum]
    table["se2"] = table["var"] / table["n"]
    return table


def symmetry(table: pd.DataFrame, seed: int, replicates: int = 3000,
             correct_noise: bool = True) -> dict:
    seen, keep = set(), []
    for a, b in table.index:
        if (b, a) in table.index and (b, a) not in seen:
            seen.add((a, b))
            keep.append((a, b))
    if len(keep) < 20:
        return {"pairs": len(keep), "fraction": np.nan, "ci_low": np.nan,
                "ci_high": np.nan, "noise_share": np.nan}
    forward = np.array([table.at[(a, b), "mean"] for a, b in keep])
    reverse = np.array([table.at[(b, a), "mean"] for a, b in keep])
    se2 = np.array([table.at[(a, b), "se2"] + table.at[(b, a), "se2"] for a, b in keep])

    def fraction(index):
        f, r, s = forward[index], reverse[index], se2[index]
        symmetric = ((f + r) / 2).var(ddof=1)
        antisymmetric = ((f - r) / 2).var(ddof=1)
        if correct_noise:
            floor = s.mean() / 4
            symmetric = max(symmetric - floor, 1e-12)
            antisymmetric = max(antisymmetric - floor, 1e-12)
        return symmetric / (symmetric + antisymmetric)

    rng = np.random.default_rng(seed)
    observed = fraction(np.arange(len(keep)))
    draws = np.array([fraction(rng.integers(0, len(keep), len(keep)))
                      for _ in range(replicates)])
    raw_symmetric = ((forward + reverse) / 2).var(ddof=1)
    raw_antisymmetric = ((forward - reverse) / 2).var(ddof=1)
    return {"pairs": len(keep), "fraction": observed,
            "ci_low": float(np.percentile(draws, 2.5)),
            "ci_high": float(np.percentile(draws, 97.5)),
            "noise_share": float((se2.mean() / 2) / (raw_symmetric + raw_antisymmetric))}


def part1(frame: pd.DataFrame) -> None:
    work = frame.dropna(subset=["observed_position_intolerance", "burial_class"]).copy()
    work["tertile"] = (work.groupby("protein_id")["observed_position_intolerance"]
                       .transform(lambda s: pd.qcut(s.rank(method="first"), 3,
                                                    labels=["low", "mid", "high"]
                                                    ).astype(object)))
    rows = []
    for tertile in ("all", "low", "mid", "high"):
        subset = work if tertile == "all" else work[work["tertile"] == tertile]
        contrasts = leave_one_out(subset, "effect_rank")
        entry = {"stratum": tertile}
        for level in ("buried", "exposed"):
            table = pair_estimates(contrasts[contrasts["burial_class"] == level], 15)
            raw = symmetry(table, SEED + len(level), correct_noise=False)
            corrected = symmetry(table, SEED + len(level), correct_noise=True)
            entry[f"{level}_raw"] = raw["fraction"]
            entry[f"{level}_corrected"] = corrected["fraction"]
            entry[f"{level}_noise_share"] = raw["noise_share"]
            entry[f"{level}_pairs"] = raw["pairs"]
        entry["gradient_raw"] = entry["buried_raw"] - entry["exposed_raw"]
        entry["gradient_corrected"] = entry["buried_corrected"] - entry["exposed_corrected"]
        rows.append(entry)
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / "corrected_intolerance_mediation.csv", index=False)
    columns = ["stratum", "buried_raw", "buried_corrected", "exposed_raw",
               "exposed_corrected", "gradient_raw", "gradient_corrected",
               "buried_noise_share", "exposed_noise_share"]
    print(table[columns].round(3).to_string(index=False))


def part2() -> None:
    abundance = pd.read_parquet(DATA / "abundance.parquet")
    biochemical = pd.read_parquet(DATA / "biochemical.parquet")
    stability = biochemical[biochemical["endpoint_class"] == "stability"]

    rows = []
    for name, frame, minimum in (("abundance (503 domains)", abundance, 25),
                                 ("VenusMutHub stability (150 parents)", stability, 10)):
        contrasts = leave_one_out(frame, "effect_rank")
        for level in ("buried", "exposed"):
            part = contrasts[contrasts["burial_class"] == level]
            table = pair_estimates(part, minimum)
            result = symmetry(table, SEED + len(level), correct_noise=True)
            raw = symmetry(table, SEED + len(level), correct_noise=False)
            rows.append({"dataset": name, "burial_class": level,
                         "pairs": result["pairs"], "positions": len(part),
                         "fraction_raw": raw["fraction"],
                         "fraction_corrected": result["fraction"],
                         "ci_low": result["ci_low"], "ci_high": result["ci_high"],
                         "noise_share": raw["noise_share"]})
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / "corrected_second_endpoint.csv", index=False)
    print(table.round(3).to_string(index=False))

    def symmetric_values(frame, minimum):
        contrasts = leave_one_out(frame, "effect_rank")
        out = {}
        for level in ("buried", "exposed"):
            table = pair_estimates(contrasts[contrasts["burial_class"] == level], minimum)
            values = {}
            for a, b in table.index:
                if (b, a) not in table.index:
                    continue
                key = tuple(sorted((a, b)))
                if key in values:
                    continue
                w = np.array([table.at[(a, b), "n"], table.at[(b, a), "n"]], float)
                m = np.array([table.at[(a, b), "mean"], table.at[(b, a), "mean"]], float)
                values[key] = float(np.average(m, weights=w))
            out[level] = values
        return out

    first = symmetric_values(abundance, 25)
    second = symmetric_values(stability, 10)
    agreement = []
    for level in ("buried", "exposed"):
        shared = sorted(set(first[level]) & set(second[level]))
        if len(shared) < 15:
            agreement.append({"burial_class": level, "shared_pairs": len(shared),
                              "spearman": np.nan})
            continue
        x = np.array([first[level][k] for k in shared])
        y = np.array([second[level][k] for k in shared])
        from scipy.stats import spearmanr
        rho = spearmanr(x, y).statistic
        rng = np.random.default_rng(SEED)
        draws = [spearmanr(x[i], y[i]).statistic
                 for i in rng.integers(0, len(x), (2000, len(x)))]
        agreement.append({"burial_class": level, "shared_pairs": len(shared),
                          "spearman": float(rho),
                          "ci_low": float(np.nanpercentile(draws, 2.5)),
                          "ci_high": float(np.nanpercentile(draws, 97.5))})
    table = pd.DataFrame(agreement)
    table.to_csv(RESULTS / "corrected_endpoint_agreement.csv", index=False)
    print("\n  do the two endpoints rank the same residue pairs the same way?")
    print("   " + table.round(3).to_string(index=False).replace("\n", "\n   "))


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(DATA / "abundance.parquet")
    print("PART 1  is the intolerance mediation real, or noise?")
    part1(frame)
    print("\nPART 2  does the symmetry replicate on VenusMutHub stability?")
    part2()


if __name__ == "__main__":
    main()

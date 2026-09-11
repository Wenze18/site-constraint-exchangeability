from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE))

PROJECT = HERE.parent
DATA = PROJECT / "data"
RESULTS = PROJECT / "results"
SEED = 20260908

_ORDER = "ARNDCQEGHILKMFPSTWYV"
_ROWS = [
    "4 -1 -2 -2  0 -1 -1  0 -2 -1 -1 -1 -1 -2 -1  1  0 -3 -2  0",
    "-1  5  0 -2 -3  1  0 -2  0 -3 -2  2 -1 -3 -2 -1 -1 -3 -2 -3",
    "-2  0  6  1 -3  0  0  0  1 -3 -3  0 -2 -3 -2  1  0 -4 -2 -3",
    "-2 -2  1  6 -3  0  2 -1 -1 -3 -4 -1 -3 -3 -1  0 -1 -4 -3 -3",
    "0 -3 -3 -3  9 -3 -4 -3 -3 -1 -1 -3 -1 -2 -3 -1 -1 -2 -2 -1",
    "-1  1  0  0 -3  5  2 -2  0 -3 -2  1  0 -3 -1  0 -1 -2 -1 -2",
    "-1  0  0  2 -4  2  5 -2  0 -3 -3  1 -2 -3 -1  0 -1 -3 -2 -2",
    "0 -2  0 -1 -3 -2 -2  6 -2 -4 -4 -2 -3 -3 -2  0 -2 -2 -3 -3",
    "-2  0  1 -1 -3  0  0 -2  8 -3 -3 -1 -2 -1 -2 -1 -2 -2  2 -3",
    "-1 -3 -3 -3 -1 -3 -3 -4 -3  4  2 -3  1  0 -3 -2 -1 -3 -1  3",
    "-1 -2 -3 -4 -1 -2 -3 -4 -3  2  4 -2  2  0 -3 -2 -1 -2 -1  1",
    "-1  2  0 -1 -3  1  1 -2 -1 -3 -2  5 -1 -3 -1  0 -1 -3 -2 -2",
    "-1 -1 -2 -3 -1  0 -2 -3 -2  1  2 -1  5  0 -2 -1 -1 -1 -1  1",
    "-2 -3 -3 -3 -2 -3 -3 -3 -1  0  0 -3  0  6 -4 -2 -2  1  3 -1",
    "-1 -2 -2 -1 -3 -1 -1 -2 -2 -3 -3 -1 -2 -4  7 -1 -1 -4 -3 -2",
    "1 -1  1  0 -1  0  0  0 -1 -2 -2  0 -1 -2 -1  4  1 -3 -2 -2",
    "0 -1  0 -1 -1 -1 -1 -2 -2 -1 -1 -1 -1 -2 -1  1  5 -2 -2  0",
    "-3 -3 -4 -4 -2 -2 -3 -2 -2 -3 -2 -3 -1  1 -4 -3 -2 11  2 -3",
    "-2 -2 -2 -3 -2 -1 -2 -3  2 -1 -1 -2 -1  3 -3 -2 -2  2  7 -1",
    "0 -3 -3 -3 -1 -2 -2 -3 -3  3  1 -2  1 -1 -2 -2  0 -3 -1  4",
]
BLOSUM62 = {(a, b): int(v) for a, row in zip(_ORDER, _ROWS)
            for b, v in zip(_ORDER, row.split())}
HYDROPATHY = {"A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5,
              "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8,
              "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2}
VOLUME = {"S": 32, "R": 124, "L": 111, "P": 32.5, "T": 61, "A": 31, "V": 84, "G": 3,
          "I": 111, "F": 132, "Y": 136, "C": 55, "H": 96, "Q": 85, "N": 56, "K": 119,
          "D": 54, "E": 83, "M": 105, "W": 170}
CHARGE = {a: 0 for a in HYDROPATHY}
CHARGE.update({"D": -1, "E": -1, "K": 1, "R": 1})


def leave_one_out(frame, outcome):
    work = frame.dropna(subset=[outcome]).copy()
    total = work.groupby(["protein_id", "position"])[outcome].transform("sum")
    count = work.groupby(["protein_id", "position"])[outcome].transform("size")
    keep = count >= 4
    work, total, count = work[keep], total[keep], count[keep]
    work["contrast"] = work[outcome] - (total - work[outcome]) / (count - 1)
    return work


def pair_components(contrasts, minimum=15):
    table = (contrasts.groupby(["reference_residue", "alternate_residue"])["contrast"]
             .agg(mean="mean", n="size"))
    table = table[table["n"] >= minimum]
    seen, rows = set(), []
    for a, b in table.index:
        if (b, a) not in table.index or (b, a) in seen:
            continue
        seen.add((a, b))
        f, r = table.at[(a, b), "mean"], table.at[(b, a), "mean"]
        rows.append({"a": a, "b": b, "symmetric": (f + r) / 2,
                     "antisymmetric_ab": (f - r) / 2,
                     "n": table.at[(a, b), "n"] + table.at[(b, a), "n"]})
    return pd.DataFrame(rows)


def constraint_bins(frame, bins=5):
    work = frame.dropna(subset=["observed_position_intolerance"]).copy()
    work["constraint_bin"] = (work.groupby("protein_id")["observed_position_intolerance"]
                              .transform(lambda s: pd.qcut(s.rank(method="first"), bins,
                                                           labels=[f"C{i+1}" for i in range(bins)]
                                                           ).astype(object)))
    return work


def part_a(frame):
    binned = constraint_bins(frame)
    rng = np.random.default_rng(SEED)
    rows = []
    for label in sorted(binned["constraint_bin"].dropna().unique()):
        components = pair_components(
            leave_one_out(binned[binned["constraint_bin"] == label], "effect_rank"))
        if len(components) < 20:
            continue
        components["blosum"] = [BLOSUM62[(a, b)] for a, b
                                in zip(components["a"], components["b"])]
        rho = spearmanr(components["symmetric"], components["blosum"]).statistic
        draws = [spearmanr(components["symmetric"].to_numpy()[i],
                           components["blosum"].to_numpy()[i]).statistic
                 for i in rng.integers(0, len(components), (2000, len(components)))]
        rows.append({"constraint_bin": label, "pairs": len(components),
                     "spearman_vs_blosum62": float(rho),
                     "ci_low": float(np.nanpercentile(draws, 2.5)),
                     "ci_high": float(np.nanpercentile(draws, 97.5))})
    for level in ("exposed", "intermediate", "buried"):
        components = pair_components(
            leave_one_out(frame[frame["burial_class"] == level], "effect_rank"))
        if len(components) < 20:
            continue
        components["blosum"] = [BLOSUM62[(a, b)] for a, b
                                in zip(components["a"], components["b"])]
        rows.append({"constraint_bin": f"burial:{level}", "pairs": len(components),
                     "spearman_vs_blosum62": float(
                         spearmanr(components["symmetric"], components["blosum"]).statistic),
                     "ci_low": np.nan, "ci_high": np.nan})
    return pd.DataFrame(rows)


def part_b(frame):
    rows = []
    for level in ("buried", "exposed"):
        components = pair_components(
            leave_one_out(frame[frame["burial_class"] == level], "effect_rank"))
        if len(components) < 20:
            continue
        delta_hydrophobicity = np.array([HYDROPATHY[b] - HYDROPATHY[a] for a, b
                                         in zip(components["a"], components["b"])])
        delta_volume = np.array([VOLUME[b] - VOLUME[a] for a, b
                                 in zip(components["a"], components["b"])])
        delta_charge = np.array([CHARGE[b] - CHARGE[a] for a, b
                                 in zip(components["a"], components["b"])])
        y = components["antisymmetric_ab"].to_numpy()
        for name, x in (("delta_hydrophobicity", delta_hydrophobicity),
                        ("delta_volume", delta_volume),
                        ("delta_charge", delta_charge)):
            rows.append({"burial_class": level, "descriptor": name, "pairs": len(y),
                         "spearman": float(spearmanr(x, y).statistic)})
        design = np.column_stack([np.ones(len(y)), delta_hydrophobicity,
                                  delta_volume, delta_charge])
        beta, residual, *_ = np.linalg.lstsq(design, y, rcond=None)
        predicted = design @ beta
        r2 = 1 - ((y - predicted) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        rows.append({"burial_class": level, "descriptor": "all three combined (R2)",
                     "pairs": len(y), "spearman": float(r2)})
        sym = components["symmetric"].to_numpy()
        beta_s, *_ = np.linalg.lstsq(design, sym, rcond=None)
        pred_s = design @ beta_s
        r2_s = 1 - ((sym - pred_s) ** 2).sum() / ((sym - sym.mean()) ** 2).sum()
        rows.append({"burial_class": level,
                     "descriptor": "all three vs SYMMETRIC component (R2)",
                     "pairs": len(y), "spearman": float(r2_s)})
    return pd.DataFrame(rows)


def part_c(frame, splits=12):
    binned = constraint_bins(frame, bins=3)
    binned["pair"] = [tuple(sorted((a, b))) for a, b
                      in zip(binned["reference_residue"], binned["alternate_residue"])]
    predictors = {"ThermoMPNN": "inverse_folding_expert_rank",
                  "SPURS": "structure_sequence_expert_rank",
                  "Pythia": "stability_language_expert_rank"}
    rng = np.random.default_rng(SEED)
    proteins = binned["protein_id"].unique()
    records = []
    for _ in range(splits):
        order = rng.permutation(len(proteins))
        train_ids = set(proteins[order[: len(proteins) // 2]])
        train = binned[binned["protein_id"].isin(train_ids)]
        test = binned[~binned["protein_id"].isin(train_ids)].copy()
        contrasts = leave_one_out(train, "effect_rank")
        contrasts["pair"] = [tuple(sorted((a, b))) for a, b
                             in zip(contrasts["reference_residue"],
                                    contrasts["alternate_residue"])]
        scale = contrasts.groupby(["pair", "burial_class"])["contrast"].mean()
        default = float(contrasts["contrast"].mean())
        for part, column in predictors.items():
            fit = train.assign(exchange=[scale.get((p, b), default) for p, b
                                         in zip(train["pair"], train["burial_class"])])
            fit = fit.dropna(subset=[column, "effect_rank"])
            design = np.column_stack([np.ones(len(fit)), fit[column], fit["exchange"]])
            beta, *_ = np.linalg.lstsq(design, fit["effect_rank"].to_numpy(float), rcond=None)
            for label in ("C1", "C2", "C3"):
                block = test[test["constraint_bin"] == label].copy()
                block["exchange"] = [scale.get((p, b), default) for p, b
                                     in zip(block["pair"], block["burial_class"])]
                block = block.dropna(subset=[column, "effect_rank"])
                if len(block) < 500:
                    continue
                combined = beta[0] + beta[1] * block[column] + beta[2] * block["exchange"]
                plain = spearmanr(block["effect_rank"], block[column]).statistic
                repaired = spearmanr(block["effect_rank"], combined).statistic
                records.append({"predictor": part, "constraint_bin": label,
                                "gain": float(repaired - plain)})
    detail = pd.DataFrame(records)
    return detail.groupby(["predictor", "constraint_bin"])["gain"].agg(
        ["mean", "std", "size"]).reset_index()


def part_d():
    biochemical = pd.read_parquet(DATA / "biochemical.parquet")
    stability = biochemical[biochemical["endpoint_class"] == "stability"]
    contrasts = leave_one_out(stability, "effect_rank")
    rows = []
    for label, mask in (("buried", contrasts["burial_class"] == "buried"),
                        ("exposed", contrasts["burial_class"] == "exposed"),
                        ("exposed + intermediate",
                         contrasts["burial_class"].isin(["exposed", "intermediate"]))):
        for minimum in (5, 8, 10, 15):
            components = pair_components(contrasts[mask], minimum=minimum)
            rows.append({"stratum": label, "min_positions": minimum,
                         "pairs": len(components)})
    return pd.DataFrame(rows)


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(DATA / "abundance.parquet")

    print("A  measured exchange matrix versus BLOSUM62, by site constraint")
    a = part_a(frame)
    a.to_csv(RESULTS / "residual_blosum_agreement.csv", index=False)
    print(a.round(3).to_string(index=False))

    print("\nB  what does the antisymmetric quarter encode?")
    b = part_b(frame)
    b.to_csv(RESULTS / "residual_antisymmetric_content.csv", index=False)
    print(b.round(3).to_string(index=False))

    print("\nC  does the repair concentrate at constrained positions?")
    c = part_c(frame)
    c.to_csv(RESULTS / "residual_repair_by_constraint.csv", index=False)
    print(c.round(4).to_string(index=False))

    print("\nD  can the stability endpoint reach its exposed side?")
    d = part_d()
    d.to_csv(RESULTS / "residual_stability_coverage.csv", index=False)
    print(d.to_string(index=False))


if __name__ == "__main__":
    main()

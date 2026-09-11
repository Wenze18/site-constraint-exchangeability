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

PREDICTORS = {
    "SPURS": ("structure_sequence_expert_rank", "structure-conditioned"),
    "SaProt-DeltaG": ("structure_language_expert_rank", "structure-conditioned"),
    "ThermoMPNN": ("inverse_folding_expert_rank", "structure-conditioned"),
    "Pythia": ("stability_language_expert_rank", "sequence-derived"),
    "ESM-2": ("sequence_language_expert_rank", "sequence-derived"),
}
GROUPS = {"aliphatic": set("ILVM"), "aromatic": set("FYW"), "small_polar": set("STA"),
          "acidic": set("DE"), "basic": set("KR"), "amide": set("NQ")}
GROUP_OF = {r: g for g, members in GROUPS.items() for r in members}


def leave_one_out(frame: pd.DataFrame, outcome: str) -> pd.DataFrame:
    work = frame.dropna(subset=[outcome]).copy()
    total = work.groupby(["protein_id", "position"])[outcome].transform("sum")
    count = work.groupby(["protein_id", "position"])[outcome].transform("size")
    keep = count >= 4
    work, total, count = work[keep], total[keep], count[keep]
    work["contrast"] = work[outcome] - (total - work[outcome]) / (count - 1)
    return work


def pair_table(contrasts: pd.DataFrame, minimum: int = 15) -> pd.DataFrame:
    grouped = contrasts.groupby(["reference_residue", "alternate_residue"])["contrast"]
    table = grouped.agg(mean="mean", var="var", n="size").dropna()
    table = table[table["n"] >= minimum]
    table["se2"] = table["var"] / table["n"]
    return table


def unordered(table: pd.DataFrame):
    seen, keep = set(), []
    for a, b in table.index:
        if (b, a) in table.index and (b, a) not in seen:
            seen.add((a, b))
            keep.append((a, b))
    return keep


def symmetry(table: pd.DataFrame, seed: int, replicates: int = 2500) -> dict:
    keep = unordered(table)
    if len(keep) < 20:
        return {"pairs": len(keep), "fraction": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    forward = np.array([table.at[(a, b), "mean"] for a, b in keep])
    reverse = np.array([table.at[(b, a), "mean"] for a, b in keep])
    se2 = np.array([table.at[(a, b), "se2"] + table.at[(b, a), "se2"] for a, b in keep])

    def fraction(index):
        f, r, s = forward[index], reverse[index], se2[index]
        floor = s.mean() / 4
        sym = max(((f + r) / 2).var(ddof=1) - floor, 1e-12)
        anti = max(((f - r) / 2).var(ddof=1) - floor, 1e-12)
        return sym / (sym + anti)

    rng = np.random.default_rng(seed)
    draws = np.array([fraction(rng.integers(0, len(keep), len(keep)))
                      for _ in range(replicates)])
    return {"pairs": len(keep), "fraction": fraction(np.arange(len(keep))),
            "ci_low": float(np.percentile(draws, 2.5)),
            "ci_high": float(np.percentile(draws, 97.5))}


def block_contrast(table: pd.DataFrame) -> dict:
    keep = unordered(table)
    within, between = [], []
    for a, b in keep:
        w = np.array([table.at[(a, b), "n"], table.at[(b, a), "n"]], float)
        m = np.array([table.at[(a, b), "mean"], table.at[(b, a), "mean"]], float)
        value = float(np.average(m, weights=w))
        if a in GROUP_OF and b in GROUP_OF:
            (within if GROUP_OF[a] == GROUP_OF[b] else between).append(value)
    if len(within) < 4 or len(between) < 20:
        return {"within_n": len(within), "between_n": len(between), "contrast": np.nan}
    return {"within_n": len(within), "between_n": len(between),
            "within_mean": float(np.mean(within)), "between_mean": float(np.mean(between)),
            "contrast": float(np.mean(within) - np.mean(between))}


def constraint_bins(frame: pd.DataFrame, bins: int = 5) -> pd.DataFrame:
    work = frame.dropna(subset=["observed_position_intolerance"]).copy()
    work["constraint_bin"] = (
        work.groupby("protein_id")["observed_position_intolerance"]
        .transform(lambda s: pd.qcut(s.rank(method="first"), bins,
                                     labels=[f"C{i + 1}" for i in range(bins)]
                                     ).astype(object)))
    return work


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    frame = constraint_bins(pd.read_parquet(DATA / "abundance.parquet"))
    labels = sorted(frame["constraint_bin"].dropna().unique())

    print("1  symmetric fraction across five constraint bins, experiment and predictors")
    rows = []
    targets = {"experiment": ("effect_rank", "measured"), **PREDICTORS}
    for name, (column, family) in targets.items():
        for label in labels:
            part = frame[frame["constraint_bin"] == label]
            result = symmetry(pair_table(leave_one_out(part, column)),
                              SEED + len(name) + len(label))
            rows.append({"target": name, "family": family, "constraint_bin": label,
                         **result})
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / "constraint_symmetry_by_target.csv", index=False)
    print(table.pivot_table(index="target", columns="constraint_bin",
                            values="fraction").round(3).to_string())

    print("\n2  constraint gradient per target (most constrained minus least)")
    gradient = []
    for name, (column, family) in targets.items():
        part = table[table["target"] == name].set_index("constraint_bin")
        if labels[0] not in part.index or labels[-1] not in part.index:
            continue
        high = part.at[labels[-1], "fraction"]
        low = part.at[labels[0], "fraction"]
        top = pair_table(leave_one_out(frame[frame["constraint_bin"] == labels[-1]], column))
        bottom = pair_table(leave_one_out(frame[frame["constraint_bin"] == labels[0]], column))
        draws = []
        if not top.empty and not bottom.empty:
            rng = np.random.default_rng(SEED + len(name))
            top_keep, bottom_keep = unordered(top), unordered(bottom)
            if len(top_keep) >= 20 and len(bottom_keep) >= 20:
                def one(table, keep, index):
                    f = np.array([table.at[(a, b), "mean"] for a, b in keep])[index]
                    r = np.array([table.at[(b, a), "mean"] for a, b in keep])[index]
                    se2 = np.array([table.at[(a, b), "se2"] + table.at[(b, a), "se2"]
                                    for a, b in keep])[index]
                    floor = se2.mean() / 4
                    sym = max(((f + r) / 2).var(ddof=1) - floor, 1e-12)
                    anti = max(((f - r) / 2).var(ddof=1) - floor, 1e-12)
                    return sym / (sym + anti)
                for _ in range(2000):
                    draws.append(one(top, top_keep,
                                     rng.integers(0, len(top_keep), len(top_keep)))
                                 - one(bottom, bottom_keep,
                                       rng.integers(0, len(bottom_keep), len(bottom_keep))))
        draws = np.array([d for d in draws if np.isfinite(d)])
        gradient.append({"target": name, "family": family,
                         "least_constrained": low, "most_constrained": high,
                         "gradient": high - low,
                         "ci_low": float(np.percentile(draws, 2.5)) if len(draws) else np.nan,
                         "ci_high": float(np.percentile(draws, 97.5)) if len(draws) else np.nan})
    gradient_table = pd.DataFrame(gradient).sort_values("gradient", ascending=False)
    gradient_table.to_csv(RESULTS / "constraint_gradient_by_target.csv", index=False)
    print(gradient_table.round(3).to_string(index=False))

    print("\n3  do the classical exchange groups emerge with constraint rather than burial?")
    blocks = []
    for label in labels:
        part = frame[frame["constraint_bin"] == label]
        blocks.append({"axis": "constraint", "level": label,
                       **block_contrast(pair_table(leave_one_out(part, "effect_rank")))})
    for level in ("exposed", "intermediate", "buried"):
        part = frame[frame["burial_class"] == level]
        blocks.append({"axis": "burial", "level": level,
                       **block_contrast(pair_table(leave_one_out(part, "effect_rank")))})
    block_table = pd.DataFrame(blocks)
    block_table.to_csv(RESULTS / "constraint_block_contrast.csv", index=False)
    print(block_table[["axis", "level", "within_mean", "between_mean",
                       "contrast"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()

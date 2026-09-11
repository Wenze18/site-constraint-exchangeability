from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROJECT = Path(__file__).resolve().parent.parent
DATA = PROJECT / "data"
RESULTS = PROJECT / "results"
SEED = 20260908
PREDICTORS = {"SPURS": "structure_sequence_expert_rank",
              "SaProt-DeltaG": "structure_language_expert_rank",
              "ThermoMPNN": "inverse_folding_expert_rank",
              "Pythia": "stability_language_expert_rank",
              "ESM-2": "sequence_language_expert_rank"}


def marginal_scale(frame: pd.DataFrame, minimum: int = 25) -> dict:
    work = frame.dropna(subset=["effect_rank"]).copy()
    total = work.groupby(["protein_id", "position"])["effect_rank"].transform("sum")
    count = work.groupby(["protein_id", "position"])["effect_rank"].transform("size")
    keep = count >= 4
    work, total, count = work[keep], total[keep], count[keep]
    work["contrast"] = work["effect_rank"] - (total - work["effect_rank"]) / (count - 1)
    table = (work.groupby(["reference_residue", "alternate_residue"])["contrast"]
             .agg(mean="mean", n="size").dropna())
    table = table[table["n"] >= minimum]
    scale = {}
    for a, b in table.index:
        if (b, a) not in table.index:
            continue
        key = tuple(sorted((a, b)))
        if key in scale:
            continue
        weights = np.array([table.at[(a, b), "n"], table.at[(b, a), "n"]], float)
        means = np.array([table.at[(a, b), "mean"], table.at[(b, a), "mean"]])
        scale[key] = float(np.average(means, weights=weights))
    return scale


def attach(block: pd.DataFrame, scale: dict) -> pd.DataFrame:
    default = float(np.mean(list(scale.values())))
    block = block.copy()
    block["exchange"] = [scale.get(tuple(sorted((str(a), str(b)))), default)
                         for a, b in zip(block["reference_residue"],
                                         block["alternate_residue"])]
    return block


def residual(values: pd.Series, nuisance: pd.Series) -> np.ndarray:
    y = values.rank().to_numpy(float)
    x = nuisance.rank().to_numpy(float)
    if np.ptp(x) == 0:
        return y - y.mean()
    slope = np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1)
    return y - y.mean() - slope * (x - x.mean())


def bootstrap(values: np.ndarray, groups: np.ndarray, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    blocks = [values[groups == g] for g in np.unique(groups)]
    index = rng.integers(0, len(blocks), (3000, len(blocks)))
    draws = np.array([np.concatenate([blocks[j] for j in row]).mean() for row in index])
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def assess(block: pd.DataFrame, label: str, minimum: int = 15) -> list[dict]:
    grouping = "homology_cluster" if "homology_cluster" in block else "protein_id"
    records: dict[str, list] = {}
    groups: dict[str, list] = {}
    for (_, group_id), part in block.groupby(["assay_id", grouping], sort=False):
        clean = part.dropna(subset=["effect_rank", "exchange"])
        if len(clean) < minimum or clean["effect_rank"].nunique() < 5 \
                or clean["exchange"].nunique() < 3:
            continue
        marginal = spearmanr(clean["effect_rank"], clean["exchange"]).statistic
        if not np.isfinite(marginal):
            continue
        records.setdefault("marginal", []).append(marginal)
        groups.setdefault("marginal", []).append(group_id)
        for name, column in PREDICTORS.items():
            usable = clean.dropna(subset=[column])
            if len(usable) < minimum or usable[column].nunique() < 5:
                continue
            absorption = spearmanr(usable[column], usable["exchange"]).statistic
            y = residual(usable["effect_rank"], usable[column])
            x = residual(usable["exchange"], usable[column])
            partial = np.corrcoef(x, y)[0, 1] if np.std(x) > 0 and np.std(y) > 0 else np.nan
            if np.isfinite(absorption):
                records.setdefault(f"absorption::{name}", []).append(absorption)
                groups.setdefault(f"absorption::{name}", []).append(group_id)
            if np.isfinite(partial):
                records.setdefault(f"partial::{name}", []).append(partial)
                groups.setdefault(f"partial::{name}", []).append(group_id)
    rows = []
    for key, values in records.items():
        values = np.array(values)
        keys = np.array(groups[key])
        if len(values) < 12:
            continue
        low, high = bootstrap(values, keys, SEED + len(key))
        quantity, _, predictor = key.partition("::")
        rows.append({"endpoint_block": label, "quantity": quantity,
                     "predictor": predictor or "-", "assays": len(values),
                     "groups": len(np.unique(keys)), "estimate": float(values.mean()),
                     "ci_low": low, "ci_high": high})
    return rows


def abundance_out_of_fold(frame: pd.DataFrame, folds: int = 5) -> pd.DataFrame:
    proteins = np.array(sorted(frame["protein_id"].unique()))
    assignment = pd.Series(
        np.random.default_rng(SEED).permutation(len(proteins)) % folds, index=proteins)
    parts = []
    for fold in range(folds):
        held = assignment[assignment == fold].index
        scale = marginal_scale(frame[~frame["protein_id"].isin(held)])
        parts.append(attach(frame[frame["protein_id"].isin(held)], scale))
    return pd.concat(parts)


def main() -> None:
    abundance = pd.read_parquet(DATA / "abundance.parquet")
    scale = marginal_scale(abundance)
    print(f"frozen marginal scale: {len(scale)} of 190 pairs")

    biochemical = pd.read_parquet(DATA / "biochemical.parquet")
    blocks = {
        "abundance (out of fold)": abundance_out_of_fold(abundance),
        "stability, VenusMutHub": attach(pd.read_parquet(DATA / "stability.parquet"), scale),
    }
    for endpoint in ("stability", "activity", "ppi_binding", "selectivity", "dti_binding"):
        part = biochemical[biochemical["endpoint_class"] == endpoint]
        if len(part) < 300:
            continue
        blocks[f"{endpoint}, broad benchmark"] = attach(part, scale)

    rows = []
    for label, block in blocks.items():
        found = assess(block, label)
        rows.extend(found)
        print(f"  {label}: {len(block)} variants, {len(found)} estimates")
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / "N3_endpoint_absorption.csv", index=False)

    print("\nmarginal: how well the exchange value alone orders the outcome")
    print(table.query("quantity == 'marginal'")[
        ["endpoint_block", "assays", "groups", "estimate", "ci_low", "ci_high"]
    ].round(4).to_string(index=False))
    for quantity, caption in (("absorption", "absorption: predictor score vs exchange value"),
                              ("partial", "partial: exchange vs outcome, predictor removed")):
        print(f"\n{caption}")
        pivot = table.query("quantity == @quantity").pivot_table(
            index="endpoint_block", columns="predictor", values="estimate")
        print(pivot.round(3).to_string())


if __name__ == "__main__":
    main()

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DATA = PROJECT / "data"
RESULTS = PROJECT / "results"
SEED = 20260908

core_spec = importlib.util.spec_from_file_location("r2r5", HERE / "03_exchange_matrix_and_validation.py")
core = importlib.util.module_from_spec(core_spec)
core_spec.loader.exec_module(core)
fast_spec = importlib.util.spec_from_file_location("fast", HERE / "06_bilinear_fit.py")
fast = importlib.util.module_from_spec(fast_spec)
fast_spec.loader.exec_module(fast)

FEATURES_TWO = ["delta_hydrophobicity", "delta_volume"]
FEATURES_FOUR = FEATURES_TWO + ["delta_charge", "delta_polarity"]


def absolute(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    return np.column_stack([np.ones(len(frame))]
                           + [frame[c].abs().to_numpy(float) for c in columns])


def fit_physicochemical(contrasts: pd.DataFrame, columns: list[str]) -> np.ndarray:
    design = absolute(contrasts, columns)
    target = contrasts["contrast"].to_numpy(float)
    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    return coefficients


def main() -> None:
    abundance = pd.read_parquet(DATA / "abundance.parquet")
    contrasts = core.leave_one_out(abundance)
    print(f"fitting on {len(contrasts)} within-position contrasts from "
          f"{abundance['protein_id'].nunique()} domains")

    measured = {pair[0]: value for pair, value in
                core.symmetric_scale(contrasts, minimum=25).items()}
    values, mask = core.exchange_matrix(abundance, None, 25)
    mu, alpha, factors, weights = fast.fit(values, mask, 1)
    interaction = (factors * weights) @ factors.T
    axis = dict(zip(core.ORDER, factors[:, 0]))
    rank_one = {tuple(sorted((a, b))): float(mu + alpha[i] + alpha[j] + interaction[i, j])
                for i, a in enumerate(core.ORDER) for j, b in enumerate(core.ORDER) if i < j}
    axis_scale = {tuple(sorted((a, b))): -abs(axis[a] - axis[b])
                  for a in core.ORDER for b in core.ORDER if a < b}
    coefficients_two = fit_physicochemical(contrasts, FEATURES_TWO)
    coefficients_four = fit_physicochemical(contrasts, FEATURES_FOUR)
    print("  two-parameter coefficients (intercept, |dH|, |dV|):",
          np.round(coefficients_two, 5))
    print("  four-parameter coefficients:", np.round(coefficients_four, 5))

    matrices = pd.read_csv(DATA / "substitution_matrices.csv")
    fixed = {name: {tuple(sorted((r.residue_a, r.residue_b))): r.score
                    for r in part.itertuples()}
             for name, part in matrices.groupby("matrix")}
    shared = sorted(set(fixed["BLOSUM62"]) & {tuple(sorted(k)) for k in core.BLOSUM})
    disagree = [k for k in shared if fixed["BLOSUM62"][k] != core.BLOSUM[k]]
    print(f"  BLOSUM62 cross-check against the hardcoded table: "
          f"{len(shared)} pairs, {len(disagree)} disagreements")

    blocks = {
        "cross-source stability": pd.read_parquet(DATA / "stability.parquet"),
        "broad benchmark stability": pd.read_parquet(DATA / "biochemical.parquet")
                                       .query("endpoint_class == 'stability'"),
    }
    default = float(np.mean(list(measured.values())))

    rows = []
    direct = []
    per_assay = []
    PUBLISHED = ["BENNER74", "BLOSUM45", "BLOSUM62", "BLOSUM80", "PAM250"]
    for label, block in blocks.items():
        block = block.copy()
        pairs = [tuple(sorted((str(a), str(b)))) for a, b in
                 zip(block["reference_residue"], block["alternate_residue"])]
        columns: dict[str, np.ndarray] = {}
        for name, table in fixed.items():
            columns[name] = np.array([table.get(p, np.nan) for p in pairs])
        columns["Grantham distance"] = -block["grantham_distance"].to_numpy(float)
        columns["physicochemical, 2 parameters"] = absolute(block, FEATURES_TWO) @ coefficients_two
        columns["physicochemical, 4 parameters"] = absolute(block, FEATURES_FOUR) @ coefficients_four
        columns["R4 axis distance, 20 numbers"] = np.array(
            [axis_scale.get(p, np.nan) for p in pairs])
        columns["rank-one reconstruction, 41 numbers"] = np.array(
            [rank_one.get(p, np.nan) for p in pairs])
        columns["measured matrix, 190 numbers"] = np.array(
            [measured.get(p, default) for p in pairs])
        names = list(columns)
        for i, name in enumerate(names):
            block[f"s{i}"] = columns[name]

        grouping = "homology_cluster" if "homology_cluster" in block else "protein_id"
        collected: dict[str, list] = {name: [] for name in names}
        groups = []
        for (_, group_id), part in block.groupby(["assay_id", grouping], sort=False):
            clean = part.dropna(subset=["effect_rank"] + [f"s{i}" for i in range(len(names))])
            if len(clean) < 15 or clean["effect_rank"].nunique() < 5:
                continue
            statistics = {name: spearmanr(clean["effect_rank"], clean[f"s{i}"]).statistic
                          for i, name in enumerate(names)}
            if not all(np.isfinite(v) for v in statistics.values()):
                continue
            for name, value in statistics.items():
                collected[name].append(value)
            groups.append(group_id)
        groups = np.array(groups)
        print(f"  {label}: {len(groups)} assays, {len(np.unique(groups))} groups")

        dump = pd.DataFrame({name: collected[name] for name in names})
        dump.insert(0, "resampling_group", groups)
        dump.insert(0, "external_block", label)
        per_assay.append(dump)

        def interval(sample, seed):
            rng = np.random.default_rng(seed)
            parts = [sample[groups == g] for g in np.unique(groups)]
            index = rng.integers(0, len(parts), (4000, len(parts)))
            draws = np.array([np.concatenate([parts[j] for j in row]).mean()
                              for row in index])
            return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))

        full = np.array(collected["measured matrix, 190 numbers"])
        for name in names:
            sample = np.array(collected[name])
            low, high = interval(sample, SEED)
            difference = full - sample
            dlow, dhigh = interval(difference, SEED + 1)
            rows.append({"external_block": label, "scale": name,
                         "assays": len(sample), "groups": len(np.unique(groups)),
                         "macro_spearman": float(sample.mean()),
                         "ci_low": low, "ci_high": high,
                         "full_matrix_minus_this": float(difference.mean()),
                         "diff_ci_low": dlow, "diff_ci_high": dhigh})

        best = max(PUBLISHED, key=lambda n: float(np.mean(collected[n])))
        reference = np.array(collected[best])
        for name in ("rank-one reconstruction, 41 numbers",
                     "measured matrix, 190 numbers",
                     "R4 axis distance, 20 numbers"):
            sample = np.array(collected[name])
            difference = sample - reference
            dlow, dhigh = interval(difference, SEED + 2)
            direct.append({"external_block": label, "scale": name,
                           "strongest_published": best,
                           "published_macro_spearman": float(reference.mean()),
                           "scale_macro_spearman": float(sample.mean()),
                           "difference": float(difference.mean()),
                           "diff_ci_low": dlow, "diff_ci_high": dhigh})
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / "N6_baseline_ladder.csv", index=False)
    pd.concat(per_assay, ignore_index=True).to_csv(
        RESULTS / "N6c_per_assay_spearman.csv", index=False)
    direct_table = pd.DataFrame(direct)
    direct_table.to_csv(RESULTS / "N6b_compact_vs_strongest_published.csv", index=False)
    print("\ndirect comparison against the strongest published matrix on each resource")
    print(direct_table.round(4).to_string(index=False))
    for label in blocks:
        print(f"\n{label}")
        print(table[table["external_block"] == label]
              .drop(columns=["external_block", "assays", "groups"])
              .sort_values("macro_spearman", ascending=False)
              .round(4).to_string(index=False))


if __name__ == "__main__":
    main()

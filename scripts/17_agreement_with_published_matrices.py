from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DATA = PROJECT / "data"
RESULTS = PROJECT / "results"
SEED = 20260908
DRAWS = 2000

spec = importlib.util.spec_from_file_location("blosum", HERE / "16_blosum_and_residual.py")
core = importlib.util.module_from_spec(spec)
sys.modules["blosum"] = core
spec.loader.exec_module(core)


def published() -> dict[str, dict[tuple[str, str], float]]:
    table = pd.read_csv(DATA / "substitution_matrices.csv")
    out: dict[str, dict[tuple[str, str], float]] = {}
    for name, part in table.groupby("matrix"):
        lookup: dict[tuple[str, str], float] = {}
        for row in part.itertuples():
            lookup[(row.residue_a, row.residue_b)] = float(row.score)
            lookup[(row.residue_b, row.residue_a)] = float(row.score)
        out[str(name)] = lookup
    return out


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    frame = core.constraint_bins(pd.read_parquet(DATA / "abundance.parquet"))
    matrices = published()
    rng = np.random.default_rng(SEED)
    rows = []
    for label in sorted(frame["constraint_bin"].dropna().unique()):
        components = core.pair_components(
            core.leave_one_out(frame[frame["constraint_bin"] == label], "effect_rank"))
        if len(components) < 20:
            continue
        index = rng.integers(0, len(components), (DRAWS, len(components)))
        measured = components["symmetric"].to_numpy()
        for name, lookup in sorted(matrices.items()):
            score = np.array([lookup[(a, b)] for a, b
                              in zip(components["a"], components["b"])], float)
            rho = spearmanr(measured, score).statistic
            draws = np.array([spearmanr(measured[i], score[i]).statistic for i in index])
            rows.append({"matrix": name, "constraint_bin": label,
                         "pairs": len(components), "spearman": float(rho),
                         "ci_low": float(np.nanpercentile(draws, 2.5)),
                         "ci_high": float(np.nanpercentile(draws, 97.5))})
        print(f"  {label}: {len(components)} pairs", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / "X4_agreement_with_published_matrices.csv", index=False)
    wide = table.pivot(index="matrix", columns="constraint_bin", values="spearman")
    print()
    print(wide.round(4).to_string())
    print()
    for name in wide.index:
        row = wide.loc[name]
        print(f"  {name:18s} rise C1 to C4 {row['C4'] - row['C1']:+.3f}, "
              f"fall C4 to C5 {row['C5'] - row['C4']:+.3f}")


if __name__ == "__main__":
    main()

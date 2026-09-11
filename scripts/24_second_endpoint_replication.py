from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
RESULTS = PROJECT / "results"
spec = importlib.util.spec_from_file_location("core", HERE / "03_exchange_matrix_and_validation.py")
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)
SEED = core.SEED

STRATA = {
    "buried": lambda frame: frame["burial_class"] == "buried",
    "exposed": lambda frame: frame["burial_class"] == "exposed",
    "exposed + intermediate":
        lambda frame: frame["burial_class"].isin(["exposed", "intermediate"]),
}


def pair_table(contrasts: pd.DataFrame, minimum: int) -> pd.DataFrame:
    table = (contrasts.groupby(["reference_residue", "alternate_residue"])["contrast"]
             .agg(mean="mean", var="var", n="size").dropna())
    return table[table["n"] >= minimum]


def symmetric(table: pd.DataFrame, seed: int, draws: int = 2500) -> dict:
    forward, reverse, se2 = core.paired_arrays(table)
    if len(forward) < 8:
        return {"pairs": len(forward), "fraction": np.nan,
                "ci_low": np.nan, "ci_high": np.nan}
    rng = np.random.default_rng(seed)
    sample = np.array([core.symmetric_fraction(forward[i], reverse[i], se2[i])
                       for i in (rng.integers(0, len(forward), len(forward))
                                 for _ in range(draws))])
    return {"pairs": len(forward),
            "fraction": core.symmetric_fraction(forward, reverse, se2),
            "ci_low": float(np.percentile(sample, 2.5)),
            "ci_high": float(np.percentile(sample, 97.5))}


def values_of(table: pd.DataFrame) -> dict:
    out: dict = {}
    for a, b in table.index:
        if (b, a) not in table.index:
            continue
        key = tuple(sorted((a, b)))
        if key in out:
            continue
        weights = np.array([table.at[(a, b), "n"], table.at[(b, a), "n"]], float)
        means = np.array([table.at[(a, b), "mean"], table.at[(b, a), "mean"]])
        out[key] = float(np.average(means, weights=weights))
    return out


def main() -> None:
    abundance = pd.read_parquet(PROJECT / "data" / "abundance.parquet")
    biochemical = pd.read_parquet(PROJECT / "data" / "biochemical.parquet")
    stability = biochemical[biochemical["endpoint_class"] == "stability"]
    first = core.leave_one_out(abundance)
    second = core.leave_one_out(stability)

    rows, agreement = [], []
    for minimum in (8, 10):
        for name, selector in STRATA.items():
            a = pair_table(first[selector(first)], 25)
            b = pair_table(second[selector(second)], minimum)
            for dataset, table in (("abundance (503 domains)", a),
                                   ("stability (150 parents)", b)):
                result = symmetric(table, SEED + len(name) + minimum)
                rows.append({"stability_threshold": minimum, "stratum": name,
                             "dataset": dataset, **result})
            first_values, second_values = values_of(a), values_of(b)
            shared = sorted(set(first_values) & set(second_values))
            if len(shared) < 8:
                agreement.append({"stability_threshold": minimum, "stratum": name,
                                  "shared_pairs": len(shared), "spearman": np.nan,
                                  "ci_low": np.nan, "ci_high": np.nan})
                continue
            x = np.array([first_values[k] for k in shared])
            y = np.array([second_values[k] for k in shared])
            rng = np.random.default_rng(SEED + minimum)
            draws = []
            for _ in range(3000):
                index = rng.integers(0, len(shared), len(shared))
                if len(np.unique(x[index])) < 3:
                    continue
                draws.append(spearmanr(x[index], y[index]).statistic)
            draws = np.array([d for d in draws if np.isfinite(d)])
            agreement.append({"stability_threshold": minimum, "stratum": name,
                              "shared_pairs": len(shared),
                              "spearman": float(spearmanr(x, y).statistic),
                              "ci_low": float(np.percentile(draws, 2.5)),
                              "ci_high": float(np.percentile(draws, 97.5))})

    replication = pd.DataFrame(rows)
    replication.to_csv(RESULTS / "second_endpoint_replication.csv", index=False)
    table = pd.DataFrame(agreement)
    table.to_csv(RESULTS / "second_endpoint_agreement.csv", index=False)
    print(replication.round(3).to_string(index=False))
    print()
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

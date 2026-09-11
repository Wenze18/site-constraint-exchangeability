from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
DATA = PROJECT / "data"
RESULTS = PROJECT / "results"
SEED = 20260908
FLOOR_COUNT = 27
NULL_DRAWS = 200
REPLICATES = 2500
GRADIENT_DRAWS = 2000


def load(name: str):
    spec = importlib.util.spec_from_file_location(name.split("_")[0], PROJECT / "scripts" / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


core = load("02_constraint_reframe.py")
saturation = load("10_saturation_test.py")


def floor_pairs(frame: pd.DataFrame, count: int = FLOOR_COUNT) -> list[tuple[str, str]]:
    values = saturation.pair_values(frame[frame["constraint_bin"] == "C5"])
    cross = [(pair, value) for pair, value in values.items()
             if pair[0] in saturation.GROUP_OF and pair[1] in saturation.GROUP_OF
             and saturation.GROUP_OF[pair[0]] != saturation.GROUP_OF[pair[1]]]
    cross.sort(key=lambda item: item[1])
    return [pair for pair, _ in cross[:count]]


def cross_class_pairs(frame: pd.DataFrame) -> list[tuple[str, str]]:
    values = saturation.pair_values(frame[frame["constraint_bin"] == "C5"])
    return [pair for pair in values
            if pair[0] in saturation.GROUP_OF and pair[1] in saturation.GROUP_OF
            and saturation.GROUP_OF[pair[0]] != saturation.GROUP_OF[pair[1]]]


def drop(table: pd.DataFrame, pairs) -> pd.DataFrame:
    directed = set()
    for a, b in pairs:
        directed.add((a, b))
        directed.add((b, a))
    keep = [index for index in table.index if index not in directed]
    return table.loc[keep]


def share(table: pd.DataFrame, keep, index) -> float:
    f = np.array([table.at[(a, b), "mean"] for a, b in keep])[index]
    r = np.array([table.at[(b, a), "mean"] for a, b in keep])[index]
    se2 = np.array([table.at[(a, b), "se2"] + table.at[(b, a), "se2"]
                    for a, b in keep])[index]
    floor = se2.mean() / 4
    sym = max(((f + r) / 2).var(ddof=1) - floor, 1e-12)
    anti = max(((f - r) / 2).var(ddof=1) - floor, 1e-12)
    return sym / (sym + anti)


def gradient_interval(top: pd.DataFrame, bottom: pd.DataFrame, seed: int) -> tuple:
    rng = np.random.default_rng(seed)
    top_keep, bottom_keep = core.unordered(top), core.unordered(bottom)
    if len(top_keep) < 20 or len(bottom_keep) < 20:
        return np.nan, np.nan
    draws = [share(top, top_keep, rng.integers(0, len(top_keep), len(top_keep)))
             - share(bottom, bottom_keep, rng.integers(0, len(bottom_keep), len(bottom_keep)))
             for _ in range(GRADIENT_DRAWS)]
    draws = np.array([d for d in draws if np.isfinite(d)])
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def curve(tables: dict, labels) -> dict:
    out = {}
    for label in labels:
        table = tables[label]
        keep = core.unordered(table)
        out[label] = {"pairs": len(keep),
                      "fraction": share(table, keep, np.arange(len(keep)))}
    return out


def main() -> None:
    frame = core.constraint_bins(pd.read_parquet(DATA / "abundance.parquet"))
    labels = sorted(frame["constraint_bin"].dropna().unique())

    removed = floor_pairs(frame)
    print(f"floor-affected set, defined at C5 from measured values alone: {len(removed)} pairs")
    print("  " + ", ".join(f"{a}{b}" for a, b in removed))

    full = {label: core.pair_table(
        core.leave_one_out(frame[frame["constraint_bin"] == label], "effect_rank"))
        for label in labels}
    reduced = {label: drop(full[label], removed) for label in labels}

    rows = []
    for name, tables in (("all pairs", full), ("floor set removed", reduced)):
        values = curve(tables, labels)
        for label in labels:
            keep = core.unordered(tables[label])
            rng = np.random.default_rng(SEED + len(label) + len(name))
            draws = np.array([share(tables[label], keep,
                                    rng.integers(0, len(keep), len(keep)))
                              for _ in range(REPLICATES)])
            rows.append({"variant": name, "constraint_bin": label,
                         "pairs": values[label]["pairs"],
                         "fraction": values[label]["fraction"],
                         "ci_low": float(np.percentile(draws, 2.5)),
                         "ci_high": float(np.percentile(draws, 97.5))})
        low, high = values[labels[0]]["fraction"], values[labels[-1]]["fraction"]
        ci = gradient_interval(tables[labels[-1]], tables[labels[0]], SEED + len(name))
        rows.append({"variant": name, "constraint_bin": "gradient",
                     "pairs": np.nan, "fraction": high - low,
                     "ci_low": ci[0], "ci_high": ci[1]})
        print(f"\n{name}")
        for label in labels:
            print(f"  {label}  {values[label]['fraction']:.3f}"
                  f"  ({values[label]['pairs']} pairs)")
        print(f"  gradient {high - low:+.3f}  [{ci[0]:+.3f}, {ci[1]:+.3f}]")

    pool = cross_class_pairs(frame)
    rng = np.random.default_rng(SEED)
    null = []
    for _ in range(NULL_DRAWS):
        picked = [pool[i] for i in rng.choice(len(pool), FLOOR_COUNT, replace=False)]
        tables = {label: drop(full[label], picked) for label in labels}
        values = curve(tables, labels)
        null.append(values[labels[-1]]["fraction"] - values[labels[0]]["fraction"])
    null = np.array(null)
    print(f"\nnull, {NULL_DRAWS} random removals of {FLOOR_COUNT} cross-class pairs")
    print(f"  gradient median {np.median(null):+.3f}"
          f"  [{np.percentile(null, 2.5):+.3f}, {np.percentile(null, 97.5):+.3f}]")

    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / "X7_floor_censoring_stress_test.csv", index=False)
    pd.DataFrame({"random_removal_gradient": null}).to_csv(
        RESULTS / "X7_random_removal_null.csv", index=False)
    pd.DataFrame({"residue_a": [a for a, _ in removed],
                  "residue_b": [b for _, b in removed]}).to_csv(
        RESULTS / "X7_floor_pair_set.csv", index=False)
    print("\nwrote X7_floor_censoring_stress_test.csv, X7_random_removal_null.csv "
          "and X7_floor_pair_set.csv")


if __name__ == "__main__":
    main()

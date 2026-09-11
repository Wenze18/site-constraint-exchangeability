from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DATA = PROJECT / "data"
RESULTS = PROJECT / "results"
MIN_CELL = 15
BINS = [f"C{i + 1}" for i in range(5)]
DISPLAY = {"small_polar": "small polar"}

spec = importlib.util.spec_from_file_location("core", HERE / "02_constraint_reframe.py")
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)


def pair_value(table: pd.DataFrame, a: str, b: str) -> float:
    weights = np.array([table.at[(a, b), "n"], table.at[(b, a), "n"]], float)
    means = np.array([table.at[(a, b), "mean"], table.at[(b, a), "mean"]], float)
    return float(np.average(means, weights=weights))


def check(name: str, stored_path: Path, rebuilt: pd.DataFrame, keys: list[str],
          column: str) -> None:
    if not stored_path.exists():
        return
    stored = pd.read_csv(stored_path).set_index(keys)
    new = rebuilt.set_index(keys)
    shared = stored.index.intersection(new.index)
    gap = (stored.loc[shared, column] - new.loc[shared, column]).abs().max()
    print(f"  {name}: {len(stored)} rows stored, {len(new)} rebuilt, "
          f"{len(shared)} in common, largest disagreement {gap:.8f}")
    if len(stored) != len(new) or gap > 1e-9:
        sys.exit(f"  {name} does not reproduce the stored copy")


def main() -> None:
    frame = core.constraint_bins(pd.read_parquet(DATA / "abundance.parquet"))
    rows = []
    for label in BINS:
        table = core.pair_table(
            core.leave_one_out(frame[frame["constraint_bin"] == label], "effect_rank"),
            MIN_CELL)
        within, between = [], []
        for a, b in core.unordered(table):
            if a not in core.GROUP_OF or b not in core.GROUP_OF:
                continue
            value = pair_value(table, a, b)
            if core.GROUP_OF[a] == core.GROUP_OF[b]:
                within.append(("".join(sorted((a, b))), core.GROUP_OF[a], value))
            else:
                between.append(value)
        base = float(np.mean(between))
        for pair, group, value in within:
            rows.append({"constraint_bin": label, "pair": pair,
                         "group": DISPLAY.get(group, group), "value": value - base})

    paths = pd.DataFrame(rows)
    classes = (paths.groupby(["constraint_bin", "group"])["value"]
               .agg(pairs="size", within_minus_between="mean").reset_index())
    print(f"{paths['pair'].nunique()} within-class pairs across {len(BINS)} quintiles")
    print(paths.pivot(index="pair", columns="constraint_bin", values="value")
          .round(3).to_string())

    check("per-class summary", RESULTS / "X5_within_class_contrast_by_group.csv", classes,
          ["constraint_bin", "group"], "within_minus_between")

    paths.to_csv(RESULTS / "X5_within_class_pair_paths.csv", index=False)
    classes.to_csv(RESULTS / "X5_within_class_contrast_by_group.csv", index=False)
    print("wrote the pair paths and the per-class summary")


if __name__ == "__main__":
    main()

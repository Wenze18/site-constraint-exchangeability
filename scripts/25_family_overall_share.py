from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DATA = PROJECT / "data"
RESULTS = PROJECT / "results"
SEED = 20260908
MIN_VARIANTS = 4000
MIN_CELL = 8

spec = importlib.util.spec_from_file_location("core", HERE / "02_constraint_reframe.py")
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)


def main() -> None:
    frame = pd.read_parquet(DATA / "abundance.parquet")
    rows = []
    for family, part in frame.groupby("protein_family"):
        if len(part) < MIN_VARIANTS:
            continue
        table = core.pair_table(core.leave_one_out(part, "effect_rank"), MIN_CELL)
        result = core.symmetry(table, SEED + len(str(family)))
        if pd.isna(result["fraction"]):
            continue
        rows.append({"family": family, "pairs": result["pairs"],
                     "symmetric_fraction": result["fraction"]})
    table = pd.DataFrame(rows).sort_values("symmetric_fraction").reset_index(drop=True)
    table.to_csv(RESULTS / "X9_family_overall_share.csv", index=False)
    print(f"{len(table)} families with at least {MIN_VARIANTS} measured variants")
    print(table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()

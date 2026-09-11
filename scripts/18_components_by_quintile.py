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

spec = importlib.util.spec_from_file_location("blosum", HERE / "16_blosum_and_residual.py")
core = importlib.util.module_from_spec(spec)
sys.modules["blosum"] = core
spec.loader.exec_module(core)


def main() -> None:
    frame = core.constraint_bins(pd.read_parquet(DATA / "abundance.parquet"))
    rows, summary = [], []
    for label in sorted(frame["constraint_bin"].dropna().unique()):
        components = core.pair_components(
            core.leave_one_out(frame[frame["constraint_bin"] == label], "effect_rank"))
        if len(components) < 20:
            continue
        for row in components.itertuples():
            rows.append({"constraint_bin": label, "pair": row.a + row.b,
                         "symmetric": row.symmetric,
                         "antisymmetric": abs(row.antisymmetric_ab)})
        summary.append({"constraint_bin": label, "pairs": len(components),
                        "symmetric_sd": float(components["symmetric"].std(ddof=1)),
                        "antisymmetric_sd": float(
                            components["antisymmetric_ab"].std(ddof=1))})
        print(f"  {label}: {len(components)} pairs", flush=True)
    table = pd.DataFrame(rows)
    spread = pd.DataFrame(summary)
    spread["symmetric_over_antisymmetric"] = (spread["symmetric_sd"]
                                              / spread["antisymmetric_sd"])
    spread.to_csv(RESULTS / "X6_component_spread_by_quintile.csv", index=False)
    print()
    print(spread.round(4).to_string(index=False))


if __name__ == "__main__":
    main()

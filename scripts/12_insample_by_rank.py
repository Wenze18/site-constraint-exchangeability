from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
core_spec = importlib.util.spec_from_file_location("core", HERE / "03_exchange_matrix_and_validation.py")
core = importlib.util.module_from_spec(core_spec)
core_spec.loader.exec_module(core)
fast_spec = importlib.util.spec_from_file_location("fast", HERE / "06_bilinear_fit.py")
fast = importlib.util.module_from_spec(fast_spec)
fast_spec.loader.exec_module(fast)


def main() -> None:
    frame = pd.read_parquet(PROJECT / "data" / "abundance.parquet")
    heldout = pd.read_csv(PROJECT / "results" / "R4_dimensionality_singlestart.csv")
    rows = []
    for label, burial, minimum in (("all positions", None, 25), ("buried", "buried", 15),
                                   ("exposed", "exposed", 15)):
        values, mask = core.exchange_matrix(frame, burial, minimum)
        pairs = [(i, j) for i in range(20) for j in range(i + 1, 20) if mask[i, j]]
        actual = np.array([values[i, j] for i, j in pairs])
        for rank in range(6):
            mu, alpha, factors, weights = fast.fit(values, mask, rank)
            predicted = core.predict_pairs(mu, alpha, factors, weights, pairs)
            sse = float(((actual - predicted) ** 2).sum())
            sst = float(((actual - actual.mean()) ** 2).sum())
            match = heldout[(heldout["stratum"] == label)
                            & (heldout["interaction_rank"] == rank)]
            rows.append({"stratum": label, "interaction_rank": rank,
                         "insample_r2": 1 - sse / sst,
                         "heldout_r2": float(match["heldout_r2"].iloc[0])
                         if len(match) else np.nan})
            print(f"  {label:14} rank {rank}: in-sample {1 - sse / sst:.4f}", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(PROJECT / "results" / "R4_insample_and_heldout.csv", index=False)
    print(f"wrote R4_insample_and_heldout.csv, {len(table)} rows")


if __name__ == "__main__":
    main()

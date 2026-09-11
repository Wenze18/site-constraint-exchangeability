from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROJECT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT / "results"
spec = importlib.util.spec_from_file_location(
    "r2r5", Path(__file__).resolve().parent / "03_exchange_matrix_and_validation.py")
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)
fast = importlib.util.spec_from_file_location(
    "fast", Path(__file__).resolve().parent / "06_bilinear_fit.py")
fastmod = importlib.util.module_from_spec(fast)
fast.loader.exec_module(fastmod)

ORDER = core.ORDER


def main() -> None:
    frame = pd.read_parquet(PROJECT / "data" / "abundance.parquet")
    rows, coordinates = [], []
    for label, burial, minimum in [("all positions", None, 25), ("buried", "buried", 15),
                                   ("exposed", "exposed", 15)]:
        values, mask = core.exchange_matrix(frame, burial, minimum)
        mu, alpha, factors, weights = fastmod.fit(values, mask, 1)
        axis = pd.Series(factors[:, 0], index=ORDER)
        if axis.corr(pd.Series([core.HYDROPATHY[r] for r in ORDER], index=ORDER)) < 0:
            axis = -axis
        replaceability = pd.Series(alpha, index=ORDER)
        pairs = [(i, j) for i in range(20) for j in range(i + 1, 20) if mask[i, j]]
        predicted = core.predict_pairs(mu, alpha, factors, weights, pairs)
        actual = np.array([values[i, j] for i, j in pairs])
        for name, series in (("additive (replaceability)", replaceability),
                             ("interaction axis (rank 1)", axis)):
            rows.append({
                "stratum": label, "component": name,
                "interaction_weight": float(weights[0]) if name.startswith("inter") else np.nan,
                "insample_r2": 1 - float(((actual - predicted) ** 2).sum()
                                         / ((actual - actual.mean()) ** 2).sum()),
                "spearman_hydropathy": float(spearmanr(
                    series.values, [core.HYDROPATHY[r] for r in series.index]).statistic),
                "spearman_volume": float(spearmanr(
                    series.values, [core.VOLUME[r] for r in series.index]).statistic),
                "spearman_abs_charge": float(spearmanr(
                    series.values, [abs(core.CHARGE[r]) for r in series.index]).statistic),
                "spearman_aromatic": float(spearmanr(
                    series.values, [core.AROMATIC[r] for r in series.index]).statistic),
                "ordered_low_to_high": " ".join(series.sort_values().index)})
        for residue in ORDER:
            coordinates.append({"stratum": label, "residue": residue,
                                "replaceability": float(replaceability[residue]),
                                "axis_coordinate": float(axis[residue])})
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / "R4_rank1_axis.csv", index=False)
    pd.DataFrame(coordinates).to_csv(RESULTS / "R4_residue_coordinates.csv", index=False)
    print(table.drop(columns=["ordered_low_to_high"]).round(3).to_string(index=False))
    print()
    for row in rows:
        print(f"{row['stratum']:<14} {row['component']:<26} "
              f"{row['ordered_low_to_high']}")


if __name__ == "__main__":
    main()

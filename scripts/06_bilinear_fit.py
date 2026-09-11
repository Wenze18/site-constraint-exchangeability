from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import spearmanr

PROJECT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT / "results"
spec = importlib.util.spec_from_file_location(
    "r2r5", Path(__file__).resolve().parent / "03_exchange_matrix_and_validation.py")
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)

SEED = core.SEED
ORDER = core.ORDER


def fit(values, mask, rank, ridge=1e-3, seed=SEED):
    n = values.shape[0]
    upper = [(i, j) for i in range(n) for j in range(i + 1, n) if mask[i, j]]
    rows = np.array([i for i, _ in upper])
    columns = np.array([j for _, j in upper])
    target = np.array([values[i, j] for i, j in upper])
    centre = np.eye(n) - np.ones((n, n)) / n

    def unpack(parameters):
        mu = parameters[0]
        alpha = centre @ parameters[1:1 + n]
        rest = parameters[1 + n:]
        if rank == 0:
            return mu, alpha, np.zeros((n, 0)), np.zeros(0)
        factors = centre @ rest[:n * rank].reshape(n, rank)
        norms = np.linalg.norm(factors, axis=0)
        norms[norms == 0] = 1.0
        return mu, alpha, factors / norms, rest[n * rank:]

    def residual(parameters):
        mu, alpha, factors, weights = unpack(parameters)
        prediction = mu + alpha[rows] + alpha[columns]
        if rank:
            prediction = prediction + ((factors * weights) @ factors.T)[rows, columns]
        return np.concatenate([target - prediction, np.sqrt(ridge) * parameters[1:]])

    mu0, alpha0 = core.fit_additive(values, mask)
    if rank:
        vectors, eigenvalues = core.warm_start(values, mask, mu0, alpha0, rank)
        start = np.concatenate([[mu0], alpha0, vectors.ravel(), eigenvalues])
    else:
        start = np.concatenate([[mu0], alpha0])
    outcome = least_squares(residual, start, method="trf", max_nfev=4000,
                            xtol=1e-8, ftol=1e-8, gtol=1e-8)
    mu, alpha, factors, weights = unpack(outcome.x)
    if rank:
        w, v = np.linalg.eigh((factors * weights) @ factors.T)
        order = np.argsort(np.abs(w))[::-1][:rank]
        factors, weights = v[:, order], w[order]
    return mu, alpha, factors, weights


def main() -> None:
    frame = pd.read_parquet(PROJECT / "data" / "abundance.parquet")
    skill_path = RESULTS / "R4_dimensionality_singlestart.csv"
    loading_path = RESULTS / "R4_component_loadings_singlestart.csv"
    skill_rows, loading_rows = [], []
    rng = np.random.default_rng(SEED)
    for label, burial, minimum in [("all positions", None, 25), ("buried", "buried", 15),
                                   ("exposed", "exposed", 15)]:
        values, mask = core.exchange_matrix(frame, burial, minimum)
        pairs = [(i, j) for i in range(20) for j in range(i + 1, 20) if mask[i, j]]
        print(f"{label}: {len(pairs)} of 190 pairs measured", flush=True)
        folds = rng.permutation(len(pairs)) % 5
        for rank in range(0, 6):
            predicted, actual = [], []
            for fold in range(5):
                train = mask.copy()
                held = [pairs[k] for k in range(len(pairs)) if folds[k] == fold]
                for i, j in held:
                    train[i, j] = train[j, i] = False
                mu, alpha, factors, weights = fit(values, train, rank)
                predicted.append(core.predict_pairs(mu, alpha, factors, weights, held))
                actual.append(np.array([values[i, j] for i, j in held]))
                print(f"  rank {rank} fold {fold} done", flush=True)
            predicted, actual = np.concatenate(predicted), np.concatenate(actual)
            sse = float(((actual - predicted) ** 2).sum())
            sst = float(((actual - actual.mean()) ** 2).sum())
            skill_rows.append({"stratum": label, "measured_pairs": len(pairs),
                               "model": "additive only" if rank == 0
                               else f"additive + rank {rank}",
                               "interaction_rank": rank, "heldout_r2": 1 - sse / sst})
            pd.DataFrame(skill_rows).to_csv(skill_path, index=False)
            print(f"  -> rank {rank}: held-out R2 = {1 - sse / sst:.4f}", flush=True)

        mu, alpha, factors, weights = fit(values, mask, 3)
        share = np.abs(weights) / np.abs(weights).sum()
        reference = pd.Series([core.HYDROPATHY[r] for r in ORDER], index=ORDER)
        entries = [("additive (replaceability)", pd.Series(alpha, index=ORDER), np.nan)]
        for k in range(factors.shape[1]):
            entries.append((f"interaction PC{k + 1}",
                            pd.Series(factors[:, k], index=ORDER), float(share[k])))
        for name, series, weight_share in entries:
            if name.startswith("interaction") and series.corr(reference) < 0:
                series = -series
            loading_rows.append({
                "stratum": label, "component": name, "weight_share": weight_share,
                "spearman_hydropathy": float(spearmanr(
                    series.values, [core.HYDROPATHY[r] for r in series.index]).statistic),
                "spearman_volume": float(spearmanr(
                    series.values, [core.VOLUME[r] for r in series.index]).statistic),
                "spearman_abs_charge": float(spearmanr(
                    series.values, [abs(core.CHARGE[r]) for r in series.index]).statistic),
                "spearman_aromatic": float(spearmanr(
                    series.values, [core.AROMATIC[r] for r in series.index]).statistic),
                "low_end": " ".join(series.sort_values().index[:4]),
                "high_end": " ".join(series.sort_values().index[-4:])})
        pd.DataFrame(loading_rows).to_csv(loading_path, index=False)
        print(pd.DataFrame(loading_rows).tail(4).round(3).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

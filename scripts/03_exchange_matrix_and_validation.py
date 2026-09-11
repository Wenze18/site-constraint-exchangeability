from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DATA = PROJECT / "data"
RESULTS = PROJECT / "results"
SEED = 20260908

HYDROPATHY = {"A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5,
              "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8,
              "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2}
VOLUME = {"S": 32.0, "R": 124.0, "L": 111.0, "P": 32.5, "T": 61.0, "A": 31.0, "V": 84.0,
          "G": 3.0, "I": 111.0, "F": 132.0, "Y": 136.0, "C": 55.0, "H": 96.0, "Q": 85.0,
          "N": 56.0, "K": 119.0, "D": 54.0, "E": 83.0, "M": 105.0, "W": 170.0}
CHARGE = {a: 0.0 for a in HYDROPATHY}
CHARGE.update({"D": -1.0, "E": -1.0, "K": 1.0, "R": 1.0})
AROMATIC = {a: (1.0 if a in "FYWH" else 0.0) for a in HYDROPATHY}
ORDER = list("ILVMFYWCAGSTNQDEKRHP")

BLOSUM62_TEXT = """
   A  R  N  D  C  Q  E  G  H  I  L  K  M  F  P  S  T  W  Y  V
A  4 -1 -2 -2  0 -1 -1  0 -2 -1 -1 -1 -1 -2 -1  1  0 -3 -2  0
R -1  5  0 -2 -3  1  0 -2  0 -3 -2  2 -1 -3 -2 -1 -1 -3 -2 -3
N -2  0  6  1 -3  0  0  0  1 -3 -3  0 -2 -3 -2  1  0 -4 -2 -3
D -2 -2  1  6 -3  0  2 -1 -1 -3 -4 -1 -3 -3 -1  0 -1 -4 -3 -3
C  0 -3 -3 -3  9 -3 -4 -3 -3 -1 -1 -3 -1 -2 -3 -1 -1 -2 -2 -1
Q -1  1  0  0 -3  5  2 -2  0 -3 -2  1  0 -3 -1  0 -1 -2 -1 -2
E -1  0  0  2 -4  2  5 -2  0 -3 -3  1 -2 -3 -1  0 -1 -3 -2 -2
G  0 -2  0 -1 -3 -2 -2  6 -2 -4 -4 -2 -3 -3 -2  0 -2 -2 -3 -3
H -2  0  1 -1 -3  0  0 -2  8 -3 -3 -1 -2 -1 -2 -1 -2 -2  2 -3
I -1 -3 -3 -3 -1 -3 -3 -4 -3  4  2 -3  1  0 -3 -2 -1 -3 -1  3
L -1 -2 -3 -4 -1 -2 -3 -4 -3  2  4 -2  2  0 -3 -2 -1 -2 -1  1
K -1  2  0 -1 -3  1  1 -2 -1 -3 -2  5 -1 -3 -1  0 -1 -3 -2 -2
M -1 -1 -2 -3 -1  0 -2 -3 -2  1  2 -1  5  0 -2 -1 -1 -1 -1  1
F -2 -3 -3 -3 -2 -3 -3 -3 -1  0  0 -3  0  6 -4 -2 -2  1  3 -1
P -1 -2 -2 -1 -3 -1 -1 -2 -2 -3 -3 -1 -2 -4  7 -1 -1 -4 -3 -2
S  1 -1  1  0 -1  0  0  0 -1 -2 -2  0 -1 -2 -1  4  1 -3 -2 -2
T  0 -1  0 -1 -1 -1 -1 -2 -2 -1 -1 -1 -1 -2 -1  1  5 -2 -2  0
W -3 -3 -4 -4 -2 -2 -3 -2 -2 -3 -2 -3 -1  1 -4 -3 -2 11  2 -3
Y -2 -2 -2 -3 -2 -1 -2 -3  2 -1 -1 -2 -1  3 -3 -2 -2  2  7 -1
V  0 -3 -3 -3 -1 -2 -2 -3 -3  3  1 -2  1 -1 -2 -2  0 -3 -1  4
"""


def blosum_table() -> dict[tuple[str, str], float]:
    lines = [ln for ln in BLOSUM62_TEXT.strip().splitlines() if ln.strip()]
    header = lines[0].split()
    out = {}
    for line in lines[1:]:
        parts = line.split()
        row = parts[0]
        for column, value in zip(header, parts[1:]):
            out[(row, column)] = float(value)
    return out


BLOSUM = blosum_table()


def leave_one_out(frame: pd.DataFrame, outcome: str = "effect_rank",
                  minimum: int = 4) -> pd.DataFrame:
    work = frame.dropna(subset=[outcome]).copy()
    total = work.groupby(["protein_id", "position"])[outcome].transform("sum")
    count = work.groupby(["protein_id", "position"])[outcome].transform("size")
    keep = count >= minimum
    work, total, count = work[keep], total[keep], count[keep]
    work["contrast"] = work[outcome] - (total - work[outcome]) / (count - 1)
    return work


def directed_table(contrasts: pd.DataFrame, extra: list[str] | None = None,
                   minimum: int = 15) -> pd.DataFrame:
    keys = ["reference_residue", "alternate_residue"] + (extra or [])
    table = (contrasts.groupby(keys)["contrast"]
             .agg(mean="mean", var="var", n="size").dropna())
    return table[table["n"] >= minimum]


def symmetric_scale(contrasts: pd.DataFrame, extra: list[str] | None = None,
                    minimum: int = 15) -> dict:
    table = directed_table(contrasts, extra, minimum)
    out: dict = {}
    for index, row in table.iterrows():
        index = index if isinstance(index, tuple) else (index,)
        a, b = index[0], index[1]
        rest = index[2:]
        reverse = (b, a) + rest
        if reverse not in table.index:
            continue
        key = (tuple(sorted((a, b))),) + rest
        if key in out:
            continue
        other = table.loc[reverse]
        weights = np.array([row["n"], other["n"]], float)
        out[key] = float(np.average([row["mean"], other["mean"]], weights=weights))
    return out


def symmetric_fraction(forward, reverse, se2) -> float:
    floor = np.mean(se2) / 4
    sym = max(((forward + reverse) / 2).var(ddof=1) - floor, 1e-12)
    anti = max(((forward - reverse) / 2).var(ddof=1) - floor, 1e-12)
    return sym / (sym + anti)


def paired_arrays(table: pd.DataFrame):
    seen, keep = set(), []
    for a, b in table.index:
        if (b, a) in table.index and (b, a) not in seen:
            seen.add((a, b))
            keep.append((a, b))
    if not keep:
        return np.array([]), np.array([]), np.array([])
    forward = np.array([table.at[(a, b), "mean"] for a, b in keep])
    reverse = np.array([table.at[(b, a), "mean"] for a, b in keep])
    se2 = np.array([table.at[(a, b), "var"] / table.at[(a, b), "n"]
                    + table.at[(b, a), "var"] / table.at[(b, a), "n"] for a, b in keep])
    return forward, reverse, se2


def cluster_bootstrap(values: np.ndarray, groups: np.ndarray, draws: int = 4000,
                      seed: int = SEED) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    blocks = [values[groups == g] for g in np.unique(groups)]
    index = rng.integers(0, len(blocks), (draws, len(blocks)))
    sample = np.array([np.concatenate([blocks[j] for j in row]).mean() for row in index])
    return float(np.percentile(sample, 2.5)), float(np.percentile(sample, 97.5))


TOKEN = re.compile(r"([A-Z])(\d+)([A-Z])")
R2_DESCRIPTORS = [("measured symmetric exchange", "exchange"),
                  ("BLOSUM62", "blosum"),
                  ("hydropathy change", "hydrophobicity"),
                  ("volume change", "volume"),
                  ("charge change", "charge"),
                  ("substitution count", "substitutions"),
                  ("five-expert consensus (reference)", "consensus")]


def rank_residual(target: pd.Series, nuisance: pd.Series) -> np.ndarray:
    y = target.rank().to_numpy(float)
    x = nuisance.rank().to_numpy(float)
    if np.ptp(x) == 0:
        return y - y.mean()
    slope = np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1)
    return y - y.mean() - slope * (x - x.mean())


def r2_score_variants() -> pd.DataFrame:
    multi = pd.read_parquet(DATA / "multi_mutant_stability_variants.parquet")
    abundance = pd.read_parquet(DATA / "abundance.parquet")
    flat = {pair[0]: value
            for pair, value in symmetric_scale(leave_one_out(abundance), minimum=25).items()}
    print(f"    exchange scale covers {len(flat)} of 190 unordered pairs")

    rows, dropped = [], 0
    for record in multi.itertuples():
        parts = TOKEN.findall(str(record.mutation))
        if not parts:
            dropped += 1
            continue
        totals = dict(exchange=0.0, blosum=0.0, hydrophobicity=0.0, volume=0.0, charge=0.0)
        missing = 0
        for wild, _, mutant in parts:
            key = tuple(sorted((wild, mutant)))
            if key in flat:
                totals["exchange"] += flat[key]
            else:
                missing += 1
            totals["blosum"] += BLOSUM.get((wild, mutant), 0.0)
            totals["hydrophobicity"] -= abs(HYDROPATHY[mutant] - HYDROPATHY[wild])
            totals["volume"] -= abs(VOLUME[mutant] - VOLUME[wild])
            totals["charge"] -= abs(CHARGE[mutant] - CHARGE[wild])
        if missing:
            dropped += 1
            continue
        rows.append({"protein_id": record.protein_id,
                     "observed": record.observed_stability_change,
                     "count": float(record.mutation_count),
                     "substitutions": -float(record.mutation_count),
                     "consensus": record.five_expert_rank_consensus, **totals})
    table = pd.DataFrame(rows).dropna(subset=["observed"])
    print(f"    {len(table)} of {len(multi)} variants scored "
          f"({dropped} dropped for an unmeasured pair)")
    return table


def r2_multimutant() -> tuple[pd.DataFrame, pd.DataFrame]:
    table = r2_score_variants()
    out, paired = [], []
    for floor in (4, 6, 8):
        sizes = table["protein_id"].value_counts()
        block = table[table["protein_id"].isin(sizes[sizes >= floor].index)]
        usable = [p for p, g in block.groupby("protein_id") if g["observed"].nunique() >= 3]
        block = block[block["protein_id"].isin(usable)]
        held: dict[str, dict[str, float]] = {}
        for name, column in R2_DESCRIPTORS:
            raw, partial, labels = [], [], []
            for protein, group in block.groupby("protein_id"):
                clean = group.dropna(subset=[column])
                if len(clean) < floor or clean[column].nunique() < 3:
                    continue
                statistic = spearmanr(clean["observed"], clean[column]).statistic
                if not np.isfinite(statistic):
                    continue
                raw.append(statistic)
                labels.append(protein)
                held.setdefault(protein, {})[column] = statistic
                if column == "substitutions" or clean["count"].nunique() < 2:
                    partial.append(np.nan)
                    continue
                y = rank_residual(clean["observed"], clean["count"])
                x = rank_residual(clean[column], clean["count"])
                partial.append(np.corrcoef(x, y)[0, 1] if np.std(x) > 0 else np.nan)
            if len(raw) < 8:
                continue
            values = np.array(raw)
            low, high = cluster_bootstrap(values, np.array(labels), seed=SEED + len(name))
            residualised = np.array([v for v in partial if np.isfinite(v)])
            out.append({"minimum_variants_per_protein": floor, "descriptor": name,
                        "proteins": len(values), "variants": len(block),
                        "macro_spearman": float(values.mean()),
                        "ci_low": low, "ci_high": high,
                        "macro_partial_given_count": float(residualised.mean())
                        if len(residualised) else np.nan})
        shared = [p for p, d in held.items() if "exchange" in d and "blosum" in d]
        if len(shared) >= 8:
            difference = np.array([held[p]["exchange"] - held[p]["blosum"] for p in shared])
            low, high = cluster_bootstrap(difference, np.array(shared), seed=SEED + 3)
            paired.append({"minimum_variants_per_protein": floor,
                           "comparison": "exchange minus BLOSUM62, per protein",
                           "proteins": len(shared), "difference": float(difference.mean()),
                           "ci_low": low, "ci_high": high})

    sizes = table["protein_id"].value_counts()
    pooled_block = table[table["protein_id"].isin(sizes[sizes >= 4].index)].copy()
    pooled_block = pooled_block[pooled_block.groupby("protein_id")["observed"]
                                .transform("nunique") >= 3]
    within = pooled_block.groupby("protein_id")
    y = np.concatenate([g["observed"].rank(pct=True).to_numpy() - 0.5
                        for _, g in within])
    labels = np.concatenate([[p] * len(g) for p, g in within])
    for name, column in R2_DESCRIPTORS:
        x = np.concatenate([g[column].rank(pct=True).to_numpy() - 0.5 for _, g in within])
        good = np.isfinite(x) & np.isfinite(y)
        if good.sum() < 100:
            continue
        statistic = float(np.corrcoef(x[good], y[good])[0, 1])
        rng = np.random.default_rng(SEED + len(name))
        unique = np.unique(labels[good])
        blocks = [(x[good][labels[good] == p], y[good][labels[good] == p]) for p in unique]
        draws = []
        for _ in range(2000):
            pick = rng.integers(0, len(blocks), len(blocks))
            xs = np.concatenate([blocks[j][0] for j in pick])
            ys = np.concatenate([blocks[j][1] for j in pick])
            draws.append(np.corrcoef(xs, ys)[0, 1])
        draws = np.array(draws)
        out.append({"minimum_variants_per_protein": 4, "descriptor": f"[pooled] {name}",
                    "proteins": len(unique), "variants": int(good.sum()),
                    "macro_spearman": statistic,
                    "ci_low": float(np.percentile(draws, 2.5)),
                    "ci_high": float(np.percentile(draws, 97.5)),
                    "macro_partial_given_count": np.nan})
    return pd.DataFrame(out), pd.DataFrame(paired)


BACKBONE = ("helix_like", "extended_like", "coil")


def r3_backbone_beyond_constraint(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = frame[frame["backbone_state"].isin(BACKBONE)].copy()
    work = work.dropna(subset=["observed_position_intolerance"])
    positions = work.drop_duplicates(["protein_id", "position"])[
        ["protein_id", "position", "backbone_state", "observed_position_intolerance"]]
    positions["decile"] = pd.qcut(
        positions["observed_position_intolerance"].rank(method="first"), 10,
        labels=False)

    work = work.merge(positions[["protein_id", "position", "decile"]],
                      on=["protein_id", "position"], how="left")
    work["tertile"] = pd.cut(work["decile"], [-1, 2, 6, 9],
                             labels=["C1_least", "C2_mid", "C3_most"]).astype(object)
    unmatched = []
    for tertile in ("C1_least", "C2_mid", "C3_most"):
        for state in BACKBONE:
            part = work[(work["tertile"] == tertile) & (work["backbone_state"] == state)]
            forward, reverse, se2 = paired_arrays(directed_table(leave_one_out(part),
                                                                 minimum=10))
            unmatched.append({
                "tertile": tertile, "backbone_state": state, "pairs": len(forward),
                "positions": part.drop_duplicates(["protein_id", "position"]).shape[0],
                "mean_intolerance": float(part["observed_position_intolerance"].mean()),
                "symmetric_fraction": symmetric_fraction(forward, reverse, se2)
                if len(forward) >= 20 else np.nan})

    rng = np.random.default_rng(SEED)
    keyed = work.set_index(["protein_id", "position"]).sort_index()
    quota = (positions.groupby(["decile", "backbone_state"]).size().unstack(fill_value=0)
             .reindex(columns=list(BACKBONE), fill_value=0).min(axis=1))
    print("    positions available per backbone state after matching:", int(quota.sum()))
    draws: dict[str, list[float]] = {state: [] for state in BACKBONE}
    intolerance: dict[str, list[float]] = {state: [] for state in BACKBONE}
    pair_counts: dict[str, list[int]] = {state: [] for state in BACKBONE}
    for replicate in range(200):
        for state in BACKBONE:
            chosen = []
            pool = positions[positions["backbone_state"] == state]
            for decile, take in quota.items():
                if take <= 0:
                    continue
                candidates = pool[pool["decile"] == decile]
                if len(candidates) == 0:
                    continue
                picked = candidates.iloc[rng.integers(0, len(candidates), int(take))]
                chosen.append(picked)
            selected = pd.concat(chosen)
            block = keyed.loc[list(zip(selected["protein_id"], selected["position"]))]
            forward, reverse, se2 = paired_arrays(directed_table(leave_one_out(block),
                                                                 minimum=10))
            if len(forward) >= 20:
                draws[state].append(symmetric_fraction(forward, reverse, se2))
                pair_counts[state].append(len(forward))
            intolerance[state].append(float(selected["observed_position_intolerance"].mean()))
    matched = []
    for state in BACKBONE:
        values = np.array(draws[state])
        matched.append({
            "backbone_state": state, "replicates": len(values),
            "matched_positions": int(quota.sum()),
            "mean_intolerance": float(np.mean(intolerance[state])),
            "median_pairs": float(np.median(pair_counts[state])) if pair_counts[state] else np.nan,
            "symmetric_fraction": float(values.mean()) if len(values) else np.nan,
            "ci_low": float(np.percentile(values, 2.5)) if len(values) else np.nan,
            "ci_high": float(np.percentile(values, 97.5)) if len(values) else np.nan})
    matched_table = pd.DataFrame(matched)

    contrasts = []
    for first, second in (("helix_like", "extended_like"), ("helix_like", "coil"),
                          ("extended_like", "coil")):
        n = min(len(draws[first]), len(draws[second]))
        if n < 20:
            continue
        difference = np.array(draws[first][:n]) - np.array(draws[second][:n])
        contrasts.append({"comparison": f"{first} minus {second}",
                          "difference": float(difference.mean()),
                          "ci_low": float(np.percentile(difference, 2.5)),
                          "ci_high": float(np.percentile(difference, 97.5))})
    matched_table = pd.concat(
        [matched_table, pd.DataFrame(contrasts)], ignore_index=True)
    return pd.DataFrame(unmatched), matched_table


def exchange_matrix(frame: pd.DataFrame, burial: str | None, minimum: int = 15):
    extra = ["burial_class"] if burial else None
    scale = symmetric_scale(leave_one_out(frame), extra, minimum)
    values, mask = np.full((20, 20), np.nan), np.zeros((20, 20), bool)
    index = {residue: i for i, residue in enumerate(ORDER)}
    for key, value in scale.items():
        if burial and key[1] != burial:
            continue
        a, b = key[0]
        i, j = index[a], index[b]
        values[i, j] = values[j, i] = value
        mask[i, j] = mask[j, i] = True
    return values, mask


def fit_additive(values: np.ndarray, mask: np.ndarray, rounds: int = 200):
    mu = float(np.nanmean(values[mask]))
    alpha = np.zeros(values.shape[0])
    for _ in range(rounds):
        for i in range(values.shape[0]):
            row = mask[i]
            if not row.any():
                continue
            alpha[i] = np.mean(values[i, row] - mu - alpha[row])
        alpha -= alpha.mean()
        residual = values - (mu + alpha[:, None] + alpha[None, :])
        mu += float(np.mean(residual[mask]))
    return mu, alpha


CENTRE = np.eye(20) - np.ones((20, 20)) / 20


def warm_start(values: np.ndarray, mask: np.ndarray, mu: float, alpha: np.ndarray,
               rank: int, rounds: int = 40):
    residual = values - (mu + alpha[:, None] + alpha[None, :])
    filled = np.where(mask, residual, 0.0)
    vectors = np.zeros((20, rank))
    eigenvalues = np.zeros(rank)
    for _ in range(rounds):
        w, v = np.linalg.eigh(filled)
        order = np.argsort(np.abs(w))[::-1][:rank]
        eigenvalues, vectors = w[order], v[:, order]
        approximation = (vectors * eigenvalues) @ vectors.T
        filled = np.where(mask, residual, approximation)
    return vectors, eigenvalues


def fit_bilinear(values: np.ndarray, mask: np.ndarray, rank: int, ridge: float = 1e-3,
                 restarts: int = 2, seed: int = SEED):
    n = values.shape[0]
    upper = [(i, j) for i in range(n) for j in range(i + 1, n) if mask[i, j]]
    rows = np.array([i for i, _ in upper])
    columns = np.array([j for _, j in upper])
    target = np.array([values[i, j] for i, j in upper])

    def unpack(parameters):
        mu = parameters[0]
        alpha = CENTRE @ parameters[1:1 + n]
        rest = parameters[1 + n:]
        if rank == 0:
            return mu, alpha, np.zeros((n, 0)), np.zeros(0)
        factors = CENTRE @ rest[:n * rank].reshape(n, rank)
        norms = np.linalg.norm(factors, axis=0)
        norms[norms == 0] = 1.0
        return mu, alpha, factors / norms, rest[n * rank:]

    def residual(parameters):
        mu, alpha, factors, weights = unpack(parameters)
        prediction = mu + alpha[rows] + alpha[columns]
        if rank:
            interaction = (factors * weights) @ factors.T
            prediction = prediction + interaction[rows, columns]
        return np.concatenate([target - prediction,
                               np.sqrt(ridge) * parameters[1:]])

    mu0, alpha0 = fit_additive(values, mask)
    starts = []
    if rank:
        vectors, eigenvalues = warm_start(values, mask, mu0, alpha0, rank)
        starts.append(np.concatenate([[mu0], alpha0, vectors.ravel(), eigenvalues]))
    else:
        starts.append(np.concatenate([[mu0], alpha0]))
    for attempt in range(restarts - 1):
        rng = np.random.default_rng(seed + attempt)
        starts.append(np.concatenate([[mu0], alpha0,
                                      rng.normal(scale=0.3, size=n * rank),
                                      np.full(rank, 0.01)]))
    best = None
    for start in starts:
        outcome = least_squares(residual, start, method="trf", max_nfev=20000,
                                xtol=1e-10, ftol=1e-10)
        if best is None or outcome.cost < best.cost:
            best = outcome
    mu, alpha, factors, weights = unpack(best.x)
    if rank:
        interaction = (factors * weights) @ factors.T
        w, v = np.linalg.eigh(interaction)
        order = np.argsort(np.abs(w))[::-1][:rank]
        factors, weights = v[:, order], w[order]
    return mu, alpha, factors, weights


def predict_pairs(mu, alpha, factors, weights, pairs):
    if factors.shape[1] == 0:
        return np.array([mu + alpha[a] + alpha[b] for a, b in pairs])
    interaction = (factors * weights) @ factors.T
    return np.array([mu + alpha[a] + alpha[b] + interaction[a, b] for a, b in pairs])


def r4_dimensionality(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    skill_rows, loading_rows = [], []
    settings = [("all positions", None, 25), ("buried", "buried", 15),
                ("exposed", "exposed", 15)]
    for label, burial, minimum in settings:
        values, mask = exchange_matrix(frame, burial, minimum)
        pairs = [(i, j) for i in range(20) for j in range(i + 1, 20) if mask[i, j]]
        print(f"    {label}: {len(pairs)} of 190 pairs measured")
        if len(pairs) < 120:
            continue
        folds = rng.permutation(len(pairs)) % 5
        for rank in range(0, 6):
            predicted, actual = [], []
            for fold in range(5):
                train_mask = mask.copy()
                held = [pairs[k] for k in range(len(pairs)) if folds[k] == fold]
                for i, j in held:
                    train_mask[i, j] = train_mask[j, i] = False
                mu, alpha, factors, weights = fit_bilinear(values, train_mask, rank,
                                                           seed=SEED + fold)
                predicted.append(predict_pairs(mu, alpha, factors, weights, held))
                actual.append(np.array([values[i, j] for i, j in held]))
            predicted = np.concatenate(predicted)
            actual = np.concatenate(actual)
            sse = float(((actual - predicted) ** 2).sum())
            sst = float(((actual - actual.mean()) ** 2).sum())
            skill_rows.append({"stratum": label, "measured_pairs": len(pairs),
                               "model": "additive only" if rank == 0
                               else f"additive + rank {rank}",
                               "interaction_rank": rank,
                               "heldout_r2": 1 - sse / sst})
        mu, alpha, factors, weights = fit_bilinear(values, mask, 3)
        interaction = np.abs(weights) / np.abs(weights).sum()
        components = [("additive (replaceability)", pd.Series(alpha, index=ORDER), np.nan)]
        for k in range(factors.shape[1]):
            components.append((f"interaction PC{k + 1}",
                               pd.Series(factors[:, k], index=ORDER),
                               float(interaction[k])))
        reference = pd.Series([HYDROPATHY[r] for r in ORDER], index=ORDER)
        for name, series, share in components:
            if name.startswith("interaction") and series.corr(reference) < 0:
                series = -series
            loading_rows.append({
                "stratum": label, "component": name, "weight_share": share,
                "spearman_hydropathy": float(spearmanr(
                    series.values, [HYDROPATHY[r] for r in series.index]).statistic),
                "spearman_volume": float(spearmanr(
                    series.values, [VOLUME[r] for r in series.index]).statistic),
                "spearman_abs_charge": float(spearmanr(
                    series.values, [abs(CHARGE[r]) for r in series.index]).statistic),
                "spearman_aromatic": float(spearmanr(
                    series.values, [AROMATIC[r] for r in series.index]).statistic),
                "low_end": " ".join(series.sort_values().index[:4]),
                "high_end": " ".join(series.sort_values().index[-4:])})
    return pd.DataFrame(skill_rows), pd.DataFrame(loading_rows)


PREDICTORS = {"SPURS": "structure_sequence_expert_rank",
              "SaProt-DeltaG": "structure_language_expert_rank",
              "ThermoMPNN": "inverse_folding_expert_rank",
              "Pythia": "stability_language_expert_rank",
              "ESM-2": "sequence_language_expert_rank"}
WEIGHT_GRID = np.round(np.arange(0.0, 1.01, 0.05), 2)


def freeze_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    marginal = symmetric_scale(leave_one_out(frame), minimum=25)
    conditioned = symmetric_scale(leave_one_out(frame), ["burial_class"], minimum=15)
    rows = [{"burial_class": "any", "residue_a": pair[0][0], "residue_b": pair[0][1],
             "exchange": value} for pair, value in marginal.items()]
    rows += [{"burial_class": key[1], "residue_a": key[0][0], "residue_b": key[0][1],
              "exchange": value} for key, value in conditioned.items()]
    return pd.DataFrame(rows).sort_values(["burial_class", "residue_a", "residue_b"])


def lookup(frozen: pd.DataFrame):
    conditioned = {(row.burial_class, row.residue_a, row.residue_b): row.exchange
                   for row in frozen.itertuples() if row.burial_class != "any"}
    marginal = {(row.residue_a, row.residue_b): row.exchange
                for row in frozen.itertuples() if row.burial_class == "any"}
    default = float(np.mean(list(marginal.values())))

    def score(reference, alternate, burial):
        pair = tuple(sorted((str(reference), str(alternate))))
        if isinstance(burial, str) and (burial, *pair) in conditioned:
            return conditioned[(burial, *pair)]
        return marginal.get(pair, default)

    return score


def apply_scale(block: pd.DataFrame, score) -> pd.DataFrame:
    block = block.copy()
    burial = block["burial_class"] if "burial_class" in block else pd.Series(
        [None] * len(block), index=block.index)
    block["exchange"] = [score(a, b, c) for a, b, c in
                         zip(block["reference_residue"], block["alternate_residue"], burial)]
    block["blosum_feature"] = [BLOSUM.get((str(a), str(b)), 0.0) for a, b in
                               zip(block["reference_residue"], block["alternate_residue"])]
    return block


def skill_curve(block: pd.DataFrame, column: str, feature: str = "exchange",
                minimum: int = 15):
    grouping = "homology_cluster" if "homology_cluster" in block else "protein_id"
    plain, repaired, groups = [], [], []
    for (_, group_id), part in block.groupby(["assay_id", grouping], sort=False):
        clean = part.dropna(subset=[column, "effect_rank", feature])
        if len(clean) < minimum or clean["effect_rank"].nunique() < 5 \
                or clean[feature].nunique() < 3:
            continue
        outcome = clean["effect_rank"].to_numpy()
        base = spearmanr(outcome, clean[column]).statistic
        if not np.isfinite(base):
            continue
        predictor_rank = clean[column].rank(pct=True).to_numpy()
        feature_rank = clean[feature].rank(pct=True).to_numpy()
        row = [spearmanr(outcome, predictor_rank + w * feature_rank).statistic
               for w in WEIGHT_GRID]
        if not np.all(np.isfinite(row)):
            continue
        plain.append(base)
        repaired.append(row)
        groups.append(group_id)
    return np.array(plain), np.array(repaired), np.array(groups)


def choose_weight(frame: pd.DataFrame, folds: int = 5) -> tuple[float, pd.DataFrame]:
    proteins = np.array(sorted(frame["protein_id"].unique()))
    assignment = pd.Series(
        np.random.default_rng(SEED).permutation(len(proteins)) % folds, index=proteins)
    collected = []
    for fold in range(folds):
        held = assignment[assignment == fold].index
        train = frame[~frame["protein_id"].isin(held)]
        test = frame[frame["protein_id"].isin(held)]
        scored = apply_scale(test, lookup(freeze_matrix(train)))
        for name, column in PREDICTORS.items():
            plain, repaired, _ = skill_curve(scored, column)
            if not len(plain):
                continue
            for index, weight in enumerate(WEIGHT_GRID):
                collected.append({"fold": fold, "predictor": name,
                                  "weight": float(weight), "assays": len(plain),
                                  "heldout_gain": float((repaired[:, index]
                                                         - plain).mean())})
    curve = pd.DataFrame(collected)
    averaged = curve.groupby("weight")["heldout_gain"].mean()
    return float(averaged.idxmax()), curve


def r5_frozen_tool(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    weight, curve = choose_weight(frame)
    index = int(np.argmin(np.abs(WEIGHT_GRID - weight)))
    print(f"    mixing weight chosen on held-out abundance proteins: w = {weight}")

    frozen = freeze_matrix(frame)
    frozen.to_csv(RESULTS / "R5_frozen_exchange_matrix.csv", index=False)
    print(f"    frozen matrix: {len(frozen)} rows -> R5_frozen_exchange_matrix.csv")
    score = lookup(frozen)

    external_blocks = {
        "VenusMutHub stability (no structure)": pd.read_parquet(DATA / "stability.parquet"),
        "biochemical, non-stability endpoints":
            pd.read_parquet(DATA / "biochemical.parquet").query(
                "endpoint_class != 'stability'"),
    }
    rows = []
    for label, block in external_blocks.items():
        scored = apply_scale(block, score)
        standalone: dict[str, list[float]] = {"exchange": [], "blosum_feature": []}
        standalone_groups: list = []
        for name, column in PREDICTORS.items():
            plain, repaired, groups = skill_curve(scored, column)
            if len(plain) < 15:
                continue
            gain = repaired[:, index] - plain
            low, high = cluster_bootstrap(gain, groups, seed=SEED + len(name))
            rows.append({"external_block": label, "predictor": name,
                         "assays": len(plain), "clusters": len(np.unique(groups)),
                         "macro_plain": float(plain.mean()),
                         "macro_repaired": float(repaired[:, index].mean()),
                         "gain": float(gain.mean()), "ci_low": low, "ci_high": high})
        grouping = "homology_cluster" if "homology_cluster" in scored else "protein_id"
        for (_, group_id), part in scored.groupby(["assay_id", grouping], sort=False):
            clean = part.dropna(subset=["effect_rank", "exchange", "blosum_feature"])
            if len(clean) < 15 or clean["effect_rank"].nunique() < 5:
                continue
            statistics = {f: spearmanr(clean["effect_rank"], clean[f]).statistic
                          for f in standalone}
            if not all(np.isfinite(v) for v in statistics.values()):
                continue
            for f, v in statistics.items():
                standalone[f].append(v)
            standalone_groups.append(group_id)
        groups = np.array(standalone_groups)
        if len(groups) >= 15:
            for feature, name in (("exchange", "measured symmetric exchange, frozen"),
                                  ("blosum_feature", "BLOSUM62")):
                values = np.array(standalone[feature])
                low, high = cluster_bootstrap(values, groups, seed=SEED + len(name))
                rows.append({"external_block": label,
                             "predictor": f"[standalone] {name}",
                             "assays": len(values), "clusters": len(np.unique(groups)),
                             "macro_plain": np.nan,
                             "macro_repaired": float(values.mean()),
                             "gain": np.nan, "ci_low": low, "ci_high": high})
            difference = np.array(standalone["exchange"]) - np.array(
                standalone["blosum_feature"])
            low, high = cluster_bootstrap(difference, groups, seed=SEED + 7)
            rows.append({"external_block": label,
                         "predictor": "[standalone] exchange minus BLOSUM62",
                         "assays": len(difference), "clusters": len(np.unique(groups)),
                         "macro_plain": np.nan, "macro_repaired": np.nan,
                         "gain": float(difference.mean()), "ci_low": low, "ci_high": high})
    return frozen, curve, pd.DataFrame(rows)


def r5_external_weight_sweep(frame: pd.DataFrame) -> pd.DataFrame:
    score = lookup(freeze_matrix(frame))
    blocks = {"VenusMutHub stability (no structure)":
              pd.read_parquet(DATA / "stability.parquet"),
              "biochemical, non-stability endpoints":
              pd.read_parquet(DATA / "biochemical.parquet").query(
                  "endpoint_class != 'stability'")}
    rows = []
    for label, block in blocks.items():
        scored = apply_scale(block, score)
        for name, column in PREDICTORS.items():
            plain, repaired, groups = skill_curve(scored, column)
            if len(plain) < 15:
                continue
            gains = repaired.mean(axis=0) - plain.mean()
            best = int(np.argmax(gains))
            low, high = cluster_bootstrap(repaired[:, best] - plain, groups,
                                          seed=SEED + len(name))
            rows.append({"external_block": label, "predictor": name,
                         "assays": len(plain),
                         "best_weight_on_test_set": float(WEIGHT_GRID[best]),
                         "ceiling_gain": float(gains[best]),
                         "ceiling_ci_low": low, "ceiling_ci_high": high,
                         "gain_at_w_0.25": float(gains[int(np.argmin(
                             np.abs(WEIGHT_GRID - 0.25)))]),
                         "gain_at_w_1.00": float(gains[-1])})
    return pd.DataFrame(rows)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    parts = sys.argv[1:] or ["R2", "R3", "R4", "R5"]
    frame = None
    if {"R3", "R4", "R5", "R5b"} & set(parts):
        frame = pd.read_parquet(DATA / "abundance.parquet")

    if "R2" in parts:
        print("R2  does the measured exchange scale transfer to multi-substitution variants?")
        table, paired = r2_multimutant()
        table.to_csv(RESULTS / "R2_multimutant.csv", index=False)
        paired.to_csv(RESULTS / "R2_exchange_vs_blosum.csv", index=False)
        print(table.round(4).to_string(index=False))
        print()
        print(paired.round(4).to_string(index=False))
        print()

    if "R3" in parts:
        print("R3  does backbone state carry anything beyond constraint?")
        unmatched, matched = r3_backbone_beyond_constraint(frame)
        unmatched.to_csv(RESULTS / "R3_backbone_unmatched.csv", index=False)
        matched.to_csv(RESULTS / "R3_backbone_matched.csv", index=False)
        print("  unmatched, by constraint tertile:")
        print(unmatched.round(3).to_string(index=False))
        print("  matched on constraint decile:")
        print(matched.round(4).to_string(index=False))
        print()

    if "R4" in parts:
        print("R4  how many dimensions does the exchange matrix have?")
        skill, loadings = r4_dimensionality(frame)
        skill.to_csv(RESULTS / "R4_dimensionality.csv", index=False)
        loadings.to_csv(RESULTS / "R4_component_loadings.csv", index=False)
        print(skill.round(4).to_string(index=False))
        print()
        print(loadings.round(3).to_string(index=False))
        print()

    if "R5b" in parts:
        print("R5b  external weight sweep: mis-transported weight, or inert matrix?")
        sweep = r5_external_weight_sweep(frame)
        sweep.to_csv(RESULTS / "R5b_external_weight_sweep.csv", index=False)
        print(sweep.round(4).to_string(index=False))
        print()

    if "R5" in parts:
        print("R5  frozen matrix applied unchanged to external data")
        _, tuning, external = r5_frozen_tool(frame)
        tuning.to_csv(RESULTS / "R5_weight_selection.csv", index=False)
        external.to_csv(RESULTS / "R5_external_validation.csv", index=False)
        print(external.round(4).to_string(index=False))


if __name__ == "__main__":
    main()

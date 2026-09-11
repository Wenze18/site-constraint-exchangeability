from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew, spearmanr
from sklearn.mixture import GaussianMixture

PROJECT = Path(__file__).resolve().parent.parent
DATA = PROJECT / "data"
RESULTS = PROJECT / "results"
SEED = 20260908
BINS = [f"C{i + 1}" for i in range(5)]
GROUPS = {"aliphatic": set("ILVM"), "aromatic": set("FYW"), "small_polar": set("STA"),
          "acidic": set("DE"), "basic": set("KR"), "amide": set("NQ")}
GROUP_OF = {residue: name for name, members in GROUPS.items() for residue in members}


def with_bins(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.dropna(subset=["observed_position_intolerance", "effect_rank"]).copy()
    work["constraint_bin"] = (
        work.groupby("protein_id")["observed_position_intolerance"]
        .transform(lambda s: pd.qcut(s.rank(method="first"), 5,
                                     labels=BINS).astype(object)))
    return work


def leave_one_out(frame: pd.DataFrame) -> pd.DataFrame:
    total = frame.groupby(["protein_id", "position"])["effect_rank"].transform("sum")
    count = frame.groupby(["protein_id", "position"])["effect_rank"].transform("size")
    keep = count >= 4
    work, total, count = frame[keep].copy(), total[keep], count[keep]
    work["contrast"] = work["effect_rank"] - (total - work["effect_rank"]) / (count - 1)
    return work


def pair_values(block: pd.DataFrame, minimum: int = 15) -> dict:
    table = (leave_one_out(block)
             .groupby(["reference_residue", "alternate_residue"])["contrast"]
             .agg(mean="mean", n="size").dropna())
    table = table[table["n"] >= minimum]
    out, seen = {}, set()
    for a, b in table.index:
        if (b, a) not in table.index or (b, a) in seen:
            continue
        seen.add((a, b))
        weights = np.array([table.at[(a, b), "n"], table.at[(b, a), "n"]], float)
        means = np.array([table.at[(a, b), "mean"], table.at[(b, a), "mean"]])
        out[(a, b)] = float(np.average(means, weights=weights))
    return out


def shape_by_quintile(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label in BINS:
        values = pair_values(frame[frame["constraint_bin"] == label])
        for name, selector in (("between class", lambda a, b: GROUP_OF.get(a) != GROUP_OF.get(b)),
                               ("within class", lambda a, b: GROUP_OF.get(a) == GROUP_OF.get(b))):
            sample = np.array([v for (a, b), v in values.items()
                               if a in GROUP_OF and b in GROUP_OF and selector(a, b)])
            if len(sample) < 12:
                continue
            standardised = ((sample - sample.mean()) / sample.std(ddof=1)).reshape(-1, 1)
            one = GaussianMixture(1, random_state=SEED).fit(standardised)
            two = GaussianMixture(2, random_state=SEED, n_init=5).fit(standardised)
            s, k = float(skew(sample)), float(kurtosis(sample, fisher=False))
            rows.append({
                "constraint_bin": label, "pair_class": name, "pairs": len(sample),
                "spread": float(sample.std(ddof=1)),
                "skew": s, "kurtosis": k,
                "bimodality_coefficient": (s ** 2 + 1) / k if k > 0 else np.nan,
                "bic_one_component": float(one.bic(standardised)),
                "bic_two_components": float(two.bic(standardised)),
                "bic_favours": "two" if two.bic(standardised) < one.bic(standardised)
                               else "one",
                "fraction_within_half_sd_of_min": float(
                    (sample <= sample.min() + 0.5 * sample.std(ddof=1)).mean())})
    return pd.DataFrame(rows)


def tolerance_set_size(frame: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    for label in BINS:
        block = frame[frame["constraint_bin"] == label]
        sizes, counts, proteins = [], [], []
        for (protein, _), part in block.groupby(["protein_id", "position"], sort=False):
            outcome = part["effect_rank"].to_numpy()
            if len(outcome) < 8:
                continue
            weights = outcome - outcome.min()
            if weights.sum() <= 0:
                continue
            weights = weights / weights.sum()
            positive = weights[weights > 0]
            sizes.append(float(np.exp(-(positive * np.log(positive)).sum())))
            counts.append(len(outcome))
            proteins.append(protein)
        if len(sizes) < 50:
            continue
        sizes, counts, proteins = np.array(sizes), np.array(counts), np.array(proteins)
        ratio = sizes / counts
        blocks = [ratio[proteins == p] for p in np.unique(proteins)]
        index = rng.integers(0, len(blocks), (2000, len(blocks)))
        draws = np.array([np.concatenate([blocks[j] for j in row]).mean()
                          for row in index])
        rows.append({"constraint_bin": label, "positions": len(sizes),
                     "mean_measured_substitutions": float(counts.mean()),
                     "effective_tolerated_substitutions": float(sizes.mean()),
                     "as_fraction_of_measured": float(ratio.mean()),
                     "ci_low": float(np.percentile(draws, 2.5)),
                     "ci_high": float(np.percentile(draws, 97.5))})
    return pd.DataFrame(rows)


def radius_test(frame: pd.DataFrame, axis: dict) -> pd.DataFrame:
    rows = []
    for label in BINS:
        block = leave_one_out(frame[frame["constraint_bin"] == label]).copy()
        block["distance"] = [abs(axis[b] - axis[a]) for a, b in
                             zip(block["reference_residue"], block["alternate_residue"])]
        block = block.dropna(subset=["distance", "contrast"])
        proteins = np.array(sorted(block["protein_id"].unique()))
        assignment = pd.Series(
            np.random.default_rng(SEED).permutation(len(proteins)) % 5, index=proteins)
        block["fold"] = block["protein_id"].map(assignment)
        grid = np.quantile(block["distance"], np.linspace(0.1, 0.9, 33))
        skill = {"graded (linear in distance)": [], "step at fitted radius": [],
                 "both": []}
        radii = []
        for fold in range(5):
            train = block[block["fold"] != fold]
            test = block[block["fold"] == fold]
            if len(train) < 500 or len(test) < 200:
                continue
            best, best_sse = None, np.inf
            for radius in grid:
                indicator = (train["distance"] <= radius).to_numpy(float)
                if indicator.std() == 0:
                    continue
                design = np.column_stack([np.ones(len(train)), indicator])
                coefficients, *_ = np.linalg.lstsq(design, train["contrast"], rcond=None)
                sse = float(((train["contrast"] - design @ coefficients) ** 2).sum())
                if sse < best_sse:
                    best_sse, best = sse, radius
            radii.append(best)
            designs = {
                "graded (linear in distance)": lambda d: np.column_stack(
                    [np.ones(len(d)), d["distance"]]),
                "step at fitted radius": lambda d: np.column_stack(
                    [np.ones(len(d)), (d["distance"] <= best).astype(float)]),
                "both": lambda d: np.column_stack(
                    [np.ones(len(d)), d["distance"],
                     (d["distance"] <= best).astype(float)]),
            }
            for name, build in designs.items():
                coefficients, *_ = np.linalg.lstsq(build(train), train["contrast"],
                                                   rcond=None)
                prediction = build(test) @ coefficients
                sse = float(((test["contrast"] - prediction) ** 2).sum())
                sst = float(((test["contrast"] - train["contrast"].mean()) ** 2).sum())
                skill[name].append(1 - sse / sst)
        if not radii:
            continue
        row = {"constraint_bin": label, "measurements": len(block),
               "median_fitted_radius": float(np.median(radii))}
        for name, values in skill.items():
            row[name] = float(np.mean(values))
        row["step_minus_graded"] = row["step at fitted radius"] - row["graded (linear in distance)"]
        rows.append(row)
    return pd.DataFrame(rows)


def neighbour_ordering(frame: pd.DataFrame, axis: dict) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    for label in BINS:
        block = frame[frame["constraint_bin"] == label]
        values, proteins = [], []
        for (protein, _), part in block.groupby(["protein_id", "position"], sort=False):
            if len(part) < 8:
                continue
            distance = np.array([abs(axis[b] - axis[a]) for a, b in
                                 zip(part["reference_residue"], part["alternate_residue"])])
            if len(np.unique(distance)) < 3:
                continue
            statistic = spearmanr(part["effect_rank"], -distance).statistic
            if np.isfinite(statistic):
                values.append(statistic)
                proteins.append(protein)
        if len(values) < 50:
            continue
        values, proteins = np.array(values), np.array(proteins)
        blocks = [values[proteins == p] for p in np.unique(proteins)]
        index = rng.integers(0, len(blocks), (2000, len(blocks)))
        draws = np.array([np.concatenate([blocks[j] for j in row]).mean()
                          for row in index])
        rows.append({"constraint_bin": label, "positions": len(values),
                     "macro_spearman_outcome_vs_closeness": float(values.mean()),
                     "ci_low": float(np.percentile(draws, 2.5)),
                     "ci_high": float(np.percentile(draws, 97.5))})
    return pd.DataFrame(rows)


def main() -> None:
    frame = with_bins(pd.read_parquet(DATA / "abundance.parquet"))
    coordinates = pd.read_csv(RESULTS / "R4_residue_coordinates.csv")
    axis = dict(zip(coordinates.query("stratum == 'all positions'")["residue"],
                    coordinates.query("stratum == 'all positions'")["axis_coordinate"]))

    print("1  distribution shape of the exchange values")
    shape = shape_by_quintile(frame)
    shape.to_csv(RESULTS / "N1_distribution_shape.csv", index=False)
    print(shape.round(3).to_string(index=False))

    print("\n2  effective number of tolerated substitutions per position")
    sizes = tolerance_set_size(frame)
    sizes.to_csv(RESULTS / "N1_tolerance_set_size.csv", index=False)
    print(sizes.round(3).to_string(index=False))

    print("\n3  can one radius on the R4 axis describe the tolerated set?")
    radius = radius_test(frame, axis)
    radius.to_csv(RESULTS / "N1_radius_test.csv", index=False)
    print(radius.round(4).to_string(index=False))

    print("\n4  is the outcome ordered by closeness on the axis?")
    neighbour = neighbour_ordering(frame, axis)
    neighbour.to_csv(RESULTS / "N1_neighbour_ordering.csv", index=False)
    print(neighbour.round(4).to_string(index=False))


if __name__ == "__main__":
    main()

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DATA = PROJECT / "data"
RESULTS = PROJECT / "results"
SEED = 20260908

BINS = 5
LABELS = [f"C{i + 1}" for i in range(BINS)]
MIN_CELL = 15
MIN_SITE = 8
SPLITS = 40
RATIOS = [(1, 3), (1, 2), (2, 3)]
DRAWS = 2500


def prepare() -> dict:
    frame = pd.read_parquet(DATA / "abundance.parquet")
    frame = frame.dropna(subset=["effect_rank", "reference_residue",
                                 "alternate_residue"]).reset_index(drop=True)
    site_key = frame["protein_id"].astype(str) + "\x00" + frame["position"].astype(str)
    site, site_index = pd.factorize(site_key)
    counts = np.bincount(site)
    keep = counts[site] >= MIN_SITE
    frame = frame[keep].reset_index(drop=True)
    site = pd.factorize(frame["protein_id"].astype(str) + "\x00"
                        + frame["position"].astype(str))[0]
    counts = np.bincount(site)
    protein_of_site = (pd.DataFrame({"site": site, "protein": frame["protein_id"].values})
                       .drop_duplicates("site").sort_values("site")["protein"].values)
    return {"frame": frame, "site": site, "counts": counts,
            "protein_of_site": protein_of_site,
            "y": frame["effect_rank"].to_numpy(float),
            "ref": frame["reference_residue"].to_numpy(),
            "alt": frame["alternate_residue"].to_numpy()}


def within_site_rank(site: np.ndarray, noise: np.ndarray) -> np.ndarray:
    order = np.lexsort((noise, site))
    sorted_site = site[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_site)) + 1]
    lengths = np.diff(np.r_[starts, len(site)])
    ranks = np.empty(len(site), dtype=np.int64)
    ranks[order] = np.arange(len(site)) - np.repeat(starts, lengths)
    return ranks


def quintiles_within_protein(constraint: np.ndarray, protein: np.ndarray) -> np.ndarray:
    table = pd.DataFrame({"constraint": constraint, "protein": protein})
    good = table["constraint"].notna()
    binned = pd.Series(np.full(len(table), -1), index=table.index)
    ranked = (table.loc[good].groupby("protein")["constraint"]
              .transform(lambda s: pd.qcut(s.rank(method="first"), BINS,
                                           labels=False, duplicates="drop")))
    binned.loc[good] = ranked.fillna(-1).astype(int)
    return binned.to_numpy()


def pair_arrays(bin_of_row, ref, alt, contrast, wanted) -> tuple:
    mask = bin_of_row == wanted
    table = (pd.DataFrame({"a": ref[mask], "b": alt[mask], "c": contrast[mask]})
             .groupby(["a", "b"])["c"].agg(mean="mean", var="var", n="size").dropna())
    table = table[table["n"] >= MIN_CELL]
    table["se2"] = table["var"] / table["n"]
    index = set(table.index)
    seen, keep = set(), []
    for a, b in table.index:
        if (b, a) in index and (b, a) not in seen:
            seen.add((a, b))
            keep.append((a, b))
    if len(keep) < 20:
        return np.array([]), np.array([]), np.array([])
    forward = np.array([table.at[(a, b), "mean"] for a, b in keep])
    reverse = np.array([table.at[(b, a), "mean"] for a, b in keep])
    se2 = np.array([table.at[(a, b), "se2"] + table.at[(b, a), "se2"] for a, b in keep])
    return forward, reverse, se2


def fraction(forward, reverse, se2, index=None) -> float:
    if index is not None:
        forward, reverse, se2 = forward[index], reverse[index], se2[index]
    floor = se2.mean() / 4
    sym = max(((forward + reverse) / 2).var(ddof=1) - floor, 1e-12)
    anti = max(((forward - reverse) / 2).var(ddof=1) - floor, 1e-12)
    return sym / (sym + anti)


def one_split(bag: dict, rng: np.random.Generator, numerator: int, denominator: int,
              swap: bool) -> dict:
    site, y, counts = bag["site"], bag["y"], bag["counts"]
    ranks = within_site_rank(site, rng.random(len(site)))
    take = np.maximum((counts * numerator) // denominator, 2)
    take = np.minimum(take, counts - 2)
    on_constraint_side = ranks < take[site]
    if swap:
        on_constraint_side = ~on_constraint_side

    n_sites = len(counts)
    c_sum = np.bincount(site[on_constraint_side], weights=y[on_constraint_side],
                        minlength=n_sites)
    c_n = np.bincount(site[on_constraint_side], minlength=n_sites)
    constraint = np.where(c_n > 0, -c_sum / np.maximum(c_n, 1), np.nan)

    estimation = ~on_constraint_side
    e_sum = np.bincount(site[estimation], weights=y[estimation], minlength=n_sites)
    e_n = np.bincount(site[estimation], minlength=n_sites)
    usable = estimation & (e_n[site] >= 4)
    contrast = np.full(len(y), np.nan)
    contrast[usable] = (y[usable]
                        - (e_sum[site[usable]] - y[usable]) / (e_n[site[usable]] - 1))

    site_bin = quintiles_within_protein(constraint, bag["protein_of_site"])
    row_bin = np.where(usable, site_bin[site], -1)

    out = {"strata": {}, "constraint_mean": {}, "reliability_n": int((c_n > 0).sum())}
    for index, label in enumerate(LABELS):
        out["strata"][label] = pair_arrays(row_bin, bag["ref"], bag["alt"],
                                           np.nan_to_num(contrast), index)
        chosen = site_bin == index
        out["constraint_mean"][label] = float(np.nanmean(constraint[chosen]))
    return out


def leave_one_out_reference(bag: dict) -> pd.DataFrame:
    site, y, counts = bag["site"], bag["y"], bag["counts"]
    n_sites = len(counts)
    total = np.bincount(site, weights=y, minlength=n_sites)
    constraint = -total / counts
    contrast = y - (total[site] - y) / (counts[site] - 1)
    site_bin = quintiles_within_protein(constraint, bag["protein_of_site"])
    row_bin = site_bin[site]
    rows = []
    rng = np.random.default_rng(SEED)
    for index, label in enumerate(LABELS):
        forward, reverse, se2 = pair_arrays(row_bin, bag["ref"], bag["alt"],
                                            contrast, index)
        draws = np.array([fraction(forward, reverse, se2,
                                   rng.integers(0, len(forward), len(forward)))
                          for _ in range(DRAWS)])
        rows.append({"estimator": "leave-one-out, same positions",
                     "constraint_bin": label, "pairs": len(forward),
                     "mean_constraint": float(np.nanmean(constraint[site_bin == index])),
                     "fraction": fraction(forward, reverse, se2),
                     "ci_low": float(np.percentile(draws, 2.5)),
                     "ci_high": float(np.percentile(draws, 97.5))})
    return pd.DataFrame(rows)


def reliability(bag: dict, rng: np.random.Generator, repeats: int = 20) -> pd.DataFrame:
    site, y, counts = bag["site"], bag["y"], bag["counts"]
    n_sites = len(counts)
    rows = []
    for numerator, denominator in RATIOS:
        pearson, spearman = [], []
        for _ in range(repeats):
            ranks = within_site_rank(site, rng.random(len(site)))
            take = np.minimum(np.maximum((counts * numerator) // denominator, 2),
                              counts - 2)
            first = ranks < take[site]
            a_sum = np.bincount(site[first], weights=y[first], minlength=n_sites)
            a_n = np.bincount(site[first], minlength=n_sites)
            b_sum = np.bincount(site[~first], weights=y[~first], minlength=n_sites)
            b_n = np.bincount(site[~first], minlength=n_sites)
            good = (a_n > 0) & (b_n > 0)
            first_mean = (a_sum[good] / a_n[good])
            second_mean = (b_sum[good] / b_n[good])
            pearson.append(float(np.corrcoef(first_mean, second_mean)[0, 1]))
            spearman.append(float(pd.Series(first_mean).corr(pd.Series(second_mean),
                                                             method="spearman")))
        rows.append({"constraint_side": f"{numerator}/{denominator}",
                     "sides_pearson": float(np.mean(pearson)),
                     "sides_spearman": float(np.mean(spearman)),
                     "repeats": repeats})
    return pd.DataFrame(rows)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    bag = prepare()
    print(f"positions with at least {MIN_SITE} substitutions: {len(bag['counts'])}, "
          f"carrying {len(bag['y'])} substitutions in "
          f"{bag['frame']['protein_id'].nunique()} proteins", flush=True)

    reference = leave_one_out_reference(bag)
    print("\n-- leave-one-out on these positions --")
    print(reference[["constraint_bin", "pairs", "mean_constraint", "fraction",
                     "ci_low", "ci_high"]].to_string(index=False))

    rows, sweep = [], []
    for numerator, denominator in RATIOS:
        rng = np.random.default_rng(SEED + 100 * numerator + denominator)
        held: dict[str, list] = {label: [] for label in LABELS}
        constraint_means: dict[str, list] = {label: [] for label in LABELS}
        for step in range(SPLITS):
            result = one_split(bag, rng, numerator, denominator, swap=bool(step % 2))
            for label in LABELS:
                forward, reverse, se2 = result["strata"][label]
                if len(forward):
                    held[label].append((forward, reverse, se2))
                    constraint_means[label].append(result["constraint_mean"][label])
        label_rows = []
        for label in LABELS:
            bank = held[label]
            point = float(np.mean([fraction(*item) for item in bank]))
            spread = float(np.std([fraction(*item) for item in bank], ddof=1))
            draws = []
            for index in range(DRAWS):
                forward, reverse, se2 = bank[index % len(bank)]
                draws.append(fraction(forward, reverse, se2,
                                      rng.integers(0, len(forward), len(forward))))
            label_rows.append({"estimator": "cross-fitted",
                               "constraint_side": f"{numerator}/{denominator}",
                               "constraint_bin": label,
                               "splits": len(bank),
                               "pairs": int(np.mean([len(i[0]) for i in bank])),
                               "mean_constraint": float(np.mean(constraint_means[label])),
                               "fraction": point,
                               "split_sd": spread,
                               "ci_low": float(np.percentile(draws, 2.5)),
                               "ci_high": float(np.percentile(draws, 97.5))})
        block = pd.DataFrame(label_rows)
        rows.append(block)
        gradient = block.iloc[-1]["fraction"] - block.iloc[0]["fraction"]
        sweep.append({"constraint_side": f"{numerator}/{denominator}",
                      "C1": block.iloc[0]["fraction"], "C5": block.iloc[-1]["fraction"],
                      "gradient": gradient,
                      "monotone": bool(np.all(np.diff(block["fraction"].to_numpy()) > 0)),
                      "constraint_span": (block.iloc[-1]["mean_constraint"]
                                          - block.iloc[0]["mean_constraint"])})
        print(f"\n-- cross-fitted, constraint side {numerator}/{denominator} --")
        print(block[["constraint_bin", "pairs", "mean_constraint", "fraction",
                     "split_sd", "ci_low", "ci_high"]].to_string(index=False))

    detail = pd.concat(rows, ignore_index=True)
    detail.to_csv(RESULTS / "X1_crossfit_gradient.csv", index=False)
    reference.to_csv(RESULTS / "X1_crossfit_reference.csv", index=False)
    pd.DataFrame(sweep).to_csv(RESULTS / "X1_crossfit_ratio_sweep.csv", index=False)
    rel = reliability(bag, np.random.default_rng(SEED + 5))
    rel.to_csv(RESULTS / "X1_constraint_reliability.csv", index=False)

    print("\n-- split ratio sweep --")
    print(pd.DataFrame(sweep).to_string(index=False))
    print("\n-- reliability of a partial constraint estimate --")
    print(rel.to_string(index=False))
    reference_gradient = (reference.iloc[-1]["fraction"] - reference.iloc[0]["fraction"])
    print(f"\nleave-one-out gradient on these positions: {reference_gradient:+.3f}")
    print("cross-fitted gradients: "
          + ", ".join(f"{r['constraint_side']} {r['gradient']:+.3f}" for r in sweep))


if __name__ == "__main__":
    main()

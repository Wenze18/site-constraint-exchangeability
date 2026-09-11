from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DATA = PROJECT / "data"
RESULTS = PROJECT / "results"
SEED = 20260908
DRAWS = 4000
MIN_SITE = 4
CELL = 4

GROUPS = {"aliphatic": set("ILVM"), "aromatic": set("FYW"), "small_polar": set("STA"),
          "acidic": set("DE"), "basic": set("KR"), "amide": set("NQ")}
GROUP_OF = {r: g for g, members in GROUPS.items() for r in members}


def contrasts(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.dropna(subset=["effect_rank"]).copy()
    grouped = work.groupby(["protein_id", "position"])["effect_rank"]
    total, count = grouped.transform("sum"), grouped.transform("size")
    keep = count >= MIN_SITE
    work, total, count = work[keep].copy(), total[keep], count[keep]
    work["contrast"] = work["effect_rank"] - (total - work["effect_rank"]) / (count - 1)
    work["constraint"] = -total / count
    return work


def pair_values(part: pd.DataFrame, cell: int) -> pd.DataFrame:
    table = (part.groupby(["reference_residue", "alternate_residue"])["contrast"]
             .agg(mean="mean", var="var", n="size").dropna())
    table = table[table["n"] >= cell]
    table["se2"] = table["var"] / table["n"]
    index = set(table.index)
    rows = []
    seen = set()
    for a, b in table.index:
        if (b, a) not in index or (b, a) in seen:
            continue
        seen.add((a, b))
        weights = np.array([table.at[(a, b), "n"], table.at[(b, a), "n"]], float)
        rows.append({"a": a, "b": b,
                     "forward": table.at[(a, b), "mean"],
                     "reverse": table.at[(b, a), "mean"],
                     "se2": table.at[(a, b), "se2"] + table.at[(b, a), "se2"],
                     "symmetric": float(np.average(
                         [table.at[(a, b), "mean"], table.at[(b, a), "mean"]],
                         weights=weights)),
                     "within": (a in GROUP_OF and b in GROUP_OF
                                and GROUP_OF[a] == GROUP_OF[b]),
                     "classified": a in GROUP_OF and b in GROUP_OF})
    return pd.DataFrame(rows)


def share(pairs: pd.DataFrame, index=None) -> float:
    part = pairs if index is None else pairs.iloc[index]
    floor = part["se2"].mean() / 4
    sym = max(((part["forward"] + part["reverse"]) / 2).var(ddof=1) - floor, 1e-12)
    anti = max(((part["forward"] - part["reverse"]) / 2).var(ddof=1) - floor, 1e-12)
    return sym / (sym + anti)


def block(pairs: pd.DataFrame, index=None) -> float:
    part = pairs if index is None else pairs.iloc[index]
    part = part[part["classified"]]
    within, between = part[part["within"]], part[~part["within"]]
    if len(within) < 3 or len(between) < 10:
        return np.nan
    return float(within["symmetric"].mean() - between["symmetric"].mean())


def interval(pairs: pd.DataFrame, statistic, rng) -> tuple[float, float, float]:
    point = statistic(pairs)
    draws = np.array([statistic(pairs, rng.integers(0, len(pairs), len(pairs)))
                      for _ in range(DRAWS)])
    draws = draws[np.isfinite(draws)]
    if not len(draws):
        return point, np.nan, np.nan
    return point, float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def evaluate(name: str, work: pd.DataFrame, scheme: str) -> list[dict]:
    per_site = (work.groupby(["protein_id", "position"])["constraint"].first()
                .reset_index())
    if scheme == "within protein":
        cut = per_site.groupby("protein_id")["constraint"].transform("median")
    else:
        cut = per_site["constraint"].median()
    per_site["half"] = np.where(per_site["constraint"] <= cut, "less constrained",
                                "more constrained")
    work = work.merge(per_site[["protein_id", "position", "half"]],
                      on=["protein_id", "position"], how="left")
    rows = []
    rng = np.random.default_rng(SEED)
    for half in ("less constrained", "more constrained"):
        part = work[work["half"] == half]
        pairs = pair_values(part, CELL)
        row = {"resource": name, "split": scheme, "half": half,
               "variants": int(len(part)),
               "positions": int(part.groupby(["protein_id", "position"]).ngroups),
               "pairs": int(len(pairs)),
               "mean_constraint": float(part["constraint"].mean())}
        if len(pairs) >= 20:
            point, low, high = interval(pairs, share, rng)
            row |= {"symmetric_share": point, "share_ci_low": low,
                    "share_ci_high": high}
        else:
            row |= {"symmetric_share": np.nan, "share_ci_low": np.nan,
                    "share_ci_high": np.nan}
        classified = pairs[pairs["classified"]] if len(pairs) else pairs
        if len(classified) >= 15:
            point, low, high = interval(pairs, block, rng)
            row |= {"block_contrast": point, "block_ci_low": low,
                    "block_ci_high": high,
                    "within_pairs": int(classified["within"].sum()),
                    "between_pairs": int((~classified["within"]).sum())}
        else:
            row |= {"block_contrast": np.nan, "block_ci_low": np.nan,
                    "block_ci_high": np.nan, "within_pairs": len(classified),
                    "between_pairs": 0}
        rows.append(row)
    return rows


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, path in (("abundance (reference)", DATA / "abundance.parquet"),
                       ("external stability", DATA / "stability.parquet"),
                       ("biochemical endpoints", DATA / "biochemical.parquet")):
        frame = pd.read_parquet(path)
        work = contrasts(frame)
        print(f"{name}: {len(work)} substitutions at "
              f"{work.groupby(['protein_id', 'position']).ngroups} positions with at "
              f"least {MIN_SITE} measured, in {work['protein_id'].nunique()} proteins",
              flush=True)
        for scheme in ("pooled", "within protein"):
            rows += evaluate(name, work, scheme)
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / "X3_external_constraint_law.csv", index=False)
    show = ["resource", "split", "half", "variants", "positions", "pairs",
            "mean_constraint", "symmetric_share", "share_ci_low", "share_ci_high",
            "block_contrast", "block_ci_low", "block_ci_high"]
    print()
    print(table[show].to_string(index=False))

    print("\n-- the contrast that matters --")
    for (resource, scheme), part in table.groupby(["resource", "split"], sort=False):
        part = part.set_index("half")
        for label, column in (("symmetric share", "symmetric_share"),
                              ("block contrast", "block_contrast")):
            low = part.at["less constrained", column]
            high = part.at["more constrained", column]
            if np.isfinite(low) and np.isfinite(high):
                print(f"{resource:22s} {scheme:14s} {label:16s} "
                      f"{low:+.3f} -> {high:+.3f}  (change {high - low:+.3f})")
            else:
                print(f"{resource:22s} {scheme:14s} {label:16s} "
                      f"not estimable at this power")


if __name__ == "__main__":
    main()

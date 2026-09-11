from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROJECT = Path(__file__).resolve().parent.parent
DATA = PROJECT / "data"
RESULTS = PROJECT / "results"
SEED = 20260908
GROUPS = {"aliphatic": set("ILVM"), "aromatic": set("FYW"), "small_polar": set("STA"),
          "acidic": set("DE"), "basic": set("KR"), "amide": set("NQ")}
GROUP_OF = {r: g for g, members in GROUPS.items() for r in members}
BLOSUM_ROWS = """A R -1|A N -2|A D -2|A C 0|A Q -1|A E -1|A G 0|A H -2|A I -1|A L -1|A K -1
A M -1|A F -2|A P -1|A S 1|A T 0|A W -3|A Y -2|A V 0|R N 0|R D -2|R C -3|R Q 1|R E 0
R G -2|R H 0|R I -3|R L -2|R K 2|R M -1|R F -3|R P -2|R S -1|R T -1|R W -3|R Y -2
R V -3|N D 1|N C -3|N Q 0|N E 0|N G 0|N H 1|N I -3|N L -3|N K 0|N M -2|N F -3|N P -2
N S 1|N T 0|N W -4|N Y -2|N V -3|D C -3|D Q 0|D E 2|D G -1|D H -1|D I -3|D L -4|D K -1
D M -3|D F -3|D P -1|D S 0|D T -1|D W -4|D Y -3|D V -3|C Q -3|C E -4|C G -3|C H -3
C I -1|C L -1|C K -3|C M -1|C F -2|C P -3|C S -1|C T -1|C W -2|C Y -2|C V -1|Q E 2
Q G -2|Q H 0|Q I -3|Q L -2|Q K 1|Q M 0|Q F -3|Q P -1|Q S 0|Q T -1|Q W -2|Q Y -1|Q V -2
E G -2|E H 0|E I -3|E L -3|E K 1|E M -2|E F -3|E P -1|E S 0|E T -1|E W -3|E Y -2|E V -2
G H -2|G I -4|G L -4|G K -2|G M -3|G F -3|G P -2|G S 0|G T -2|G W -2|G Y -3|G V -3
H I -3|H L -3|H K -1|H M -2|H F -1|H P -2|H S -1|H T -2|H W -2|H Y 2|H V -3|I L 2
I K -3|I M 1|I F 0|I P -3|I S -2|I T -1|I W -3|I Y -1|I V 3|L K -2|L M 2|L F 0|L P -3
L S -2|L T -1|L W -2|L Y -1|L V 1|K M -1|K F -3|K P -1|K S 0|K T -1|K W -3|K Y -2
K V -2|M F 0|M P -2|M S -1|M T -1|M W -1|M Y -1|M V 1|F P -4|F S -2|F T -2|F W 1|F Y 3
F V -1|P S -1|P T -1|P W -4|P Y -3|P V -2|S T 1|S W -3|S Y -2|S V -2|T W -2|T Y -2
T V 0|W Y 2|W V -3|Y V -1"""
BLOSUM = {}
for entry in BLOSUM_ROWS.replace("\n", "|").split("|"):
    a, b, value = entry.split()
    BLOSUM[(a, b)] = BLOSUM[(b, a)] = float(value)


def pair_values(block: pd.DataFrame, minimum: int = 15) -> dict:
    work = block.dropna(subset=["effect_rank"]).copy()
    total = work.groupby(["protein_id", "position"])["effect_rank"].transform("sum")
    count = work.groupby(["protein_id", "position"])["effect_rank"].transform("size")
    keep = count >= 4
    work, total, count = work[keep], total[keep], count[keep]
    work["contrast"] = work["effect_rank"] - (total - work["effect_rank"]) / (count - 1)
    table = (work.groupby(["reference_residue", "alternate_residue"])["contrast"]
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


def agreement(values: dict, drop_worst: int, pair_class: str = "between") -> tuple:
    items = [(k, v) for k, v in values.items()
             if k[0] in GROUP_OF and k[1] in GROUP_OF
             and ((GROUP_OF[k[0]] != GROUP_OF[k[1]]) == (pair_class == "between"))]
    items.sort(key=lambda kv: kv[1])
    kept = items[drop_worst:]
    if len(kept) < 20:
        return np.nan, len(kept)
    measured = np.array([v for _, v in kept])
    reference = np.array([BLOSUM[k] for k, _ in kept])
    return float(spearmanr(measured, reference).statistic), len(kept)


def main() -> None:
    frame = pd.read_parquet(DATA / "abundance.parquet")
    frame = frame.dropna(subset=["observed_position_intolerance", "effect_rank"]).copy()
    frame["constraint_bin"] = (
        frame.groupby("protein_id")["observed_position_intolerance"]
        .transform(lambda s: pd.qcut(s.rank(method="first"), 5,
                                     labels=[f"C{i+1}" for i in range(5)]).astype(object)))
    values = {label: pair_values(frame[frame["constraint_bin"] == label])
              for label in ("C4", "C5")}

    proteins = np.array(sorted(frame["protein_id"].unique()))
    rng = np.random.default_rng(SEED)
    resampled = []
    for _ in range(200):
        picked = set(proteins[rng.integers(0, len(proteins), len(proteins))])
        part = frame[frame["protein_id"].isin(picked)]
        resampled.append({label: pair_values(part[part["constraint_bin"] == label])
                          for label in ("C4", "C5")})

    rows = []
    for drop in (0, 10, 20, 27, 35):
        entry = {"pairs_dropped": drop}
        for label in ("C4", "C5"):
            statistic, kept = agreement(values[label], drop)
            entry[f"{label}_agreement"] = statistic
            entry[f"{label}_pairs_kept"] = kept
        entry["C4_minus_C5"] = entry["C4_agreement"] - entry["C5_agreement"]
        gaps = []
        for draw in resampled:
            four, _ = agreement(draw["C4"], drop)
            five, _ = agreement(draw["C5"], drop)
            if np.isfinite(four) and np.isfinite(five):
                gaps.append(four - five)
        gaps = np.array(gaps)
        entry["gap_ci_low"] = float(np.percentile(gaps, 2.5))
        entry["gap_ci_high"] = float(np.percentile(gaps, 97.5))
        rows.append(entry)
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / "N1_saturation_test.csv", index=False)
    print("dropping the worst cross-class pairs, equally at C4 and C5")
    print("(27 is the count that the floor-mass share of 25.7 per cent implies at C5)")
    print(table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()

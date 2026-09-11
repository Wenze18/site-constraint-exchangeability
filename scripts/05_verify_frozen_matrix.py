from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROJECT = Path(__file__).resolve().parent.parent
DATA = PROJECT / "data"
RESULTS = PROJECT / "results"
SEED = 20260908

BLOSUM = {}
for line in """A R -1|A N -2|A D -2|A C 0|A Q -1|A E -1|A G 0|A H -2|A I -1|A L -1|A K -1
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
T V 0|W Y 2|W V -3|Y V -1""".replace("\n", "|").split("|"):
    a, b, value = line.split()
    BLOSUM[(a, b)] = BLOSUM[(b, a)] = float(value)


def frozen_scale(frame: pd.DataFrame, minimum: int = 25) -> dict:
    work = frame.dropna(subset=["effect_rank"]).copy()
    total = work.groupby(["protein_id", "position"])["effect_rank"].transform("sum")
    count = work.groupby(["protein_id", "position"])["effect_rank"].transform("size")
    work, total, count = work[count >= 4], total[count >= 4], count[count >= 4]
    work["contrast"] = work["effect_rank"] - (total - work["effect_rank"]) / (count - 1)
    table = (work.groupby(["reference_residue", "alternate_residue"])["contrast"]
             .agg(mean="mean", n="size").dropna())
    table = table[table["n"] >= minimum]
    scale = {}
    for a, b in table.index:
        if (b, a) not in table.index:
            continue
        key = tuple(sorted((a, b)))
        if key in scale:
            continue
        weights = np.array([table.at[(a, b), "n"], table.at[(b, a), "n"]], float)
        means = np.array([table.at[(a, b), "mean"], table.at[(b, a), "mean"]])
        scale[key] = float(np.average(means, weights=weights))
    return scale


def compare(block: pd.DataFrame, scale: dict, label: str) -> dict:
    default = float(np.mean(list(scale.values())))
    block = block.copy()
    block["exchange"] = [scale.get(tuple(sorted((str(a), str(b)))), default)
                         for a, b in zip(block["reference_residue"],
                                         block["alternate_residue"])]
    block["blosum"] = [BLOSUM.get((str(a), str(b)), 0.0)
                       for a, b in zip(block["reference_residue"],
                                       block["alternate_residue"])]
    exchange, blosum, clusters = [], [], []
    for (_, cluster), part in block.groupby(["assay_id", "homology_cluster"]):
        clean = part.dropna(subset=["effect_rank"])
        if len(clean) < 15 or clean["effect_rank"].nunique() < 5:
            continue
        one = spearmanr(clean["effect_rank"], clean["exchange"]).statistic
        two = spearmanr(clean["effect_rank"], clean["blosum"]).statistic
        if np.isfinite(one) and np.isfinite(two):
            exchange.append(one)
            blosum.append(two)
            clusters.append(cluster)
    exchange, blosum, clusters = map(np.array, (exchange, blosum, clusters))
    difference = exchange - blosum
    rng = np.random.default_rng(SEED)
    blocks = [difference[clusters == c] for c in np.unique(clusters)]
    index = rng.integers(0, len(blocks), (4000, len(blocks)))
    draws = np.array([np.concatenate([blocks[j] for j in row]).mean() for row in index])
    return {"subset": label, "assays": len(exchange),
            "clusters": len(np.unique(clusters)),
            "exchange": float(exchange.mean()), "blosum": float(blosum.mean()),
            "difference": float(difference.mean()),
            "ci_low": float(np.percentile(draws, 2.5)),
            "ci_high": float(np.percentile(draws, 97.5))}


def main() -> None:
    abundance = pd.read_parquet(DATA / "abundance.parquet")
    stability = pd.read_parquet(DATA / "stability.parquet")
    scale = frozen_scale(abundance)
    print(f"frozen scale covers {len(scale)} of 190 unordered pairs")

    shared = set(abundance["sequence_sha256"].dropna()) & set(stability["protein_id"])
    print(f"external stability proteins also present in the training block: {len(shared)}"
          f" of {stability['protein_id'].nunique()}")

    rows = [compare(stability, scale, "all external stability proteins"),
            compare(stability[~stability["protein_id"].isin(shared)], scale,
                    "excluding the 4 shared sequences")]
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / "R5_leakage_check.csv", index=False)
    print(table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()

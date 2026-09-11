from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DATA = PROJECT / "data"
RESULTS = PROJECT / "results"
CACHE = DATA / "uniprot_features.json"
SEED = 20260908

FIELDS = ["accession", "sequence", "ft_act_site", "ft_binding", "ft_site",
          "ft_dna_bind", "ft_mod_res", "ft_carbohyd", "ft_disulfid", "ft_lipid"]
FUNCTIONAL = {"Active site", "Binding site", "Site", "DNA binding"}
OTHER_ANNOTATION = {"Modified residue", "Glycosylation", "Disulfide bond", "Lipidation"}


def accessions(frame: pd.DataFrame) -> pd.DataFrame:
    table = frame[["protein_id"]].drop_duplicates().copy()
    parts = table["protein_id"].str.rsplit("_", n=2, expand=True)
    table["accession"] = parts[0]
    table["family"] = parts[1]
    table["domain_start"] = pd.to_numeric(parts[2], errors="coerce")
    return table


def fetch(frame: pd.DataFrame) -> dict:
    wanted = sorted(a for a in accessions(frame)["accession"].unique()
                    if not a.startswith(("EEHEE", "EHEE", "HHH", "HEEH")))
    cached = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    missing = [a for a in wanted if a not in cached]
    print(f"    {len(wanted)} accessions, {len(cached)} already cached, "
          f"{len(missing)} to fetch")
    for start in range(0, len(missing), 60):
        batch = missing[start:start + 60]
        query = " OR ".join(f"accession:{a}" for a in batch)
        url = ("https://rest.uniprot.org/uniprotkb/search?"
               + urllib.parse.urlencode({"query": query, "format": "json",
                                         "fields": ",".join(FIELDS), "size": 100}))
        for attempt in range(4):
            try:
                with urllib.request.urlopen(url, timeout=90) as response:
                    payload = json.loads(response.read().decode())
                break
            except (urllib.error.URLError, TimeoutError) as error:
                print(f"      retry {attempt + 1} after {error}")
                time.sleep(3 * (attempt + 1))
        else:
            print("      batch failed, leaving it uncached")
            continue
        for entry in payload.get("results", []):
            cached[entry["primaryAccession"]] = {
                "sequence": entry.get("sequence", {}).get("value", ""),
                "features": [{"type": f["type"],
                              "start": f["location"]["start"].get("value"),
                              "end": f["location"]["end"].get("value"),
                              "description": f.get("description", "")}
                             for f in entry.get("features", [])]}
        for accession in batch:
            cached.setdefault(accession, {"sequence": "", "features": []})
        CACHE.write_text(json.dumps(cached))
        print(f"      cached {len(cached)}/{len(wanted)}")
        time.sleep(0.5)
    return cached


def annotate(frame: pd.DataFrame, cached: dict) -> pd.DataFrame:
    keys = accessions(frame).set_index("protein_id")
    positions = (frame.drop_duplicates(["protein_id", "position"])
                 [["protein_id", "position", "reference_residue",
                   "observed_position_intolerance", "burial_class"]].copy())
    positions = positions.join(keys, on="protein_id")
    positions["uniprot_position"] = (positions["position"]
                                     + positions["domain_start"] - 1)

    site_index: dict[tuple[str, int], set[str]] = {}
    for accession, record in cached.items():
        for feature in record["features"]:
            start, end = feature["start"], feature["end"]
            if start is None or end is None or end - start > 30:
                continue
            for residue in range(int(start), int(end) + 1):
                site_index.setdefault((accession, residue), set()).add(feature["type"])

    def types(row):
        return site_index.get((row.accession, int(row.uniprot_position)), set())

    labels = [types(row) for row in positions.itertuples()]
    positions["functional_site"] = [bool(t & FUNCTIONAL) for t in labels]
    positions["other_annotation"] = [bool(t & OTHER_ANNOTATION) for t in labels]
    positions["annotation_types"] = ["|".join(sorted(t)) for t in labels]

    checked = []
    for row in positions.itertuples():
        sequence = cached.get(row.accession, {}).get("sequence", "")
        index = int(row.uniprot_position) - 1
        checked.append(sequence[index] == row.reference_residue
                       if 0 <= index < len(sequence) else None)
    positions["mapping_verified"] = checked
    return positions


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


def blosum_table() -> dict:
    lines = [ln for ln in BLOSUM62_TEXT.strip().splitlines() if ln.strip()]
    header = lines[0].split()
    return {(line.split()[0], column): float(value)
            for line in lines[1:]
            for column, value in zip(header, line.split()[1:])}


BLOSUM = blosum_table()


def leave_one_out(frame: pd.DataFrame, outcome: str = "effect_rank") -> pd.DataFrame:
    work = frame.dropna(subset=[outcome]).copy()
    total = work.groupby(["protein_id", "position"])[outcome].transform("sum")
    count = work.groupby(["protein_id", "position"])[outcome].transform("size")
    keep = count >= 4
    work, total, count = work[keep], total[keep], count[keep]
    work["contrast"] = work[outcome] - (total - work[outcome]) / (count - 1)
    return work


def blosum_agreement(block: pd.DataFrame, minimum: int = 15):
    from scipy.stats import spearmanr
    table = (leave_one_out(block)
             .groupby(["reference_residue", "alternate_residue"])["contrast"]
             .agg(mean="mean", n="size").dropna())
    table = table[table["n"] >= minimum]
    measured, reference = [], []
    seen = set()
    for a, b in table.index:
        if (b, a) not in table.index or (b, a) in seen:
            continue
        seen.add((a, b))
        weights = np.array([table.at[(a, b), "n"], table.at[(b, a), "n"]], float)
        means = np.array([table.at[(a, b), "mean"], table.at[(b, a), "mean"]])
        measured.append(float(np.average(means, weights=weights)))
        reference.append(BLOSUM[(a, b)])
    if len(measured) < 20:
        return np.nan, len(measured)
    return float(spearmanr(measured, reference).statistic), len(measured)


def analyse(frame: pd.DataFrame, positions: pd.DataFrame) -> None:
    verified = positions["mapping_verified"]
    print(f"    mapping check: {int(verified.sum())} of {int(verified.notna().sum())} "
          f"positions match the UniProt residue "
          f"({100 * verified.mean():.1f}% of those checkable)")
    positions = positions[verified.fillna(False)].copy()
    positions["constraint_bin"] = (
        positions.groupby("protein_id")["observed_position_intolerance"]
        .transform(lambda s: pd.qcut(s.rank(method="first"), 5,
                                     labels=[f"C{i + 1}" for i in range(5)]).astype(object)))
    labelled = frame.merge(
        positions[["protein_id", "position", "constraint_bin", "functional_site",
                   "other_annotation"]], on=["protein_id", "position"], how="inner")

    print("\n  1  are functional sites concentrated at the most constrained positions?")
    enrichment = (positions.groupby("constraint_bin")
                  .agg(positions=("functional_site", "size"),
                       functional=("functional_site", "sum"),
                       other=("other_annotation", "sum")))
    enrichment["functional_fraction"] = enrichment["functional"] / enrichment["positions"]
    enrichment["other_fraction"] = enrichment["other"] / enrichment["positions"]
    enrichment.to_csv(RESULTS / "R1_functional_enrichment.csv")
    print(enrichment.round(4).to_string())

    print("\n  2  BLOSUM agreement by constraint, all positions and with functional "
          "sites removed")
    rows = []
    for label in [f"C{i + 1}" for i in range(5)]:
        block = labelled[labelled["constraint_bin"] == label]
        full, pairs_full = blosum_agreement(block)
        without, pairs_without = blosum_agreement(block[~block["functional_site"]])
        only, pairs_only = blosum_agreement(block[block["functional_site"]])
        control, pairs_control = blosum_agreement(block[~block["other_annotation"]])
        rows.append({"constraint_bin": label,
                     "agreement_all": full, "pairs_all": pairs_full,
                     "agreement_without_functional": without,
                     "pairs_without_functional": pairs_without,
                     "agreement_functional_only": only,
                     "pairs_functional_only": pairs_only,
                     "agreement_without_other_annotation": control,
                     "pairs_without_other": pairs_control})
    curve = pd.DataFrame(rows)
    curve.to_csv(RESULTS / "R1_blosum_agreement_by_annotation.csv", index=False)
    print(curve.round(3).to_string(index=False))

    print("\n  3  is the C5 fall attributable to functional sites? permutation control")
    top = labelled[labelled["constraint_bin"] == "C5"]
    observed_without, _ = blosum_agreement(top[~top["functional_site"]])
    observed_all, _ = blosum_agreement(top)
    site_positions = top.drop_duplicates(["protein_id", "position"])
    share = float(site_positions["functional_site"].mean())
    rng = np.random.default_rng(SEED)
    null = []
    keys = site_positions[["protein_id", "position"]].to_numpy()
    for _ in range(400):
        drop = rng.random(len(keys)) < share
        removed = set(map(tuple, keys[drop]))
        mask = [(p, q) not in removed for p, q in
                zip(top["protein_id"], top["position"])]
        value, pairs = blosum_agreement(top[np.array(mask)])
        if np.isfinite(value):
            null.append(value)
    null = np.array(null)
    summary = pd.DataFrame([{
        "quintile": "C5", "agreement_all": observed_all,
        "agreement_removing_functional": observed_without,
        "shift": observed_without - observed_all,
        "matched_random_removal_mean": float(null.mean()),
        "matched_random_ci_low": float(np.percentile(null, 2.5)),
        "matched_random_ci_high": float(np.percentile(null, 97.5)),
        "permutation_p": float((null >= observed_without).mean()),
        "functional_share_of_positions": share}])
    summary.to_csv(RESULTS / "R1_c5_permutation.csv", index=False)
    print(summary.round(4).to_string(index=False))

    print("\n  4  direct test at annotated sites, against a constraint-matched control")
    matched = matched_site_test(labelled, positions)
    matched.to_csv(RESULTS / "R1_matched_site_test.csv", index=False)
    print(matched.round(4).to_string(index=False))

    print("\n  5  is the top quintile simply measured less reliably?")
    reliability = reliability_by_constraint(labelled)
    reliability.to_csv(RESULTS / "R1_reliability_by_constraint.csv", index=False)
    print(reliability.round(4).to_string(index=False))

    print("\n  6  where in the matrix is the agreement lost?")
    split = within_between_agreement(labelled)
    split.to_csv(RESULTS / "R1_within_between_agreement.csv", index=False)
    print(split.round(4).to_string(index=False))


def matched_site_test(labelled: pd.DataFrame, positions: pd.DataFrame) -> pd.DataFrame:
    annotated = positions[positions["functional_site"]]
    pool = positions[~positions["functional_site"] & ~positions["other_annotation"]]
    deciles = pd.qcut(positions["observed_position_intolerance"].rank(method="first"),
                      10, labels=False)
    positions = positions.assign(decile=deciles)
    annotated = positions[positions["functional_site"]]
    pool = positions[~positions["functional_site"] & ~positions["other_annotation"]]
    quota = annotated.groupby("decile").size()

    keyed = labelled.set_index(["protein_id", "position"]).sort_index()

    def agreement_for(selection, minimum):
        keys = list(zip(selection["protein_id"], selection["position"]))
        return blosum_agreement(keyed.loc[keys].reset_index(), minimum=minimum)

    rng = np.random.default_rng(SEED)
    rows = []
    for minimum in (5, 8):
        observed, pairs_observed = agreement_for(annotated, minimum)
        control = []
        for _ in range(300):
            chosen = []
            for decile, take in quota.items():
                candidates = pool[pool["decile"] == decile]
                if len(candidates) == 0 or take <= 0:
                    continue
                chosen.append(candidates.iloc[rng.integers(0, len(candidates), int(take))])
            value, _ = agreement_for(pd.concat(chosen), minimum)
            if np.isfinite(value):
                control.append(value)
        control = np.array(control)
        spread = []
        for _ in range(300):
            picked = annotated.iloc[rng.integers(0, len(annotated), len(annotated))]
            value, _ = agreement_for(picked, minimum)
            if np.isfinite(value):
                spread.append(value)
        spread = np.array(spread)
        rows.append({
            "min_observations_per_direction": minimum,
            "annotated_positions": len(annotated),
            "agreement_functional": observed, "pairs_functional": pairs_observed,
            "functional_ci_low": float(np.percentile(spread, 2.5)) if len(spread) else np.nan,
            "functional_ci_high": float(np.percentile(spread, 97.5)) if len(spread) else np.nan,
            "agreement_constraint_matched_control": float(control.mean()),
            "control_ci_low": float(np.percentile(control, 2.5)),
            "control_ci_high": float(np.percentile(control, 97.5)),
            "difference": observed - float(control.mean()),
            "permutation_p_lower": float((control <= observed).mean())})
    return pd.DataFrame(rows)


def reliability_by_constraint(labelled: pd.DataFrame) -> pd.DataFrame:
    from scipy.stats import spearmanr
    rng = np.random.default_rng(SEED)
    proteins = np.array(sorted(labelled["protein_id"].unique()))
    rows = []
    for label in [f"C{i + 1}" for i in range(5)]:
        block = labelled[labelled["constraint_bin"] == label]
        agreements, reliabilities, spreads = [], [], []
        for _ in range(60):
            shuffled = rng.permutation(proteins)
            half = set(shuffled[:len(shuffled) // 2])
            first = block[block["protein_id"].isin(half)]
            second = block[~block["protein_id"].isin(half)]
            values = []
            for part in (first, second):
                table = (leave_one_out(part)
                         .groupby(["reference_residue", "alternate_residue"])["contrast"]
                         .agg(mean="mean", n="size").dropna())
                table = table[table["n"] >= 8]
                pairs, seen = {}, set()
                for a, b in table.index:
                    if (b, a) not in table.index or (b, a) in seen:
                        continue
                    seen.add((a, b))
                    weights = np.array([table.at[(a, b), "n"], table.at[(b, a), "n"]], float)
                    means = np.array([table.at[(a, b), "mean"], table.at[(b, a), "mean"]])
                    pairs[(a, b)] = float(np.average(means, weights=weights))
                values.append(pairs)
            shared = sorted(set(values[0]) & set(values[1]))
            if len(shared) < 40:
                continue
            left = np.array([values[0][k] for k in shared])
            right = np.array([values[1][k] for k in shared])
            reference = np.array([BLOSUM[k] for k in shared])
            reliabilities.append(spearmanr(left, right).statistic)
            agreements.append(np.mean([spearmanr(left, reference).statistic,
                                       spearmanr(right, reference).statistic]))
            spreads.append(np.std(np.concatenate([left, right])))
        if not reliabilities:
            continue
        reliability = float(np.mean(reliabilities))
        agreement = float(np.mean(agreements))
        rows.append({"constraint_bin": label,
                     "split_half_reliability": reliability,
                     "half_sample_blosum_agreement": agreement,
                     "disattenuated_agreement": agreement / np.sqrt(max(reliability, 1e-6))
                     if reliability > 0 else np.nan,
                     "exchange_value_spread": float(np.mean(spreads))})
    return pd.DataFrame(rows)


GROUPS = {"aliphatic": set("ILVM"), "aromatic": set("FYW"), "small_polar": set("STA"),
          "acidic": set("DE"), "basic": set("KR"), "amide": set("NQ")}
GROUP_OF = {residue: name for name, members in GROUPS.items() for residue in members}


def within_between_agreement(labelled: pd.DataFrame) -> pd.DataFrame:
    from scipy.stats import spearmanr
    rng = np.random.default_rng(SEED)
    proteins = np.array(sorted(labelled["protein_id"].unique()))
    rows = []
    for label in [f"C{i + 1}" for i in range(5)]:
        block = labelled[labelled["constraint_bin"] == label]
        table = (leave_one_out(block)
                 .groupby(["reference_residue", "alternate_residue"])["contrast"]
                 .agg(mean="mean", n="size").dropna())
        table = table[table["n"] >= 15]
        pairs, seen = {}, set()
        for a, b in table.index:
            if (b, a) not in table.index or (b, a) in seen:
                continue
            seen.add((a, b))
            weights = np.array([table.at[(a, b), "n"], table.at[(b, a), "n"]], float)
            means = np.array([table.at[(a, b), "mean"], table.at[(b, a), "mean"]])
            pairs[(a, b)] = float(np.average(means, weights=weights))
        classified = {"within class": [], "between class": []}
        for (a, b), value in pairs.items():
            if a not in GROUP_OF or b not in GROUP_OF:
                continue
            key = "within class" if GROUP_OF[a] == GROUP_OF[b] else "between class"
            classified[key].append((value, BLOSUM[(a, b)]))
        for subset, entries in classified.items():
            if len(entries) < 8:
                continue
            measured = np.array([e[0] for e in entries])
            reference = np.array([e[1] for e in entries])
            statistic = spearmanr(measured, reference).statistic
            draws = []
            for _ in range(2000):
                index = rng.integers(0, len(measured), len(measured))
                if len(np.unique(reference[index])) < 3:
                    continue
                draws.append(spearmanr(measured[index], reference[index]).statistic)
            draws = np.array([d for d in draws if np.isfinite(d)])
            rows.append({"constraint_bin": label, "pair_class": subset,
                         "pairs": len(entries),
                         "blosum_agreement": float(statistic),
                         "ci_low": float(np.percentile(draws, 2.5)),
                         "ci_high": float(np.percentile(draws, 97.5)),
                         "mean_exchange": float(measured.mean()),
                         "exchange_spread": float(measured.std(ddof=1))})
    return pd.DataFrame(rows)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(DATA / "abundance.parquet")
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    cached = fetch(frame) if stage in ("fetch", "all") else json.loads(CACHE.read_text())
    if stage == "fetch":
        return
    positions = annotate(frame, cached)
    positions.to_csv(RESULTS / "R1_position_annotations.csv", index=False)
    analyse(frame, positions)


if __name__ == "__main__":
    main()

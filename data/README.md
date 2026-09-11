# Input data

The parquet files are not redistributed here. They are built from published resources,
all of which are freely available:

| File | Built from | Source |
|---|---|---|
| `abundance.parquet` | Human Domainome 1 site-saturation abundance measurements | https://doi.org/10.1038/s41586-024-08370-4 |
| `stability.parquet` | VenusMutHub cross-source stability compilation and the stability portion of the broad benchmark | https://doi.org/10.1016/j.apsb.2025.03.028 |
| `biochemical.parquet` | the non-cross-source portions of the broad benchmark: stability (9,376 rows), protein-protein binding (2,298), activity (1,725), selectivity (708) and drug-target binding (358). The stability portion is the second external resource and is the part most of the code uses | https://doi.org/10.1016/j.apsb.2025.03.028 |
| `multi_mutant_stability_variants.parquet` | PTMUL multi-substitution stability variants, used by the additivity test in `03_` | https://doi.org/10.64898/2026.06.04.730231 |

In the first three files a row is one single substitution; in the multi-substitution
file a row is one variant carrying several. The columns the analysis uses are

    protein_id  protein_family  sequence_sha256  position
    reference_residue  alternate_residue  effect_rank
    observed_position_intolerance  burial_class  backbone_state
    assay_id  endpoint_class  block

`effect_rank` is the within-assay percentile of the measured outcome, so that assays on
different scales can be pooled. `observed_position_intolerance` is the negated mean
`effect_rank` at the position. `endpoint_class` selects the portion of a file: the code
reads `stability` out of `biochemical.parquet` for the second external resource, and the
other classes only in the absorption analysis. `homology_cluster`, which is present in
`stability.parquet` only, is the resampling unit for intervals on that resource; where it
is absent the code falls back to `protein_id`. `grantham_distance` and the four `delta_*`
descriptors are read by the baseline ladder.

Five predictor scores are carried per variant, frozen at their published values and never
refitted. They are the comparison set for the absorption analysis:

| column | model | source |
|---|---|---|
| `structure_sequence_expert_rank` | SPURS | https://doi.org/10.1038/s41467-025-67609-4 |
| `structure_language_expert_rank` | SaProt-DeltaG | https://doi.org/10.1101/2023.10.01.560349 |
| `inverse_folding_expert_rank` | ThermoMPNN | https://doi.org/10.1073/pnas.2314853121 |
| `stability_language_expert_rank` | Pythia | https://doi.org/10.1016/j.xinn.2024.100750 |
| `sequence_language_expert_rank` | ESM-2 | https://doi.org/10.1126/science.ade2574 |

Two files are included rather than downloaded.

`uniprot_features.json` is the residue-level feature annotation for the 419 UniProt
accessions behind the 503 domains, as retrieved from the UniProt REST API and used in the
paper. It is shipped because UniProt is revised continuously: `04_functional_sites.py`
will fetch the annotations itself if the file is absent, but a later fetch returns a later
snapshot and need not give the published numbers. UniProt is distributed under CC BY 4.0.
This is the only script that touches the network, and only when the file is missing.

`substitution_matrices.csv` is included, not downloaded. It holds BLOSUM45, BLOSUM62,
BLOSUM80, PAM250 and BENNER74, 190 unordered pairs each, exported from the matrices
bundled with Biopython, which redistributes them from the original publications: BLOSUM45,
BLOSUM62 and BLOSUM80 from Henikoff and Henikoff (1992), https://doi.org/10.1073/pnas.89.22.10915;
PAM250 from Dayhoff, Schwartz and Orcutt (1978); BENNER74 from Benner, Cohen and Gonnet (1994),
https://doi.org/10.1006/jmbi.1994.1146. Grantham distance is not in that file; it is carried per
variant in the parquet inputs, from Grantham (1974), https://doi.org/10.1126/science.185.4154.862.

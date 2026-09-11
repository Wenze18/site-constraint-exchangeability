# The measured exchange scale

Both files are copies of pipeline output, not separate artefacts. `exchange_matrix.csv` is
`results/R5_frozen_exchange_matrix.csv`, written by `03_exchange_matrix_and_validation.py`;
`compact_41.csv` is `results/R4_residue_coordinates.csv`, written by `07_rank_one_axis.py`;
and the weights below are the `interaction_weight` column of `results/R4_rank1_axis.csv`,
also from `07_`. Running the pipeline regenerates all three.

`exchange_matrix.csv` holds all 190 unordered residue pairs, marginal over burial and
again conditioned on each burial class. Columns: `burial_class`, which is `any` for the
marginal matrix and otherwise `buried`, `intermediate` or `exposed`; `residue_a` and
`residue_b`, one-letter codes in alphabetical order; and `exchange`, in units of the
within-position contrast on the within-assay percentile scale. Higher means more readily
interchanged. Use this file for lookups.

`compact_41.csv` holds the rank-one form: one additive replaceability offset and one
interaction axis coordinate per residue, plus a shared weight, which is 20 + 20 + 1 = 41
numbers per stratum. It covers three of the four strata; `intermediate` has no compact form
because `07_rank_one_axis.py` fits only the marginal, buried and exposed matrices. The
`all positions` stratum is the same set of positions as `burial_class == "any"` in the other
file, under a different name. A pair value is

    exchange(a, b) = intercept + r[a] + r[b] + weight * x[a] * x[b]

| stratum | matches `burial_class` | weight | intercept | Spearman against the full matrix |
|---|---|---:|---:|---:|
| all positions | `any` | 1.02261 | -0.002689 | 0.910 |
| buried | `buried` | 0.96650 | -0.004741 | 0.846 |
| exposed | `exposed` | 0.82787 | -0.019249 | 0.961 |

The intercept is a constant and does not affect the ordering of substitutions. The
compact form preserves the ordering of the full table but not its spacing: individual
entries can differ from the measured value by more than the gap between neighbouring
pairs, so it is the form to use when what you want is the structure or a parametric term,
not when you want the value of one pair.

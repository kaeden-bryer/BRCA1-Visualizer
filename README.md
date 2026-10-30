# BRCT Variant Explorer

An interactive deep mutational scan of the **BRCA1 BRCT domain** (residues 1646–1859).
It scores every possible amino-acid substitution with the open-weight protein language
model **ESM-2**, then checks those predictions against **experimental function scores**
from BRCA1 saturation genome editing (Findlay et al., *Nature* 2018).

Built for MLH Hacktoberfest (open-source / open-weight AI in biology).

> **Current status:** the `esm_score` column in `data/variant_scores.csv` is
> **placeholder random data**. The real ESM-2 scan has not been run yet. Everything else
> is real: the wild-type sequence, the experimental SGE scores and the PDB structure. The
> displayed AUROC (~0.51) therefore reflects random scores, not the model. To produce real
> scores, see [Regenerating the data](#regenerating-the-data).

## What it shows

- **Mutational heatmap:** position × amino acid, coloured by ESM-2 score (red = predicted
  damaging, blue = tolerated). Wild-type cells are marked.
- **3D structure:** the BRCT tandem repeat (PDB 1JNX) coloured by average sensitivity at
  each position. Clicking a heatmap cell (or picking a residue in the sidebar) highlights
  that residue.
- **Validation panel:** a scatter plot of ESM-2 score against the experimental SGE score,
  and an ROC curve for how well ESM-2 separates loss-of-function (LOF) variants from
  functional (FUNC) ones.

## Method

**ESM-2 masked-marginal scoring.** For each position *i*, mask it, run one forward pass,
and read the model's probabilities over the 20 amino acids:

```
score(i, mut) = log P(mut at i) − log P(wt at i)
```

Strongly negative scores mean evolution has conserved the wild-type residue, so a
substitution is probably damaging. The scan needs 214 forward passes. The model is
`esm2_t33_650M_UR50D`, with `esm2_t30_150M_UR50D` as a faster CPU fallback.

**Experimental ground truth.** SGE function scores are log2-scaled, with the median
synonymous variant set to 0. Classes: `FUNC` > −0.748, `INT` between −0.748 and −1.328,
`LOF` < −1.328. Several DNA changes can encode the same amino-acid substitution; their
scores are averaged. The result is 1,178 unique BRCT missense substitutions with
experimental data.

ESM-2 and SGE scores both treat "more negative" as "worse", so they should correlate
positively. AUROC treats LOF as the positive class and uses `−esm_score` as the predictor.

**Why not fold the mutants?** Folding the wild type and each mutant with AlphaFold or
ESMFold, then comparing the structures, shows almost nothing. A single substitution gives
well under 1 Å RMSD even when it is known to destroy function. Likelihood scores respond
to mutations; predicted structures don't.

## Residue numbering

All data uses **full-length BRCA1 numbering (UniProt P38398)**. PDB 1JNX (chain X,
residues 1649–1859) uses the same numbering, so no offset is applied. At startup, the app
checks that the wild-type residue at every position matches the structure. If any don't
match, it refuses to render the 3D view rather than colour the wrong residues.

## Running the app

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

The app reads only local files (`data/variant_scores.csv`, `data/metrics.json` and
`data/1JNX.pdb`). It needs no GPU, no network access and no model at runtime.

## Regenerating the data

```bash
# Placeholder data: real WT sequence + real SGE scores, random ESM scores (instant)
python3 scripts/generate_stubs.py

# Experimental SGE scores from Findlay et al. Supplementary Table 3 → cache/sge_scores.csv
python3 scripts/fetch_sge_from_paper.py

# Full pipeline: ESM-2 scan → join with SGE → AUROC → data/*.csv, data/metrics.json
python3 scripts/run_pipeline.py [--force]
```

The full pipeline caches intermediate results in `cache/`. The ESM-2 scan is the slow
step. `notebooks/run_pipeline.ipynb` runs the same pipeline on Google Colab when a GPU is
available.

The MaveDB deposit of the Findlay dataset is nucleotide-level only (no protein-level
HGVS). That's why the SGE scores come from the paper's supplementary table.

## Project layout

```
app/streamlit_app.py        Streamlit UI (heatmap, 3D viewer, validation panel)
pipeline/config.py          Domain bounds, WT sequence, thresholds, model names
pipeline/esm_scoring.py     ESM-2 masked-marginal scan
pipeline/sge_data.py        SGE data loading / MaveDB access
pipeline/join_and_evaluate.py  Join ESM + SGE, compute AUROC, write outputs
scripts/                    Entry points (pipeline, SGE fetch, stub generator)
notebooks/                  Colab notebook for the one-off GPU precompute
data/                       Files the app reads
cache/                      Intermediate pipeline outputs
```

Output schema (`data/variant_scores.csv`):

| column | meaning |
|---|---|
| `position` | BRCA1 residue number (UniProt P38398) |
| `wt_aa`, `mut_aa` | one-letter amino-acid codes |
| `esm_score` | ESM-2 log-ratio; negative = predicted damaging |
| `sge_score` | experimental SGE score; empty if not assayed |
| `sge_class` | `FUNC` / `INT` / `LOF`; empty if not assayed |

## Citations and licensing

- **SGE data:** Findlay G.M. *et al.* "Accurate classification of BRCA1 variants with
  saturation genome editing." *Nature* 562, 217–222 (2018).
  [doi:10.1038/s41586-018-0461-z](https://doi.org/10.1038/s41586-018-0461-z).
  SGE function scores and data are copyright University of Washington.
- **ESM-2:** Lin Z. *et al.* "Evolutionary-scale prediction of atomic-level protein
  structure with a language model." *Science* 379, 1123–1130 (2023). Model weights and
  code ([facebookresearch/esm](https://github.com/facebookresearch/esm)) are MIT licensed.
- **Structure:** PDB [1JNX](https://www.rcsb.org/structure/1JNX), Williams R.S. *et al.*
  "Crystal structure of the BRCT repeat region from the breast cancer-associated protein
  BRCA1." *Nat. Struct. Biol.* 8, 838–842 (2001). PDB data is released under CC0 1.0.
- **Sequence:** UniProt [P38398](https://www.uniprot.org/uniprotkb/P38398) (CC BY 4.0).

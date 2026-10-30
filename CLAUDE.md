# BRCT Variant Explorer — Project Brief

Hackathon project (MLH Hacktoberfest, open-source / open-weight AI in biology).
**Hard constraint: ~4 hours total.** Scope discipline matters more than completeness.

---

## What we are building

An interactive deep mutational scan of the **BRCA1 BRCT domain**.

1. Score every possible amino-acid substitution across the BRCT domain using
   **ESM-2** (open-weight protein language model) masked-marginal scoring.
2. Render it as an interactive **position × amino-acid heatmap**.
3. Clicking a cell colors a **3D BRCT structure** by sensitivity and highlights the residue.
4. A side panel compares the model's prediction against **real experimental function
   scores** from BRCA1 saturation genome editing (Findlay et al., Nature 2018),
   including an ROC curve.

The validation-against-wet-lab-data panel is the differentiator. A demo that grades
itself against experimental ground truth beats a demo that only renders something.

---

## What we are explicitly NOT building (and why)

**Do not build a "fold WT vs fold mutant, show structural diff" demo.**
Single amino-acid substitutions produce backbone RMSDs well under 1 Å in
AlphaFold/ESMFold predictions, even for mutations known experimentally to destroy the
protein. These models predict the native fold the sequence *belongs to*, not the fold it
would actually achieve. The two structures will overlay perfectly and the demo will show
nothing. This was evaluated and rejected deliberately — do not reintroduce it.

What *does* respond to mutations: PLM likelihood scores (what we use), per-residue pLDDT,
and ΔΔG stability predictors.

**Do not attempt protein–DNA or G-quadruplex binding affinity prediction.** It was
scoped out. Boltz-2's affinity head covers protein–small molecule only, and co-folding
models are documented to fail on G-quadruplex topology.

---

## The core method (conceptually)

ESM-2 masked-marginal scoring: mask position *i*, read the model's distribution over the
20 amino acids at that position, and score a substitution as

```
score = log P(mutant_aa at i) - log P(wildtype_aa at i)
```

Strongly negative => evolution has conserved the wild-type residue => substituting it
probably breaks the protein. ~15 lines of core logic. One forward pass per masked
position (~210 passes for this domain).

---

## Stack

- **Model:** ESM-2. Try `esm2_t33_650M_UR50D` first; fall back to `esm2_t30_150M_UR50D`
  if too slow.
- **Compute:** try CPU first — ~210 forward passes is plausibly 5–15 min on CPU and
  removes a whole dependency. Google Colab (via the official Colab VS Code extension)
  is the fallback, not the default. User is on an older Intel Mac.
- **App:** Streamlit.
- **3D viewer:** `py3Dmol` via `stmol`.
- **Hosting:** Hugging Face Spaces (free, public URL for the demo).
- **Notebook vs scripts:** notebook ONLY for the one-time precompute and the exploratory
  ROC check. The app is plain `.py` files in the repo.

---

## THE FROZEN DATA CONTRACT

Everything parallelizes off this. Generate stub CSVs with random values matching this
schema **first**, so app work can begin before any model has run.

`data/variant_scores.csv`
```
position,wt_aa,mut_aa,esm_score,sge_score,sge_class
```
- `position` — integer, in a single declared coordinate system (see Gotcha 1)
- `wt_aa`, `mut_aa` — single-letter amino acid codes
- `esm_score` — float, log-ratio from ESM-2 (negative = predicted damaging)
- `sge_score` — float, experimental; may be NaN where not assayed
- `sge_class` — one of `FUNC` / `INT` / `LOF`; may be empty

`data/metrics.json`
```
{"auroc": float, "n_variants": int, "n_with_sge": int, "model": str}
```

---

## Gotchas — read before writing code

**1. Residue numbering is the silent project-killer.**
BRCA1 full-protein numbering, the BRCT construct's local numbering, and the PDB file's
author numbering will NOT agree. An off-by-N error means heatmap clicks highlight the
wrong residue in the 3D viewer and nothing visibly errors. Pick ONE canonical coordinate
system, write it down at the top of the pipeline module, and convert at the boundaries
only. Add an assertion that the wild-type amino acid at each position matches the
residue in the PDB structure — this catches offset bugs immediately.

**2. Prefer the amino-acid score set over the nucleotide score set.**
MaveDB commonly stores both. The nucleotide set forces you to map SNVs through the
transcript to amino-acid substitutions yourself — the single biggest time sink in this
project. The amino-acid set joins directly to ESM-2 output on (position, mut_aa).

**3. Cache every external dependency to disk in the first 15 minutes.**
Commit the CSVs. The live demo must touch nothing but local files — no GPU, no network
call, no model loading at demo time. Hackathon demos die from exactly those three things.

**4. If the ESM-2 vs SGE correlation is weak, do not debug it.**
Ship the app and report the correlation honestly as a finding. "Here is what we measured"
is a legitimate result. A half-finished UI is not.

---

## Data sources (in fallback order)

1. **MaveDB** (mavedb.org) — primary. Has a real API; the 2024 update added a data model
   explicitly supporting saturation genome editing. Findlay BRCA1 SGE is deposited here.
2. **Findlay et al. 2018 supplementary tables** — DOI `10.1038/s41586-018-0461-z`.
   Permanent, server-independent. Open-access preprint: bioRxiv `10.1101/294520`.
3. **sge.gs.washington.edu/BRCA1** — the UW site; was intermittently unreachable.
4. **ClinVar** — weakest fallback. Clinical interpretations, not functional measurements,
   and BRCA1 specifically has many variants with conflicting expert interpretations.

**SGE score interpretation:** log2-scaled, median synonymous SNV = 0. Thresholds:
`FUNC` > −0.748; `INT` between −0.748 and −1.328; `LOF` < −1.328. Median nonsense SNV
scored −2.12. More negative = loss of function. Original dataset: 3,893 SNVs across the
13 exons encoding the RING and BRCT domains.

Note the orientation mismatch: ESM-2 scores and SGE scores are both "more negative =
worse," so they should correlate *positively*. Verify this before building the ROC curve
— a sign flip will silently invert your AUROC.

**Structure:** pull a BRCA1 BRCT tandem-repeat structure from the PDB. 1JNX (apo BRCT
tandem) and 1T29 (BACH1 phosphopeptide complex) are candidates — VERIFY these IDs before
relying on them; they are from recall, not checked.

---

## Build phases

**Phase 0 (serial, ~15 min, no subagents)**
Repo skeleton, `requirements.txt`, frozen schema above, stub-data generator.

**Phase 1 (parallel)**
- Agent A owns `pipeline/` — MaveDB fetch, ESM-2 scan, join, ROC. Produces real CSVs.
- Agent B owns `app/` — Streamlit UI built against stub CSVs.
- Human/lead owns structure selection + the numbering map (Gotcha 1).

Agents must not write outside their directory. The CSV contract is the only interface.

**Phase 2 (serial)** — swap stubs for real data. If the contract held, this is a file copy.

**Phase 3 (parallel, only once something works)** — deploy to Spaces; write README.
Documentation goes LAST, so it describes what exists rather than what was planned.

---

## Licensing / attribution (for the README)

- Cite Findlay et al., Nature 2018 (DOI `10.1038/s41586-018-0461-z`) for SGE scores.
  SGE function scores and data are copyright University of Washington.
- Cite MaveDB if used as the access route.
- Cite ESM-2 (Meta / Lin et al.) — MIT licensed.
- State the license of any PDB structure used.
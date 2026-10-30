"""Generate stub data matching the frozen CSV contract.

Produces:
    data/variant_scores.csv   — random ESM + SGE scores for every position x mutation
    data/metrics.json         — placeholder summary metrics

The BRCT tandem-repeat domain spans roughly residues 1646–1859 of full-length BRCA1.
We use full-protein numbering as the canonical coordinate system (see CLAUDE.md Gotcha 1).
"""

import csv
import json
import math
import os
import random
import string

# ---------------------------------------------------------------------------
# Constants — canonical coordinate system is full-protein BRCA1 numbering
# ---------------------------------------------------------------------------
BRCT_START = 1646
BRCT_END = 1859  # inclusive

# Wild-type sequence for the BRCT domain (BRCA1 1646-1859, UniProt P38398).
# This is a placeholder — the real pipeline must assert-match against the PDB.
# For stub purposes we use a random realistic sequence of the right length.
random.seed(42)
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
WT_SEQUENCE = "".join(random.choice(AMINO_ACIDS) for _ in range(BRCT_END - BRCT_START + 1))

# SGE thresholds from Findlay et al. 2018
SGE_FUNC_THRESHOLD = -0.748
SGE_LOF_THRESHOLD = -1.328


def classify_sge(score: float) -> str:
    if score > SGE_FUNC_THRESHOLD:
        return "FUNC"
    elif score > SGE_LOF_THRESHOLD:
        return "INT"
    else:
        return "LOF"


def generate_variant_scores(out_path: str) -> dict:
    """Write stub variant_scores.csv and return summary stats."""
    random.seed(0)
    rows = []

    for i, wt_aa in enumerate(WT_SEQUENCE):
        position = BRCT_START + i
        for mut_aa in AMINO_ACIDS:
            if mut_aa == wt_aa:
                continue  # skip synonymous

            # Stub ESM score: normally distributed around -1
            esm_score = round(random.gauss(-1.0, 2.0), 4)

            # ~70% of variants have SGE data; rest are NaN
            if random.random() < 0.7:
                sge_score = round(random.gauss(-0.5, 1.5), 4)
                sge_class = classify_sge(sge_score)
            else:
                sge_score = ""
                sge_class = ""

            rows.append({
                "position": position,
                "wt_aa": wt_aa,
                "mut_aa": mut_aa,
                "esm_score": esm_score,
                "sge_score": sge_score,
                "sge_class": sge_class,
            })

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["position", "wt_aa", "mut_aa", "esm_score", "sge_score", "sge_class"]
        )
        writer.writeheader()
        writer.writerows(rows)

    n_with_sge = sum(1 for r in rows if r["sge_score"] != "")
    return {
        "n_variants": len(rows),
        "n_with_sge": n_with_sge,
    }


def generate_metrics(stats: dict, out_path: str) -> None:
    """Write stub metrics.json."""
    metrics = {
        "auroc": round(random.uniform(0.65, 0.85), 4),
        "n_variants": stats["n_variants"],
        "n_with_sge": stats["n_with_sge"],
        "model": "esm2_t33_650M_UR50D",
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Wrote {out_path}: {metrics}")


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(repo_root, "data", "variant_scores.csv")
    metrics_path = os.path.join(repo_root, "data", "metrics.json")

    stats = generate_variant_scores(csv_path)
    print(f"Wrote {csv_path}: {stats['n_variants']} variants, {stats['n_with_sge']} with SGE")

    generate_metrics(stats, metrics_path)


if __name__ == "__main__":
    main()

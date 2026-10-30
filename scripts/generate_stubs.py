"""Generate stub data matching the frozen CSV contract.

Produces:
    data/variant_scores.csv   — random ESM scores + real SGE scores (from cache/) for
                                every position x mutation
    data/metrics.json         — summary metrics; "model" is marked STUB

The WT sequence is the real one from pipeline/config.py so the app's PDB numbering
check passes. Only esm_score is fake — replace with the real ESM-2 scan when it runs.

The BRCT tandem-repeat domain spans roughly residues 1646–1859 of full-length BRCA1.
We use full-protein numbering as the canonical coordinate system (see CLAUDE.md Gotcha 1).
"""

import csv
import json
import math
import os
import random
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
from pipeline.config import AMINO_ACIDS, BRCT_START, BRCT_WT_SEQ  # noqa: E402

# ---------------------------------------------------------------------------
# Constants — canonical coordinate system is full-protein BRCA1 numbering
# ---------------------------------------------------------------------------
WT_SEQUENCE = BRCT_WT_SEQ
SGE_CACHE_PATH = os.path.join(REPO_ROOT, "cache", "sge_scores.csv")

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


def load_sge() -> dict:
    """Real SGE scores keyed by (position, mut_aa); empty if the cache is missing."""
    if not os.path.exists(SGE_CACHE_PATH):
        print(f"No SGE cache at {SGE_CACHE_PATH}; SGE columns will be empty")
        return {}
    # Several SNVs can give the same AA change; average them as pipeline/sge_data.py does
    scores = {}
    with open(SGE_CACHE_PATH) as f:
        for r in csv.DictReader(f):
            pos = int(r["position"])
            assert WT_SEQUENCE[pos - BRCT_START] == r["wt_aa"], f"SGE WT mismatch at {pos}"
            scores.setdefault((pos, r["mut_aa"]), []).append(float(r["sge_score"]))
    sge = {}
    for key, vals in scores.items():
        mean = sum(vals) / len(vals)
        sge[key] = (round(mean, 4), classify_sge(mean))
    return sge


def auroc(rows) -> float:
    """AUROC for -esm_score predicting LOF vs FUNC (Mann-Whitney U)."""
    lof = [-r["esm_score"] for r in rows if r["sge_class"] == "LOF"]
    func = [-r["esm_score"] for r in rows if r["sge_class"] == "FUNC"]
    if not lof or not func:
        return float("nan")
    wins = sum((a > b) + 0.5 * (a == b) for a in lof for b in func)
    return wins / (len(lof) * len(func))


def generate_variant_scores(out_path: str) -> dict:
    """Write stub variant_scores.csv and return summary stats."""
    random.seed(0)
    sge = load_sge()
    rows = []

    for i, wt_aa in enumerate(WT_SEQUENCE):
        position = BRCT_START + i
        for mut_aa in AMINO_ACIDS:
            if mut_aa == wt_aa:
                continue  # skip synonymous

            # Stub ESM score: normally distributed around -1
            esm_score = round(random.gauss(-1.0, 2.0), 4)

            sge_score, sge_class = sge.get((position, mut_aa), ("", ""))

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
        "auroc": round(auroc(rows), 4),
    }


def generate_metrics(stats: dict, out_path: str) -> None:
    """Write stub metrics.json."""
    metrics = {
        "auroc": stats["auroc"],
        "n_variants": stats["n_variants"],
        "n_with_sge": stats["n_with_sge"],
        "model": "STUB (random ESM scores)",
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Wrote {out_path}: {metrics}")


def main():
    csv_path = os.path.join(REPO_ROOT, "data", "variant_scores.csv")
    metrics_path = os.path.join(REPO_ROOT, "data", "metrics.json")

    stats = generate_variant_scores(csv_path)
    print(f"Wrote {csv_path}: {stats['n_variants']} variants, {stats['n_with_sge']} with SGE")

    generate_metrics(stats, metrics_path)


if __name__ == "__main__":
    main()

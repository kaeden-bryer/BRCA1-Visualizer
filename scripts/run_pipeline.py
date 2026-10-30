#!/usr/bin/env python3
"""Run the full BRCA1 BRCT variant scoring pipeline end-to-end.

Usage:
    python scripts/run_pipeline.py [--force]

Stages:
    1. ESM-2 masked-marginal scoring (~5-15 min on CPU)
    2. SGE experimental data fetch (MaveDB API)
    3. Join + evaluate (AUROC, write final CSVs)

The --force flag re-runs all stages even if cached results exist.
"""

import os
import sys
import time

# Ensure the repo root is on sys.path so `pipeline` is importable
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)


def main():
    force = "--force" in sys.argv
    t_start = time.time()

    print("=" * 60)
    print("BRCA1 BRCT Variant Scoring Pipeline")
    print("=" * 60)
    print(f"Repo root: {REPO_ROOT}")
    print(f"Force recompute: {force}")
    print()

    # --- Stage 1: Config sanity check ---
    print("[Stage 0] Config check...")
    from pipeline.config import BRCT_START, BRCT_END, BRCT_LENGTH, BRCT_WT_SEQ
    print(f"  Domain: {BRCT_START}-{BRCT_END} ({BRCT_LENGTH} residues)")
    assert len(BRCT_WT_SEQ) == BRCT_LENGTH
    print(f"  WT seq OK: {BRCT_WT_SEQ[:20]}...")
    print()

    # --- Stage 1: ESM-2 scoring ---
    print("[Stage 1] ESM-2 masked-marginal scoring...")
    t1 = time.time()
    from pipeline.esm_scoring import run_esm_scoring
    esm_df = run_esm_scoring(force=force)
    print(f"  Completed in {time.time() - t1:.1f}s")
    print(f"  {len(esm_df)} variant scores")
    print()

    # --- Stage 2: SGE data ---
    print("[Stage 2] SGE experimental data...")
    t2 = time.time()
    from pipeline.sge_data import fetch_sge_data
    sge_df = fetch_sge_data(force=force)
    print(f"  Completed in {time.time() - t2:.1f}s")
    print(f"  {len(sge_df)} SGE variants")
    print()

    # --- Stage 3: Join & evaluate ---
    print("[Stage 3] Join & evaluate...")
    t3 = time.time()
    from pipeline.join_and_evaluate import run_join_and_evaluate
    merged, metrics = run_join_and_evaluate()
    print(f"  Completed in {time.time() - t3:.1f}s")
    print()

    # --- Summary ---
    total = time.time() - t_start
    print("=" * 60)
    print(f"Pipeline complete in {total:.1f}s")
    print("=" * 60)
    print(f"  Variants scored:   {metrics['n_variants']}")
    print(f"  With SGE data:     {metrics['n_with_sge']}")
    print(f"  AUROC:             {metrics['auroc']}")
    print(f"  Model:             {metrics['model']}")
    print()
    print(f"Outputs:")
    from pipeline.config import data_path, VARIANT_SCORES_FILE, METRICS_FILE
    print(f"  {data_path(VARIANT_SCORES_FILE)}")
    print(f"  {data_path(METRICS_FILE)}")


if __name__ == "__main__":
    main()

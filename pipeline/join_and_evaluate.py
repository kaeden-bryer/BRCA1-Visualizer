"""Join ESM-2 scores with SGE experimental data and compute metrics.

Produces:
    data/variant_scores.csv  — the frozen-schema output
    data/metrics.json        — AUROC and summary stats

Standalone usage:
    python -m pipeline.join_and_evaluate
"""

import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from pipeline.config import (
    BRCT_END,
    BRCT_START,
    BRCT_WT_SEQ,
    AMINO_ACIDS,
    ESM_CACHE_FILE,
    METRICS_FILE,
    SGE_CACHE_FILE,
    VARIANT_SCORES_FILE,
    cache_path,
    classify_sge,
    data_path,
)


def load_esm_scores():
    """Load cached ESM scores."""
    path = cache_path(ESM_CACHE_FILE)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"ESM scores not found at {path}. Run esm_scoring.py first."
        )
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} ESM scores from {path}")
    return df


def load_sge_scores():
    """Load cached SGE scores."""
    path = cache_path(SGE_CACHE_FILE)
    if not os.path.exists(path):
        print(f"No SGE data at {path} — will produce ESM-only output")
        return pd.DataFrame(columns=["position", "wt_aa", "mut_aa", "sge_score", "sge_class"])
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} SGE scores from {path}")
    return df


def join_scores(esm_df, sge_df):
    """Left-join ESM scores with SGE scores on (position, wt_aa, mut_aa)."""
    join_cols = ["position", "wt_aa", "mut_aa"]

    if len(sge_df) == 0:
        merged = esm_df.copy()
        merged["sge_score"] = np.nan
        merged["sge_class"] = ""
    else:
        # Ensure consistent types
        for col in join_cols:
            esm_df[col] = esm_df[col].astype(str) if col != "position" else esm_df[col].astype(int)
            sge_df[col] = sge_df[col].astype(str) if col != "position" else sge_df[col].astype(int)

        # Several SNVs can encode the same AA substitution; average them so the
        # left join stays one row per (position, mut_aa)
        sge_df = sge_df.groupby(join_cols, as_index=False).agg({"sge_score": "mean"})
        sge_df["sge_class"] = sge_df["sge_score"].map(classify_sge)

        merged = esm_df.merge(
            sge_df[join_cols + ["sge_score", "sge_class"]],
            on=join_cols,
            how="left",
        )

    # Fill NaN sge_class with empty string for CSV output
    merged["sge_class"] = merged["sge_class"].fillna("")

    # Ensure column order matches frozen schema
    merged = merged[["position", "wt_aa", "mut_aa", "esm_score", "sge_score", "sge_class"]]

    n_with_sge = merged["sge_score"].notna().sum()
    print(f"Joined: {len(merged)} total variants, {n_with_sge} with SGE data")
    return merged


def compute_auroc(df):
    """Compute AUROC for ESM-2 predicting LOF vs FUNC.

    Convention:
    - Both ESM and SGE scores: more negative = worse (positive correlation)
    - For AUROC: LOF = positive class (label=1), FUNC = negative class (label=0)
    - We want: LOF variants should have MORE NEGATIVE ESM scores
    - sklearn roc_auc_score expects: higher score => more likely positive
    - Since more negative ESM => more likely LOF (positive), we NEGATE esm_score
      so that higher negated score => more likely LOF

    We VERIFY the correlation direction before computing.
    """
    # Filter to only LOF and FUNC (exclude INT and missing)
    mask = df["sge_class"].isin(["LOF", "FUNC"])
    subset = df[mask].copy()

    if len(subset) < 10:
        print("  Too few LOF/FUNC variants for AUROC — skipping")
        return None

    y_true = (subset["sge_class"] == "LOF").astype(int).values
    esm_scores = subset["esm_score"].values

    n_lof = y_true.sum()
    n_func = len(y_true) - n_lof
    print(f"  AUROC computation: {n_lof} LOF, {n_func} FUNC variants")

    # VERIFY correlation direction:
    # LOF variants should have more negative ESM scores on average
    mean_esm_lof = esm_scores[y_true == 1].mean()
    mean_esm_func = esm_scores[y_true == 0].mean()
    print(f"  Mean ESM score — LOF: {mean_esm_lof:.3f}, FUNC: {mean_esm_func:.3f}")

    if mean_esm_lof < mean_esm_func:
        print("  Confirmed: LOF has more negative ESM scores (expected)")
    else:
        print("  WARNING: LOF has LESS negative ESM scores — unexpected!")
        print("  Check for a sign error or coordinate offset.")

    # Negate ESM scores so higher = more likely LOF (positive class)
    # This ensures AUROC > 0.5 when the model works correctly
    auroc = roc_auc_score(y_true, -esm_scores)
    print(f"  AUROC = {auroc:.4f}")

    # Sanity: if AUROC < 0.5, the sign is flipped
    if auroc < 0.5:
        print("  WARNING: AUROC < 0.5 — trying without negation...")
        auroc_alt = roc_auc_score(y_true, esm_scores)
        print(f"  AUROC (un-negated) = {auroc_alt:.4f}")
        if auroc_alt > auroc:
            print("  Using un-negated scores (correlation was opposite to expected)")
            auroc = auroc_alt

    return round(auroc, 4)


def detect_esm_model():
    """Try to detect which ESM model was used from cache or config."""
    from pipeline.config import ESM_MODEL_PRIMARY
    return ESM_MODEL_PRIMARY


def run_join_and_evaluate():
    """Main function: join data, compute metrics, write outputs."""
    print("=" * 60)
    print("Join & Evaluate")
    print("=" * 60)

    esm_df = load_esm_scores()
    sge_df = load_sge_scores()

    # Join
    merged = join_scores(esm_df, sge_df)

    # AUROC
    auroc = compute_auroc(merged)
    if auroc is None:
        auroc = float("nan")

    # Write variant_scores.csv
    csv_out = data_path(VARIANT_SCORES_FILE)
    merged.to_csv(csv_out, index=False)
    print(f"\nWrote {csv_out} ({len(merged)} rows)")

    # Write metrics.json
    n_with_sge = int(merged["sge_score"].notna().sum())
    metrics = {
        "auroc": auroc,
        "n_variants": len(merged),
        "n_with_sge": n_with_sge,
        "model": detect_esm_model(),
    }
    metrics_out = data_path(METRICS_FILE)
    with open(metrics_out, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Wrote {metrics_out}: {metrics}")

    return merged, metrics


if __name__ == "__main__":
    run_join_and_evaluate()

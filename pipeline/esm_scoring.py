"""ESM-2 masked-marginal scoring for BRCA1 BRCT domain variants.

For each position in the BRCT domain (1646-1859), mask it and do a forward
pass through ESM-2. Score each substitution as:
    score = log P(mut_aa | context) - log P(wt_aa | context)

More negative = predicted more damaging.

Standalone usage:
    python -m pipeline.esm_scoring
"""

import os
import sys
import time

import numpy as np
import pandas as pd
import torch

from pipeline.config import (
    AMINO_ACIDS,
    BRCT_CONTEXT_SEQ,
    BRCT_END,
    BRCT_LENGTH,
    BRCT_START,
    BRCT_WT_SEQ,
    ESM_CACHE_FILE,
    ESM_MODEL_FALLBACK,
    ESM_MODEL_PRIMARY,
    cache_path,
)


def load_esm_model():
    """Load ESM-2 model and alphabet. Try 650M first, fall back to 150M."""
    import esm

    for model_name in [ESM_MODEL_PRIMARY, ESM_MODEL_FALLBACK]:
        try:
            print(f"Loading {model_name}...")
            # Try the named function on esm.pretrained (e.g. esm.pretrained.esm2_t33_650M_UR50D)
            load_fn = getattr(esm.pretrained, model_name, None)
            if load_fn is not None:
                model, alphabet = load_fn()
            else:
                # Fallback: generic loader
                model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
            model.eval()
            n_params = sum(p.numel() for p in model.parameters()) / 1e6
            print(f"  Loaded successfully ({n_params:.0f}M params)")
            return model, alphabet, model_name
        except Exception as e:
            print(f"  Failed to load {model_name}: {e}")
            continue

    raise RuntimeError("Could not load any ESM-2 model. Install with: pip install fair-esm")


def masked_marginal_scores(model, alphabet, sequence, brct_start, brct_length):
    """Compute masked-marginal log-likelihood ratios for all positions.

    Args:
        model: ESM-2 model
        alphabet: ESM-2 alphabet
        sequence: full context sequence (may be longer than scored region)
        brct_start: first residue number (full-protein numbering)
        brct_length: number of positions to score

    Returns:
        List of dicts with keys: position, wt_aa, mut_aa, esm_score
    """
    batch_converter = alphabet.get_batch_converter()

    # Tokenize the WT sequence once to get reference
    _, _, wt_tokens = batch_converter([("brct", sequence)])

    results = []
    t0 = time.time()

    for i in range(brct_length):
        pos = brct_start + i
        wt_aa = sequence[i]

        # Sanity check: WT amino acid matches config (Gotcha 1)
        assert wt_aa == BRCT_WT_SEQ[i], (
            f"WT mismatch at position {pos}: sequence has '{wt_aa}' "
            f"but config expects '{BRCT_WT_SEQ[i]}'"
        )

        # Create masked sequence: replace position i with <mask>
        masked_tokens = wt_tokens.clone()
        # Token index is i+1 because of the <cls> token at position 0
        token_idx = i + 1
        masked_tokens[0, token_idx] = alphabet.mask_idx

        with torch.no_grad():
            logits = model(masked_tokens)["logits"]

        # Extract log-probabilities at the masked position
        log_probs = torch.log_softmax(logits[0, token_idx], dim=-1)

        # Get WT log-prob
        wt_token_id = alphabet.get_idx(wt_aa)
        wt_log_prob = log_probs[wt_token_id].item()

        # Score each possible substitution
        for mut_aa in AMINO_ACIDS:
            if mut_aa == wt_aa:
                continue
            mut_token_id = alphabet.get_idx(mut_aa)
            mut_log_prob = log_probs[mut_token_id].item()
            esm_score = mut_log_prob - wt_log_prob

            results.append({
                "position": pos,
                "wt_aa": wt_aa,
                "mut_aa": mut_aa,
                "esm_score": round(esm_score, 6),
            })

        # Progress
        elapsed = time.time() - t0
        rate = (i + 1) / elapsed if elapsed > 0 else 0
        eta = (brct_length - i - 1) / rate if rate > 0 else 0
        if (i + 1) % 10 == 0 or i == 0:
            print(
                f"  Position {pos} ({i+1}/{brct_length}) | "
                f"{elapsed:.0f}s elapsed | ETA {eta:.0f}s"
            )

    total = time.time() - t0
    print(f"  Done: {brct_length} positions, {len(results)} variants in {total:.1f}s")
    return results


def run_esm_scoring(force=False):
    """Run ESM-2 scoring. Returns a DataFrame. Uses cache if available."""
    out = cache_path(ESM_CACHE_FILE)

    if os.path.exists(out) and not force:
        print(f"Loading cached ESM scores from {out}")
        df = pd.read_csv(out)
        print(f"  {len(df)} variants loaded")
        return df

    print("=" * 60)
    print("ESM-2 Masked-Marginal Scoring")
    print("=" * 60)

    model, alphabet, model_name = load_esm_model()

    # Use context sequence for better boundary predictions, but only score
    # the first BRCT_LENGTH positions (the actual BRCT domain 1646-1859)
    print(f"Scoring {BRCT_LENGTH} positions ({BRCT_START}-{BRCT_END})...")
    print(f"  Using context sequence of {len(BRCT_CONTEXT_SEQ)} residues")

    results = masked_marginal_scores(
        model, alphabet, BRCT_CONTEXT_SEQ, BRCT_START, BRCT_LENGTH
    )

    df = pd.DataFrame(results)
    df.to_csv(out, index=False)
    print(f"Saved ESM scores to {out}")

    # Quick sanity check: most scores should be negative (mutations hurt)
    median_score = df["esm_score"].median()
    print(f"  Median ESM score: {median_score:.3f} (should be negative)")
    if median_score > 0:
        print("  WARNING: Median is positive — check scoring logic!")

    return df


if __name__ == "__main__":
    force = "--force" in sys.argv
    df = run_esm_scoring(force=force)
    print(f"\nSummary:")
    print(df.describe())

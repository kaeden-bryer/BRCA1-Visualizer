"""Fetch and parse SGE experimental data for BRCA1 BRCT domain.

Attempts MaveDB API first, falls back to cached/bundled data.
Outputs amino-acid-level scores (NOT nucleotide-level — see Gotcha 2).

Standalone usage:
    python -m pipeline.sge_data
"""

import os
import sys
import time

import pandas as pd
import requests

from pipeline.config import (
    BRCT_END,
    BRCT_START,
    BRCT_WT_SEQ,
    SGE_CACHE_FILE,
    cache_path,
    classify_sge,
)

# ---------------------------------------------------------------------------
# MaveDB API configuration
# ---------------------------------------------------------------------------
MAVEDB_API_BASE = "https://api.mavedb.org/api/v1"

# Known MaveDB score set URNs for Findlay et al. BRCA1 SGE data.
# These are the amino-acid-level score sets (preferred over nucleotide-level).
# The BRCA1 SGE data may be under multiple URNs for different exon groups.
# We search for them dynamically, but keep known URNs as fallback.
KNOWN_SCORESETS = [
    # Findlay 2018 BRCA1 SGE - various exon groups
    # These will be searched dynamically via the API
]


def search_mavedb_scoresets():
    """Search MaveDB for BRCA1 SGE amino-acid-level score sets."""
    print("Searching MaveDB for BRCA1 SGE score sets...")

    try:
        # Search for BRCA1 experiments
        url = f"{MAVEDB_API_BASE}/experiments/search"
        resp = requests.post(
            url,
            json={"text": "BRCA1 saturation genome editing"},
            timeout=30,
        )
        resp.raise_for_status()
        experiments = resp.json()

        scoreset_urns = []
        for exp in experiments:
            urn = exp.get("urn", "")
            title = exp.get("title", "")
            print(f"  Found experiment: {urn} — {title[:80]}")

            # Get score sets for this experiment
            for ss in exp.get("scoreSetUrns", exp.get("score_set_urns", [])):
                scoreset_urns.append(ss)

            # Also try the scoresets field
            for ss in exp.get("scoreSets", exp.get("score_sets", [])):
                if isinstance(ss, dict):
                    scoreset_urns.append(ss.get("urn", ""))
                elif isinstance(ss, str):
                    scoreset_urns.append(ss)

        print(f"  Found {len(scoreset_urns)} score set URN(s)")
        return scoreset_urns

    except Exception as e:
        print(f"  MaveDB search failed: {e}")
        return []


def fetch_mavedb_scoreset(urn):
    """Fetch scores from a MaveDB score set by URN."""
    print(f"  Fetching scores for {urn}...")
    try:
        url = f"{MAVEDB_API_BASE}/score-sets/{urn}/scores"
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        if isinstance(data, list) and len(data) > 0:
            print(f"    Got {len(data)} score records")
            return data
        else:
            print(f"    Empty or unexpected response format")
            return None

    except Exception as e:
        print(f"    Failed: {e}")
        return None


def parse_mavedb_scores(records):
    """Parse MaveDB score records into a clean DataFrame.

    MaveDB records typically have:
    - hgvs_pro: protein-level HGVS notation (e.g., "p.Lys1646Arg")
    - hgvs_nt: nucleotide-level HGVS notation
    - score: the function score

    We prefer protein-level (amino acid) records (Gotcha 2).
    """
    rows = []

    # Three-letter to one-letter amino acid mapping
    aa3to1 = {
        "Ala": "A", "Cys": "C", "Asp": "D", "Glu": "E", "Phe": "F",
        "Gly": "G", "His": "H", "Ile": "I", "Lys": "K", "Leu": "L",
        "Met": "M", "Asn": "N", "Pro": "P", "Gln": "Q", "Arg": "R",
        "Ser": "S", "Thr": "T", "Val": "V", "Trp": "W", "Tyr": "Y",
        "Ter": "*",
    }

    import re
    # Pattern: p.Lys1646Arg or p.K1646R
    pattern_3letter = re.compile(r"p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})")
    pattern_1letter = re.compile(r"p\.([A-Z])(\d+)([A-Z])")

    for rec in records:
        hgvs_pro = rec.get("hgvs_pro", "") or ""
        score = rec.get("score")

        if score is None or hgvs_pro == "":
            continue

        # Try 3-letter notation first
        m = pattern_3letter.search(hgvs_pro)
        if m:
            wt_3 = m.group(1)
            pos = int(m.group(2))
            mut_3 = m.group(3)
            wt_aa = aa3to1.get(wt_3)
            mut_aa = aa3to1.get(mut_3)
        else:
            # Try 1-letter notation
            m = pattern_1letter.search(hgvs_pro)
            if m:
                wt_aa = m.group(1)
                pos = int(m.group(2))
                mut_aa = m.group(3)
            else:
                continue

        # Skip nonsense/synonymous
        if wt_aa is None or mut_aa is None:
            continue
        if mut_aa == "*" or wt_aa == "*":
            continue
        if wt_aa == mut_aa:
            continue

        # Only keep BRCT domain positions
        if pos < BRCT_START or pos > BRCT_END:
            continue

        try:
            score_f = float(score)
        except (ValueError, TypeError):
            continue

        rows.append({
            "position": pos,
            "wt_aa": wt_aa,
            "mut_aa": mut_aa,
            "sge_score": round(score_f, 6),
        })

    df = pd.DataFrame(rows)
    if len(df) > 0:
        # De-duplicate: if multiple nucleotide variants map to the same
        # amino acid substitution, take the mean score
        df = df.groupby(["position", "wt_aa", "mut_aa"], as_index=False).agg(
            {"sge_score": "mean"}
        )
        df["sge_score"] = df["sge_score"].round(6)

    return df


def validate_sge_against_wt(df):
    """Validate that WT amino acids in SGE data match our config sequence."""
    mismatches = 0
    for pos in df["position"].unique():
        idx = pos - BRCT_START
        if idx < 0 or idx >= len(BRCT_WT_SEQ):
            continue
        expected_wt = BRCT_WT_SEQ[idx]
        observed_wts = df.loc[df["position"] == pos, "wt_aa"].unique()
        for obs in observed_wts:
            if obs != expected_wt:
                mismatches += 1
                if mismatches <= 5:
                    print(
                        f"  WARNING: WT mismatch at {pos}: "
                        f"SGE has '{obs}', config has '{expected_wt}'"
                    )
    if mismatches > 0:
        print(f"  Total WT mismatches: {mismatches}")
        print("  This may indicate a coordinate offset — check numbering!")
    else:
        print("  WT validation passed: all SGE positions match config sequence")
    return mismatches


def fetch_sge_data(force=False):
    """Fetch SGE data for BRCT domain. Returns DataFrame with sge_score and sge_class."""
    out = cache_path(SGE_CACHE_FILE)

    if os.path.exists(out) and not force:
        print(f"Loading cached SGE data from {out}")
        df = pd.read_csv(out)
        print(f"  {len(df)} variants loaded")
        return df

    print("=" * 60)
    print("Fetching SGE Experimental Data")
    print("=" * 60)

    df = pd.DataFrame()

    # Strategy 1: MaveDB API
    urns = search_mavedb_scoresets()
    for urn in urns:
        records = fetch_mavedb_scoreset(urn)
        if records:
            chunk = parse_mavedb_scores(records)
            if len(chunk) > 0:
                print(f"    Parsed {len(chunk)} BRCT variants from {urn}")
                df = pd.concat([df, chunk], ignore_index=True)

    # De-duplicate across score sets
    if len(df) > 0:
        df = df.groupby(["position", "wt_aa", "mut_aa"], as_index=False).agg(
            {"sge_score": "mean"}
        )
        df["sge_score"] = df["sge_score"].round(6)

    if len(df) == 0:
        print("\n  MaveDB did not yield BRCT-domain variants.")
        print("  To add SGE data manually:")
        print("    1. Download Findlay et al. 2018 supplementary Table S3")
        print("       DOI: 10.1038/s41586-018-0461-z")
        print("    2. Place the amino-acid-level scores CSV at:")
        print(f"       {out}")
        print("    3. Required columns: position, wt_aa, mut_aa, sge_score")
        print("\n  Proceeding without SGE data (ESM-only output).")
        df = pd.DataFrame(columns=["position", "wt_aa", "mut_aa", "sge_score", "sge_class"])
        df.to_csv(out, index=False)
        return df

    # Validate WT amino acids match our sequence
    validate_sge_against_wt(df)

    # Classify
    df["sge_class"] = df["sge_score"].apply(classify_sge)

    # Summary
    print(f"\nSGE data summary:")
    print(f"  Total BRCT variants: {len(df)}")
    print(f"  Position range: {df['position'].min()}-{df['position'].max()}")
    print(f"  Class distribution:")
    if "sge_class" in df.columns:
        for cls, count in df["sge_class"].value_counts().items():
            print(f"    {cls}: {count}")

    df.to_csv(out, index=False)
    print(f"Saved SGE data to {out}")
    return df


if __name__ == "__main__":
    force = "--force" in sys.argv
    df = fetch_sge_data(force=force)
    if len(df) > 0:
        print(f"\nSample:")
        print(df.head(10).to_string(index=False))

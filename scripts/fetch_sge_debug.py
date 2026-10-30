"""Debug script to fetch SGE data from MaveDB with verbose output.

Run in Colab or anywhere with `requests` and `pandas` installed:
    python scripts/fetch_sge_debug.py

Tries multiple approaches to find the Findlay BRCA1 SGE data.
"""

import json
import os
import re
import sys

import pandas as pd
import requests

MAVEDB_API = "https://api.mavedb.org/api/v1"

BRCT_START = 1646
BRCT_END = 1859

AA3TO1 = {
    "Ala": "A", "Cys": "C", "Asp": "D", "Glu": "E", "Phe": "F",
    "Gly": "G", "His": "H", "Ile": "I", "Lys": "K", "Leu": "L",
    "Met": "M", "Asn": "N", "Pro": "P", "Gln": "Q", "Arg": "R",
    "Ser": "S", "Thr": "T", "Val": "V", "Trp": "W", "Tyr": "Y",
    "Ter": "*",
}

PATTERN_3 = re.compile(r"p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})")
PATTERN_1 = re.compile(r"p\.([A-Z])(\d+)([A-Z])")


def search_experiments(query):
    """Search MaveDB experiments and return full JSON."""
    print(f"\n--- Searching MaveDB: '{query}' ---")
    resp = requests.post(f"{MAVEDB_API}/experiments/search", json={"text": query}, timeout=30)
    resp.raise_for_status()
    exps = resp.json()
    print(f"  Got {len(exps)} experiment(s)")
    for exp in exps:
        print(f"  URN: {exp.get('urn')}")
        print(f"  Title: {exp.get('title', '')[:100]}")
        print(f"  Keys: {list(exp.keys())}")
        # Look for score set references in any field
        for key in exp:
            val = exp[key]
            if isinstance(val, list) and len(val) > 0:
                print(f"    {key}: {val[:3]}{'...' if len(val)>3 else ''}")
        print()
    return exps


def get_experiment_scoresets(exp_urn):
    """Fetch score sets for a specific experiment URN."""
    print(f"--- Fetching score sets for {exp_urn} ---")
    try:
        resp = requests.get(f"{MAVEDB_API}/experiments/{exp_urn}/score-sets", timeout=30)
        resp.raise_for_status()
        scoresets = resp.json()
        print(f"  Got {len(scoresets)} score set(s)")
        for ss in scoresets:
            if isinstance(ss, dict):
                print(f"  URN: {ss.get('urn')} | Title: {ss.get('title', '')[:80]}")
            else:
                print(f"  {ss}")
        return scoresets
    except Exception as e:
        print(f"  Failed: {e}")
        # Try alternate endpoint
        try:
            resp = requests.get(f"{MAVEDB_API}/score-sets/?experiment={exp_urn}", timeout=30)
            resp.raise_for_status()
            scoresets = resp.json()
            print(f"  (alt endpoint) Got {len(scoresets)} score set(s)")
            return scoresets
        except Exception as e2:
            print(f"  Alt also failed: {e2}")
            return []


def fetch_scores(ss_urn):
    """Fetch all scores from a score set."""
    print(f"  Fetching scores for {ss_urn}...")
    resp = requests.get(f"{MAVEDB_API}/score-sets/{ss_urn}/scores", timeout=60)
    resp.raise_for_status()
    data = resp.json()
    print(f"  Got {len(data)} record(s)")
    if len(data) > 0:
        print(f"  Sample keys: {list(data[0].keys())}")
        print(f"  Sample record: {json.dumps(data[0], indent=2)[:300]}")
    return data


def parse_record(rec):
    """Parse one MaveDB score record. Returns dict or None."""
    hgvs_pro = rec.get("hgvs_pro") or rec.get("hgvs_protein") or ""
    score = rec.get("score")
    if score is None or not hgvs_pro:
        return None

    m = PATTERN_3.search(hgvs_pro)
    if m:
        wt_aa = AA3TO1.get(m.group(1))
        pos = int(m.group(2))
        mut_aa = AA3TO1.get(m.group(3))
    else:
        m = PATTERN_1.search(hgvs_pro)
        if m:
            wt_aa, pos, mut_aa = m.group(1), int(m.group(2)), m.group(3)
        else:
            return None

    if not wt_aa or not mut_aa or wt_aa == "*" or mut_aa == "*" or wt_aa == mut_aa:
        return None

    try:
        return {"position": pos, "wt_aa": wt_aa, "mut_aa": mut_aa, "sge_score": float(score)}
    except (ValueError, TypeError):
        return None


def main():
    all_rows = []

    # Try multiple search queries
    queries = [
        "BRCA1 saturation genome editing",
        "BRCA1 Findlay",
        "BRCA1",
        "BRCA1 SGE",
    ]

    seen_urns = set()
    all_scoresets = []

    for q in queries:
        try:
            exps = search_experiments(q)
            for exp in exps:
                exp_urn = exp.get("urn", "")
                if exp_urn in seen_urns:
                    continue
                seen_urns.add(exp_urn)

                # Try to get score sets from the experiment
                scoresets = get_experiment_scoresets(exp_urn)
                for ss in scoresets:
                    if isinstance(ss, dict):
                        all_scoresets.append(ss.get("urn", ""))
                    elif isinstance(ss, str):
                        all_scoresets.append(ss)

                # Also check inline score set URNs
                for key in ["scoreSetUrns", "score_set_urns"]:
                    for urn in exp.get(key, []):
                        all_scoresets.append(urn)
        except Exception as e:
            print(f"  Search '{q}' failed: {e}")

    # De-dup score set URNs
    all_scoresets = list(dict.fromkeys(u for u in all_scoresets if u))
    print(f"\n=== Found {len(all_scoresets)} unique score set URN(s) ===")
    for urn in all_scoresets:
        print(f"  {urn}")

    # Fetch each score set
    for ss_urn in all_scoresets:
        try:
            records = fetch_scores(ss_urn)
            if not records:
                continue

            # Parse and check position range
            parsed = [parse_record(r) for r in records]
            parsed = [r for r in parsed if r is not None]

            if parsed:
                positions = [r["position"] for r in parsed]
                print(f"  Position range: {min(positions)}-{max(positions)}")
                brct_rows = [r for r in parsed if BRCT_START <= r["position"] <= BRCT_END]
                print(f"  In BRCT domain ({BRCT_START}-{BRCT_END}): {len(brct_rows)}")
                all_rows.extend(brct_rows)
            else:
                # Show raw hgvs to debug parsing
                sample_hgvs = [r.get("hgvs_pro", "") for r in records[:5]]
                print(f"  No records parsed. Sample hgvs_pro: {sample_hgvs}")
        except Exception as e:
            print(f"  Error fetching {ss_urn}: {e}")

    print(f"\n=== Total BRCT rows collected: {len(all_rows)} ===")

    if all_rows:
        df = pd.DataFrame(all_rows)
        df = df.groupby(["position", "wt_aa", "mut_aa"], as_index=False).agg({"sge_score": "mean"})
        df["sge_score"] = df["sge_score"].round(6)
        print(f"After dedup: {len(df)} unique variants")
        print(df.head(10).to_string(index=False))

        out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache", "sge_scores.csv")
        os.makedirs(os.path.dirname(out), exist_ok=True)

        # Add sge_class
        def classify(s):
            if s > -0.748: return "FUNC"
            elif s >= -1.328: return "INT"
            else: return "LOF"

        df["sge_class"] = df["sge_score"].apply(classify)
        df.to_csv(out, index=False)
        print(f"\nSaved to {out}")
        print(f"Class distribution:")
        print(df["sge_class"].value_counts().to_string())
    else:
        print("\nNo BRCT data found from MaveDB.")
        print("The SGE data likely uses nucleotide-level HGVS only,")
        print("or the positions are in a different numbering system.")
        print("\nFallback: download Findlay et al. 2018 Supplementary Table S3")
        print("  DOI: 10.1038/s41586-018-0461-z")


if __name__ == "__main__":
    main()

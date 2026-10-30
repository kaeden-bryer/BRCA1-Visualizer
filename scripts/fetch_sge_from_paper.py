"""Fetch SGE scores from Findlay et al. 2018 supplementary data.

The MaveDB deposit uses nucleotide-level HGVS only (no hgvs_pro),
so we use the paper's Supplementary Table 3 which has AA-level scores.

Source: Findlay et al., Nature 2018, DOI 10.1038/s41586-018-0461-z

Usage:
    python3 scripts/fetch_sge_from_paper.py
"""

import io
import os

import pandas as pd
import requests

BRCT_START = 1646
BRCT_END = 1859

SUPP_TABLE_URL = (
    "https://static-content.springer.com/esm/"
    "art%3A10.1038%2Fs41586-018-0461-z/MediaObjects/"
    "41586_2018_461_MOESM3_ESM.xlsx"
)


def classify(score):
    if score > -0.748:
        return "FUNC"
    elif score >= -1.328:
        return "INT"
    else:
        return "LOF"


def main():
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "sge_scores.csv")

    # Download supplementary table
    print("Downloading Findlay et al. 2018 Supplementary Table 3...")
    resp = requests.get(SUPP_TABLE_URL, timeout=60, headers={
        "User-Agent": "Mozilla/5.0 (research-tool; BRCA1-Visualizer)"
    })
    resp.raise_for_status()
    print(f"  Downloaded {len(resp.content)} bytes")

    # Row 2 is the real header (rows 0-1 are category groupings)
    df = pd.read_excel(io.BytesIO(resp.content), header=2)
    print(f"  {len(df)} rows, columns: {list(df.columns)}")

    # Filter to missense variants with valid AA info and scores
    miss = df[df["consequence"] == "Missense"].copy()
    print(f"  Missense variants: {len(miss)}")

    miss = miss.dropna(subset=["aa_pos", "aa_ref", "aa_alt", "function.score.mean"])
    miss["aa_pos"] = miss["aa_pos"].astype(int)
    print(f"  With complete AA + score data: {len(miss)}")

    # Filter to BRCT domain
    brct = miss[(miss["aa_pos"] >= BRCT_START) & (miss["aa_pos"] <= BRCT_END)].copy()
    print(f"  In BRCT domain ({BRCT_START}-{BRCT_END}): {len(brct)}")

    # Build output dataframe
    out = pd.DataFrame({
        "position": brct["aa_pos"].values,
        "wt_aa": brct["aa_ref"].values,
        "mut_aa": brct["aa_alt"].values,
        "sge_score": brct["function.score.mean"].round(6).values,
    })

    # Use the paper's own classification if available, otherwise compute
    if "func.class" in brct.columns:
        out["sge_class"] = brct["func.class"].values
    else:
        out["sge_class"] = out["sge_score"].apply(classify)

    out = out.sort_values(["position", "mut_aa"]).reset_index(drop=True)

    print(f"\n=== Results ===")
    print(f"Total BRCT missense variants: {len(out)}")
    print(f"Position range: {out['position'].min()}-{out['position'].max()}")
    print(f"Class distribution:\n{out['sge_class'].value_counts().to_string()}")
    print(f"\nSample:\n{out.head(10).to_string(index=False)}")

    out.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()

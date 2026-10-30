"""
BRCT Variant Explorer -- Deep Mutational Scan
Streamlit app for interactive visualization of BRCA1 BRCT domain variant effects.
"""

import json
import pathlib

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import py3Dmol
import streamlit as st
from stmol import showmol

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = pathlib.Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "data" / "variant_scores.csv"
METRICS_PATH = ROOT / "data" / "metrics.json"

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
PDB_ID = "1JNX"
# 1JNX: BRCA1 BRCT tandem-repeat, chain X, residues 1649-1859.
# Residue numbering matches full-protein UniProt P38398 numbering.
PDB_FILE = ROOT / "data" / f"{PDB_ID}.pdb"
PDB_CHAIN = "X"

# ---------------------------------------------------------------------------
# PDB residue-number lookup (three-letter -> one-letter)
# ---------------------------------------------------------------------------
THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M", "HSD": "H", "HSE": "H", "HSP": "H", "SEC": "U", "PYL": "O",
}

# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------

@st.cache_data
def load_variants() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    df["position"] = df["position"].astype(int)
    return df


@st.cache_data
def load_metrics() -> dict:
    with open(METRICS_PATH) as f:
        return json.load(f)


@st.cache_data
def load_pdb(path: pathlib.Path) -> str:
    return path.read_text()


@st.cache_data
def parse_pdb_residues(pdb_text: str, chain: str | None = None) -> dict[int, str]:
    """Map residue number -> one-letter amino acid from ATOM records."""
    residues: dict[int, str] = {}
    for line in pdb_text.splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        if len(line) < 27:
            continue
        alt_loc = line[16]
        if alt_loc not in (" ", "A"):
            continue
        if line[26] != " ":  # insertion code -> skip
            continue
        if chain is not None and line[21] != chain:
            continue
        res_name = line[17:20].strip().upper()
        one = THREE_TO_ONE.get(res_name)
        if one is None:
            continue
        try:
            res_seq = int(line[22:26])
        except ValueError:
            continue
        residues.setdefault(res_seq, one)
    return residues


def check_numbering(df: pd.DataFrame, pdb_residues: dict[int, str]):
    """Verify CSV wild-type residues against the structure."""
    wt_by_pos = (
        df.drop_duplicates(subset=["position"])
          .set_index("position")["wt_aa"]
          .to_dict()
    )
    matched, mismatches = 0, []
    for pos, csv_wt in sorted(wt_by_pos.items()):
        pdb_wt = pdb_residues.get(pos)
        if pdb_wt is None:
            continue
        if pdb_wt == csv_wt:
            matched += 1
        else:
            mismatches.append((pos, csv_wt, pdb_wt))
    return matched, len(wt_by_pos), mismatches


def guess_offset(df: pd.DataFrame, pdb_residues: dict[int, str],
                 search: int = 60) -> tuple[int, int]:
    """Brute-force the offset that maximises wild-type agreement."""
    wt_by_pos = (
        df.drop_duplicates(subset=["position"])
          .set_index("position")["wt_aa"]
          .to_dict()
    )
    best = (0, -1)
    for off in range(-search, search + 1):
        n = sum(
            1 for pos, wt in wt_by_pos.items()
            if pdb_residues.get(pos + off) == wt
        )
        if n > best[1]:
            best = (off, n)
    return best


# ---------------------------------------------------------------------------
# Helper: build matrix for heatmap
# ---------------------------------------------------------------------------

def build_matrix(df: pd.DataFrame, score_col: str):
    """Return a 2-D numpy array (aa x position) and axis labels."""
    positions = sorted(df["position"].unique())
    aa_order = AMINO_ACIDS

    matrix = np.full((len(aa_order), len(positions)), np.nan)
    wt_mask = np.zeros_like(matrix, dtype=bool)

    pos_idx = {p: i for i, p in enumerate(positions)}
    aa_idx = {a: i for i, a in enumerate(aa_order)}

    for _, row in df.iterrows():
        pi = pos_idx.get(row["position"])
        ai = aa_idx.get(row["mut_aa"])
        if pi is not None and ai is not None:
            matrix[ai, pi] = row[score_col]
        # Mark WT
        wi = aa_idx.get(row["wt_aa"])
        if wi is not None and pi is not None:
            wt_mask[wi, pi] = True

    return matrix, positions, aa_order, wt_mask


def avg_sensitivity(df: pd.DataFrame) -> dict:
    """Mean ESM score per position (across all mutations)."""
    return df.groupby("position")["esm_score"].mean().to_dict()

# ---------------------------------------------------------------------------
# 3-D viewer (Issue 2: zero-centred colour scale)
# ---------------------------------------------------------------------------

def render_3d(pdb_text: str, sensitivity: dict, selected_pos: int | None = None):
    view = py3Dmol.view(width=480, height=480)
    view.addModel(pdb_text, "pdb")

    # Base cartoon coloured grey
    view.setStyle({"cartoon": {"color": "grey"}})

    # Zero-centred colour scale: red (negative/damaging) - white (0) - blue (positive/benign)
    sens_vals = list(sensitivity.values())
    if sens_vals:
        bound = max(abs(min(sens_vals)), abs(max(sens_vals)))
    else:
        bound = 1.0
    bound = max(bound, 1e-6)

    for pos, score in sensitivity.items():
        t = max(-1.0, min(1.0, score / bound))
        if t < 0:  # red -> white
            f = 1.0 + t
            r, g, b = 255, int(255 * f), int(255 * f)
        else:  # white -> blue
            f = 1.0 - t
            r, g, b = int(255 * f), int(255 * f), 255
        hex_col = f"#{r:02x}{g:02x}{b:02x}"
        view.setStyle({"resi": str(pos)}, {"cartoon": {"color": hex_col}})

    # Highlight selected residue
    if selected_pos is not None:
        view.setStyle({"resi": str(selected_pos)},
                       {"cartoon": {"color": "yellow"},
                        "stick": {"color": "yellow"}})
        view.addLabel(str(selected_pos),
                      {"backgroundColor": "black", "fontColor": "white",
                       "position": {"resi": str(selected_pos)}},
                      {"resi": str(selected_pos)})

    view.zoomTo()
    return view, bound

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="BRCT Variant Explorer", layout="wide")
st.title("BRCT Variant Explorer -- Deep Mutational Scan")

df = load_variants()
metrics = load_metrics()
pdb_text = load_pdb(PDB_FILE)
sensitivity = avg_sensitivity(df)

# ---------------------------------------------------------------------------
# Issue 1: Verify residue numbering before rendering
# ---------------------------------------------------------------------------
pdb_residues = parse_pdb_residues(pdb_text, chain=PDB_CHAIN)
n_matched, n_positions, mismatches = check_numbering(df, pdb_residues)

if mismatches:
    best_offset, best_n = guess_offset(df, pdb_residues)
    st.error(
        f"Residue numbering mismatch between {PDB_ID} and variant_scores.csv: "
        f"{len(mismatches)} position(s) disagree. "
        f"Best-fitting offset is {best_offset:+d} "
        f"({best_n}/{n_positions} positions would match). "
        "Refusing to render a structure that may be coloured incorrectly."
    )
    st.write("First mismatches (position, CSV wild-type, PDB residue):")
    st.table(pd.DataFrame(mismatches[:10],
                          columns=["position", "csv_wt_aa", "pdb_residue"]))
    st.stop()

coverage_pct = 100.0 * n_matched / max(n_positions, 1)

# ---------------------------------------------------------------------------
# Session-state for selection
# ---------------------------------------------------------------------------
if "sel_pos" not in st.session_state:
    st.session_state.sel_pos = int(df["position"].min())
if "sel_aa" not in st.session_state:
    st.session_state.sel_aa = "A"

# ---------------------------------------------------------------------------
# Sidebar: selection controls + validation (Issue 3: single AUROC)
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Residue selector")
    positions = sorted(df["position"].unique())
    sel_pos = st.selectbox("Position", positions,
                           index=positions.index(st.session_state.sel_pos)
                           if st.session_state.sel_pos in positions else 0)
    st.session_state.sel_pos = sel_pos

    wt_at_pos = df.loc[df["position"] == sel_pos, "wt_aa"].iloc[0]
    st.write(f"Wild-type at {sel_pos}: **{wt_at_pos}**")

    possible_muts = sorted(df.loc[df["position"] == sel_pos, "mut_aa"].unique())
    sel_aa = st.selectbox("Mutant amino acid", possible_muts,
                          index=possible_muts.index(st.session_state.sel_aa)
                          if st.session_state.sel_aa in possible_muts else 0)
    st.session_state.sel_aa = sel_aa

    row = df[(df["position"] == sel_pos) & (df["mut_aa"] == sel_aa)]
    if not row.empty:
        r = row.iloc[0]
        st.metric("ESM score", f"{r['esm_score']:.3f}")
        if pd.notna(r["sge_score"]):
            st.metric("SGE score", f"{r['sge_score']:.3f}")
            st.write(f"SGE class: **{r['sge_class']}**")
        else:
            st.write("SGE score: _not available_")

    st.divider()
    st.header("Dataset info")
    st.write(f"**Model:** {metrics['model']}")
    st.write(f"**Variants:** {metrics['n_variants']}")
    st.write(f"**With SGE data:** {metrics['n_with_sge']}")

# ---------------------------------------------------------------------------
# Main layout: heatmap (left) + 3D viewer (right)
# ---------------------------------------------------------------------------
col_heat, col_3d = st.columns([3, 2])

# -- Heatmap --
with col_heat:
    score_toggle = st.radio("Color by", ["ESM score", "SGE score"], horizontal=True)
    score_col = "esm_score" if score_toggle == "ESM score" else "sge_score"

    matrix, pos_labels, aa_labels, wt_mask = build_matrix(df, score_col)

    # Build custom hover text
    hover = np.empty_like(matrix, dtype=object)
    for ai, aa in enumerate(aa_labels):
        for pi, pos in enumerate(pos_labels):
            val = matrix[ai, pi]
            wt = "WT" if wt_mask[ai, pi] else ""
            val_str = f"{val:.3f}" if not np.isnan(val) else "N/A"
            hover[ai, pi] = f"Pos {pos} | {aa} | {score_col}={val_str} {wt}"

    # Diverging colour scale: red = negative (damaging), blue = positive (benign)
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=[str(p) for p in pos_labels],
        y=aa_labels,
        text=hover,
        hoverinfo="text",
        colorscale="RdBu",
        zmid=0,
        colorbar=dict(title=score_col),
    ))

    # Mark WT cells with a symbol overlay (non-interactive so clicks reach heatmap)
    wt_y, wt_x = np.where(wt_mask)
    fig.add_trace(go.Scatter(
        x=[str(pos_labels[xi]) for xi in wt_x],
        y=[aa_labels[yi] for yi in wt_y],
        mode="markers",
        marker=dict(symbol="diamond-open", size=7, color="black", line=dict(width=1.5)),
        name="Wild-type",
        hoverinfo="skip",
    ))

    fig.update_layout(
        title=f"Heatmap: {score_col} (position x amino acid)",
        xaxis=dict(title="Position", dtick=10, tickangle=-45),
        yaxis=dict(title="Amino acid"),
        height=600,
        margin=dict(l=50, r=20, t=50, b=60),
    )

    # Issue 4: Click-to-select on heatmap
    event = st.plotly_chart(
        fig,
        width="stretch",
        on_select="rerun",
        selection_mode="points",
        key="heatmap",
    )

    if event and event.get("selection", {}).get("points"):
        pt = event["selection"]["points"][0]
        try:
            clicked_pos = int(pt["x"])
            clicked_aa = str(pt["y"])
        except (KeyError, ValueError, TypeError):
            clicked_pos = clicked_aa = None
        if clicked_pos is not None and clicked_aa is not None:
            if (clicked_pos != st.session_state.sel_pos
                    or clicked_aa != st.session_state.sel_aa):
                st.session_state.sel_pos = clicked_pos
                st.session_state.sel_aa = clicked_aa
                st.rerun()

# -- 3D viewer --
with col_3d:
    st.subheader(f"3D structure ({PDB_ID})")
    st.caption("Cartoon coloured by avg ESM sensitivity (red=damaging, white=neutral, blue=benign). "
               "Selected residue in yellow.")
    view, bound = render_3d(pdb_text, sensitivity, selected_pos=st.session_state.sel_pos)
    showmol(view, height=480, width=480)
    st.caption(
        f"Structure coverage: {n_matched}/{n_positions} scanned positions "
        f"present in {PDB_ID} ({coverage_pct:.0f}%). "
        f"Colour scale centred at 0, range \u00b1{bound:.2f}."
    )

# ---------------------------------------------------------------------------
# Validation panel (below)
# ---------------------------------------------------------------------------
st.divider()
st.header("ESM vs SGE validation")

val_col1, val_col2 = st.columns(2)

with val_col1:
    sge_df = df.dropna(subset=["sge_score", "sge_class"]).copy()
    color_map = {"FUNC": "green", "INT": "goldenrod", "LOF": "red"}

    if not sge_df.empty:
        scatter = px.scatter(
            sge_df, x="esm_score", y="sge_score", color="sge_class",
            color_discrete_map=color_map,
            hover_data=["position", "wt_aa", "mut_aa"],
            labels={"esm_score": "ESM score", "sge_score": "SGE score",
                    "sge_class": "SGE class"},
            title="ESM score vs SGE score",
        )
        scatter.update_layout(height=420)
        st.plotly_chart(scatter, width="stretch")
    else:
        st.info("No variants with both ESM and SGE scores available.")

with val_col2:
    # ROC curve: LOF vs FUNC (binary) — single AUROC source of truth (Issue 3)
    try:
        from sklearn.metrics import roc_curve, auc as sk_auc

        binary = sge_df[sge_df["sge_class"].isin(["LOF", "FUNC"])].copy()
        if len(binary) > 10:
            binary["label"] = (binary["sge_class"] == "LOF").astype(int)
            # More negative ESM = more damaging, so negate for ROC
            fpr, tpr, _ = roc_curve(binary["label"], -binary["esm_score"])
            roc_auc = sk_auc(fpr, tpr)

            roc_fig = go.Figure()
            roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                          name=f"AUROC = {roc_auc:.3f}"))
            roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                          line=dict(dash="dash", color="grey"),
                                          name="Random"))
            roc_fig.update_layout(
                title=f"ROC curve (LOF vs FUNC, n={len(binary)}) -- AUROC={roc_auc:.3f}",
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                height=420,
            )
            st.plotly_chart(roc_fig, width="stretch")
        else:
            st.info("Not enough LOF/FUNC variants for ROC curve.")
    except ImportError:
        st.warning("scikit-learn not installed; ROC curve unavailable.")

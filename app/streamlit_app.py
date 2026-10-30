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
import requests
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
# The 1JNX structure covers BRCA1 BRCT residues; PDB residue numbering
# matches full-protein numbering for this entry.
PDB_URL = f"https://files.rcsb.org/download/{PDB_ID}.pdb"

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
def fetch_pdb(url: str) -> str:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.text


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
# 3-D viewer
# ---------------------------------------------------------------------------

def render_3d(pdb_text: str, sensitivity: dict, selected_pos: int | None = None):
    view = py3Dmol.view(width=480, height=480)
    view.addModel(pdb_text, "pdb")

    # Base cartoon coloured grey
    view.setStyle({"cartoon": {"color": "grey"}})

    # Colour by average ESM sensitivity
    sens_vals = list(sensitivity.values())
    if sens_vals:
        vmin, vmax = min(sens_vals), max(sens_vals)
    else:
        vmin, vmax = -5, 5
    rng = max(vmax - vmin, 1e-6)

    for pos, score in sensitivity.items():
        frac = (score - vmin) / rng  # 0 = most negative (damaging), 1 = benign
        r = int(255 * (1 - frac))
        b = int(255 * frac)
        hex_col = f"#{r:02x}00{b:02x}"
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
    return view

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="BRCT Variant Explorer", layout="wide")
st.title("BRCT Variant Explorer -- Deep Mutational Scan")

df = load_variants()
metrics = load_metrics()
pdb_text = fetch_pdb(PDB_URL)
sensitivity = avg_sensitivity(df)

# ---------------------------------------------------------------------------
# Session-state for selection
# ---------------------------------------------------------------------------
if "sel_pos" not in st.session_state:
    st.session_state.sel_pos = int(df["position"].min())
if "sel_aa" not in st.session_state:
    st.session_state.sel_aa = "A"

# ---------------------------------------------------------------------------
# Sidebar: selection controls + validation
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
    st.header("Validation metrics")
    st.write(f"**Model:** {metrics['model']}")
    st.write(f"**Variants:** {metrics['n_variants']}")
    st.write(f"**With SGE data:** {metrics['n_with_sge']}")
    st.metric("AUROC (ESM vs SGE)", f"{metrics['auroc']:.4f}")

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

    # Mark WT cells with a symbol overlay
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

    st.plotly_chart(fig, use_container_width=True)

# -- 3D viewer --
with col_3d:
    st.subheader(f"3D structure ({PDB_ID})")
    st.caption("Cartoon coloured by avg ESM sensitivity (red=damaging, blue=benign). "
               "Selected residue in yellow.")
    view = render_3d(pdb_text, sensitivity, selected_pos=st.session_state.sel_pos)
    showmol(view, height=480, width=480)

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
        st.plotly_chart(scatter, use_container_width=True)
    else:
        st.info("No variants with both ESM and SGE scores available.")

with val_col2:
    # ROC curve: LOF vs FUNC (binary)
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
                title=f"ROC curve (LOF vs FUNC) -- AUROC={roc_auc:.3f}",
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                height=420,
            )
            st.plotly_chart(roc_fig, use_container_width=True)
        else:
            st.info("Not enough LOF/FUNC variants for ROC curve.")
    except ImportError:
        st.warning("scikit-learn not installed; ROC curve unavailable.")

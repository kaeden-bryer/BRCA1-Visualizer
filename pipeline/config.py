"""Pipeline constants for BRCA1 BRCT variant scoring.

Canonical coordinate system: full-protein BRCA1 numbering (UniProt P38398).
BRCT tandem-repeat domain = residues 1646-1859 (inclusive, 214 residues).
"""

# ---------------------------------------------------------------------------
# Domain boundaries (full-protein numbering)
# ---------------------------------------------------------------------------
BRCT_START = 1646
BRCT_END = 1859  # inclusive
BRCT_LENGTH = BRCT_END - BRCT_START + 1  # 214 residues

# ---------------------------------------------------------------------------
# Wild-type BRCT domain sequence (UniProt P38398, residues 1646-1859)
# Verified against https://www.uniprot.org/uniprot/P38398
#
# We also store a longer context sequence (extends ~30 residues past 1859)
# which helps ESM-2 make better predictions near the C-terminal boundary.
# Only positions 1646-1859 are scored.
# ---------------------------------------------------------------------------
# Verified against https://rest.uniprot.org/uniprotkb/P38398.fasta
# Residues 1646-1863 (BRCT domain + 4 C-terminal residues to end of protein).
# We score only positions 1646-1859 (first 214 residues); the rest is context.
_BRCT_CONTEXT_SEQ = (
    "VNKRMSMVVSGLTPEEFMLVYKFA"
    "RKHHITLTNLITEETTHVVMKTDA"
    "EFVCERTLKYFLGIAGGKWVVSYF"
    "WVTQSIKERKMLNEHDFEVRGDVV"
    "NGRNHQGPKRARESQDRKIFRGLE"
    "ICCYGPFTNMPTDQLEWMVQLCGA"
    "SVVKELSSFTLGTGVHPIVVVQPD"
    "AWTEDNGFHAIGQMCEAPVVTREW"
    "VLDSVALYQCQELDTYLIPQIP"
    "HSHY"
)

# The scored domain is exactly the first 214 residues
BRCT_WT_SEQ = _BRCT_CONTEXT_SEQ[:BRCT_LENGTH]

assert len(BRCT_WT_SEQ) == BRCT_LENGTH, (
    f"WT sequence length {len(BRCT_WT_SEQ)} != expected {BRCT_LENGTH}"
)

# Full context for ESM-2 input (better predictions at boundaries)
BRCT_CONTEXT_SEQ = _BRCT_CONTEXT_SEQ

# ---------------------------------------------------------------------------
# Standard amino acid alphabet (20 canonical)
# ---------------------------------------------------------------------------
AMINO_ACIDS = sorted("ACDEFGHIKLMNPQRSTVWY")

# ---------------------------------------------------------------------------
# SGE classification thresholds (Findlay et al. 2018)
# Scores are log2, median-synonymous-normalized. More negative = worse.
# ---------------------------------------------------------------------------
SGE_FUNC_THRESHOLD = -0.748   # above this => FUNC
SGE_LOF_THRESHOLD = -1.328    # below this => LOF
# Between the two => INT (intermediate)


def classify_sge(score: float) -> str:
    """Classify an SGE function score into FUNC / INT / LOF."""
    if score > SGE_FUNC_THRESHOLD:
        return "FUNC"
    elif score >= SGE_LOF_THRESHOLD:
        return "INT"
    else:
        return "LOF"


# ---------------------------------------------------------------------------
# ESM-2 model preferences (try larger first, fall back to smaller)
# ---------------------------------------------------------------------------
ESM_MODEL_PRIMARY = "esm2_t33_650M_UR50D"
ESM_MODEL_FALLBACK = "esm2_t30_150M_UR50D"

# ---------------------------------------------------------------------------
# Cache / output paths (relative to repo root)
# ---------------------------------------------------------------------------
CACHE_DIR = "cache"
DATA_DIR = "data"
ESM_CACHE_FILE = "esm_scores.csv"
SGE_CACHE_FILE = "sge_scores.csv"
VARIANT_SCORES_FILE = "variant_scores.csv"
METRICS_FILE = "metrics.json"


def repo_root():
    """Return the absolute path to the repository root."""
    import os
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def cache_path(filename):
    """Return absolute path to a cache file, creating the dir if needed."""
    import os
    d = os.path.join(repo_root(), CACHE_DIR)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, filename)


def data_path(filename):
    """Return absolute path to a data file, creating the dir if needed."""
    import os
    d = os.path.join(repo_root(), DATA_DIR)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, filename)


if __name__ == "__main__":
    print(f"BRCT domain: {BRCT_START}-{BRCT_END} ({BRCT_LENGTH} residues)")
    print(f"WT sequence: {BRCT_WT_SEQ[:30]}...{BRCT_WT_SEQ[-10:]}")
    print(f"Context seq: {len(BRCT_CONTEXT_SEQ)} residues total")
    print(f"Amino acids: {AMINO_ACIDS}")
    print(f"SGE thresholds: FUNC > {SGE_FUNC_THRESHOLD}, LOF < {SGE_LOF_THRESHOLD}")

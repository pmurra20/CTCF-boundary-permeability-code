#!/usr/bin/env bash
# ==============================================================================
#Usage:
#  nanopore_fig.sh --fig FIG --intermediate-dir DIR --outdir DIR
#                 [--input FILE] [--motif FILE] [--bigwig FILE]
#                 [--resources-dir DIR] [--signals-dir DIR]
#                 [--env ENV] [--py SCRIPT] [--py-dir DIR] [--out-prefix PREFIX]
#                 [--all] [--keep-going]
#
#Notes:
#  - Use --all to run all configured figures. In this mode, --outdir is treated as a BASE directory
#    and each figure is written to <outdir>/<FIG_KEY>/...
#  - You can also pass a comma-separated list to --fig (e.g. --fig "Fig.1B,S1B,S21IN").

#
# Purpose
# -------
# Single entrypoint to generate nanopore-based figures from standardized
# intermediate TSV/BED files produced by the preprocessing pipeline.
#
# Key options
# ----------
# --fig               Figure ID (e.g., Fig.1B, 1B, S21D)
# --intermediate-dir  REQUIRED. Directory with Stage-1 outputs (TSV/BED files)
# --outdir            REQUIRED. Where to write outputs for this figure
# --out-prefix        Optional. Defaults to <outdir>/<FIG_KEY>
# --input             Optional. Override main input TSV for the chosen figure
# --motif             Optional. Override motif TSV for figures that need it (e.g., Fig.1C, Fig.1E)
# --bigwig            Optional. Path to bigWig for figures that need it (e.g., Fig.1E). If not provided, auto-download.
# --resources-dir     Optional. Base directory for external resources (default: <repo>/resources)
# --signals-dir       Optional. Where to find provided signal BEDs (default: <resources-dir>/signals if exists, else <resources-dir>)
# --py                Optional. Override full path to the python script to run
# --py-dir            Optional. Directory to search for python scripts
# --env               Optional. Conda env name to activate before running
#--------EXAMPLE---------------
#./scripts/run/nanopore_fig.sh \
#--fig Fig.3B \
#--intermediate-dir "/Users/sergeirudnizky/Manuscripts/CTCF_dynamics_2025/Science_submission/Final_submission/Code/Shell/data" \
#--outdir "/Users/sergeirudnizky/Manuscripts/CTCF_dynamics_2025/Science_submission/Final_submission/Code/Generated_figs/nanopore/3B" \
#--py-dir "/Users/sergeirudnizky/Manuscripts/CTCF_dynamics_2025/Science_submission/Final_submission/Code/Python" \
#--resources-dir "/Users/sergeirudnizky/Manuscripts/CTCF_dynamics_2025/Science_submission/Final_submission/Code/Shell/data/resources"
#
#--------EXAMPLE TO EMIT ALL NANOPORE-BASED FIGURES---
#
#./scripts/run/nanopore_fig.sh \
#  --all \
#  --intermediate-dir "/Users/sergeirudnizky/Manuscripts/CTCF_dynamics_2025/Science_submission/Final_submission/Code/Shell/data" \
#  --outdir "/Users/sergeirudnizky/Manuscripts/CTCF_dynamics_2025/Science_submission/Final_submission/Code/Generated_figs/nanopore" \
#  --py-dir "/Users/sergeirudnizky/Manuscripts/CTCF_dynamics_2025/Science_submission/Final_submission/Code/Python" \
#  --resources-dir "/Users/sergeirudnizky/Manuscripts/CTCF_dynamics_2025/Science_submission/Final_submission/Code/Shell/data/resources"
#
# ==============================================================================

set -euo pipefail
IFS=$'\n\t'

export PYTHONIOENCODING=UTF-8

pick_utf8_locale() {
  local candidates=("C.UTF-8" "en_US.UTF-8")
  for loc in "${candidates[@]}"; do
    if locale -a 2>/dev/null | grep -qx "$loc"; then
      echo "$loc"
      return 0
    fi
  done
  echo ""
}

UTF8_LOC="$(pick_utf8_locale)"
if [[ -n "$UTF8_LOC" ]]; then
  export LANG="$UTF8_LOC"
  export LC_CTYPE="$UTF8_LOC"
fi

need() { command -v "$1" >/dev/null 2>&1 || { echo "Missing: $1" >&2; exit 1; }; }

usage() {
  cat >&2 <<'EOF'
Usage:
  nanopore_fig.sh --fig FIG --intermediate-dir DIR --outdir DIR
                 [--input FILE] [--motif FILE] [--bigwig FILE]
                 [--resources-dir DIR] [--signals-dir DIR]
                 [--env ENV] [--py SCRIPT] [--py-dir DIR] [--out-prefix PREFIX] [--all] [--keep-going]

Examples:
  ./scripts/run/nanopore_fig.sh \
    --fig Fig.3B \
    --intermediate-dir /abs/path/to/stage1_output \
    --outdir /abs/path/to/results/Fig3B \
    --py-dir /abs/path/to/python_scripts \
    --resources-dir /abs/path/to/resources

Notes:
  - --intermediate-dir and --outdir are REQUIRED.
  - --resources-dir defaults to <repo>/resources and is used for cached downloads + provided BEDs.
EOF
}

list_figs() {
  cat <<'EOF'
Configured figures:
  Fig.1B
  Fig.1C
  Fig.1E
  Fig.3B
  Fig.3C
  S1B
  S1C
  S1D
  S1E
  S1F
  S21B
  S21CD
  S21EG
  S21FH
  S21IN
  S20D

Scaffolded:
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
ENV_NAME=""
FIG_RAW=""
FIG_KEY=""
INTERMEDIATE_DIR=""
OUTDIR=""
OUT_PREFIX=""
PY_SCRIPT_OVERRIDE=""
PY_DIR=""
INPUT_OVERRIDE=""
MOTIF_OVERRIDE=""
BIGWIG_OVERRIDE=""
RESOURCES_DIR=""
SIGNALS_DIR=""
KEEP_GOING=0

resolve_py_script() {
  local name="$1"

  if [[ -n "${PY_SCRIPT_OVERRIDE:-}" ]]; then
    [[ -f "$PY_SCRIPT_OVERRIDE" ]] && { echo "$PY_SCRIPT_OVERRIDE"; return 0; }
    echo "Error: --py provided but not found: $PY_SCRIPT_OVERRIDE" >&2
    exit 1
  fi

  if [[ -n "${PY_DIR:-}" && -f "$PY_DIR/$name" ]]; then
    echo "$PY_DIR/$name"; return 0
  fi

  for p in \
    "$ROOT_DIR/$name" \
    "$ROOT_DIR/python_scripts/$name" \
    "$ROOT_DIR/scripts/python/$name" \
    "$ROOT_DIR/src/$name"
  do
    [[ -f "$p" ]] && { echo "$p"; return 0; }
  done

  echo "Error: cannot find $name." >&2
  echo "  Provide --py /path/to/$name OR --py-dir /path/to/python_scripts" >&2
  exit 1
}

pick_first_existing() {
  for f in "$@"; do
    [[ -f "$f" ]] && { echo "$f"; return 0; }
  done
  return 1
}

# Find a signal bed under signals-dir/resources-dir, tolerating "interesect" vs "intersect"
find_signal_bed() {
  local fname="$1"
  local alt1="${fname/fimo_interesect/fimo_intersect}"
  local alt2="${fname/fimo_intersect/fimo_interesect}"

  pick_first_existing \
    "$SIGNALS_DIR/$fname" \
    "$SIGNALS_DIR/$alt1" \
    "$SIGNALS_DIR/$alt2" \
    "$RESOURCES_DIR/$fname" \
    "$RESOURCES_DIR/$alt1" \
    "$RESOURCES_DIR/$alt2" \
    "$RESOURCES_DIR/beds/$fname" \
    "$RESOURCES_DIR/beds/$alt1" \
    "$RESOURCES_DIR/beds/$alt2" \
    "$RESOURCES_DIR/signals/$fname" \
    "$RESOURCES_DIR/signals/$alt1" \
    "$RESOURCES_DIR/signals/$alt2"
}

# Find a bigWig under signals-dir/resources-dir (and common subfolders)
find_bigwig() {
  local fname="$1"
  pick_first_existing \
    "$SIGNALS_DIR/$fname" \
    "$RESOURCES_DIR/$fname" \
    "$RESOURCES_DIR/bw/$fname" \
    "$RESOURCES_DIR/bigwigs/$fname" \
    "$RESOURCES_DIR/signals/$fname"
}

# Convert motif TSV (with chr/start/end/strand columns) to BED6 for computeMatrix
make_bed_from_motif_tsv() {
  local in_tsv="$1"
  local out_bed="$2"

  "$PYTHON_BIN" - "$in_tsv" "$out_bed" <<'PY'
import sys, pandas as pd

inp, outp = sys.argv[1], sys.argv[2]
df = pd.read_csv(inp, sep="\t", comment="#")

def pick(cols):
    for c in cols:
        if c in df.columns:
            return c
    return None

chrcol   = pick(["chr_motif","chrom","chr","seqname"])
startcol = pick(["start_motif","start","chromStart"])
endcol   = pick(["end_motif","end","chromEnd"])
strandcol= pick(["strand_motif","strand"])
namecol  = pick(["motif_id","name","id"])

missing = [x for x in [chrcol,startcol,endcol,strandcol] if x is None]
if missing:
    raise SystemExit(f"TSV missing required columns. Have: {list(df.columns)}")

name = df[namecol].astype(str) if namecol else pd.Series([f"motif{i+1}" for i in range(len(df))])
strand = df[strandcol].astype(str).where(df[strandcol].astype(str).isin(["+","-"]), "+")

bed = pd.DataFrame({
    "chrom": df[chrcol].astype(str),
    "start": df[startcol].astype(int),
    "end": df[endcol].astype(int),
    "name": name,
    "score": 0,
    "strand": strand
})
bed.to_csv(outp, sep="\t", header=False, index=False)
PY
}

# ----------------------------
# Parse args
# ----------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --fig) FIG_RAW="$2"; shift 2 ;;
    --intermediate-dir) INTERMEDIATE_DIR="$2"; shift 2 ;;
    --outdir) OUTDIR="$2"; shift 2 ;;
    --out-prefix) OUT_PREFIX="$2"; shift 2 ;;
    --env) ENV_NAME="$2"; shift 2 ;;
    --py) PY_SCRIPT_OVERRIDE="$2"; shift 2 ;;
    --py-dir) PY_DIR="$2"; shift 2 ;;
    --input) INPUT_OVERRIDE="$2"; shift 2 ;;
    --motif) MOTIF_OVERRIDE="$2"; shift 2 ;;
    --bigwig) BIGWIG_OVERRIDE="$2"; shift 2 ;;
    --resources-dir) RESOURCES_DIR="$2"; shift 2 ;;
    --signals-dir) SIGNALS_DIR="$2"; shift 2 ;;
    --list) list_figs; exit 0 ;;
    --all) FIG_RAW="ALL"; shift ;;
    --keep-going) KEEP_GOING=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

[[ -n "$FIG_RAW" ]] || { echo "Error: --fig is required" >&2; usage; exit 1; }
[[ -n "$INTERMEDIATE_DIR" ]] || { echo "Error: --intermediate-dir is required" >&2; usage; exit 1; }
[[ -n "$OUTDIR" ]] || { echo "Error: --outdir is required" >&2; usage; exit 1; }

# ----------------------------
# Batch mode: --all or --fig "A,B,C"
# Runs this script repeatedly, one figure per subfolder under the provided --outdir.
# ----------------------------
fig_to_key() {
  local raw="$1"
  local key
  key="$(echo "$raw" | tr '[:lower:]' '[:upper:]' | sed 's/[^A-Z0-9]//g')"
  key="${key#FIG}"
  echo "$key"
}

SELF="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"

FIG_LIST=()
if [[ "$(printf '%s' "$FIG_RAW" | tr '[:lower:]' '[:upper:]')" == "ALL" ]]; then
  FIG_LIST=(
    "Fig.1B" "Fig.1C" "Fig.1E"
    "Fig.3B" "Fig.3C"
    "S1B" "S1C" "S1D" "S1E" "S1F"
    "S21B" "S21CD" "S21EG" "S21FH" "S21IN"
    "S20D"
  )
elif [[ "$FIG_RAW" == *","* ]]; then
  IFS=',' read -r -a FIG_LIST <<< "$FIG_RAW"
fi

if (( ${#FIG_LIST[@]} > 0 )); then
  OUTDIR_BASE="$OUTDIR"
  mkdir -p "$OUTDIR_BASE"

  # In batch mode, per-figure overrides are usually ambiguous — ignore them.
  if [[ -n "${INPUT_OVERRIDE:-}" || -n "${MOTIF_OVERRIDE:-}" || -n "${BIGWIG_OVERRIDE:-}" || -n "${PY_SCRIPT_OVERRIDE:-}" || -n "${OUT_PREFIX:-}" ]]; then
    echo "[WARN] Batch mode ignores: --input/--motif/--bigwig/--py/--out-prefix (run per-figure if you need overrides)." >&2
  fi

  failures=0
  for fig in "${FIG_LIST[@]}"; do
    fig="$(echo "$fig" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
    [[ -n "$fig" ]] || continue

    key="$(fig_to_key "$fig")"
    fig_outdir="$OUTDIR_BASE/$key"

    echo "[BATCH] $fig  ->  $fig_outdir" >&2

    cmd=( "$SELF"
      --fig "$fig"
      --intermediate-dir "$INTERMEDIATE_DIR"
      --outdir "$fig_outdir"
    )
    [[ -n "${ENV_NAME:-}" ]] && cmd+=( --env "$ENV_NAME" )
    [[ -n "${PY_DIR:-}" ]] && cmd+=( --py-dir "$PY_DIR" )
    [[ -n "${RESOURCES_DIR:-}" ]] && cmd+=( --resources-dir "$RESOURCES_DIR" )
    [[ -n "${SIGNALS_DIR:-}" ]] && cmd+=( --signals-dir "$SIGNALS_DIR" )

    if ! "${cmd[@]}"; then
      echo "[BATCH] FAILED: $fig" >&2
      failures=$((failures+1))
      (( KEEP_GOING )) || exit 1
    fi
  done

  if (( failures > 0 )); then
    echo "[BATCH] Completed with $failures failure(s). See logs above." >&2
    exit 2
  fi

  echo "[BATCH] All figures completed successfully. Outputs in: $OUTDIR_BASE" >&2
  exit 0
fi


FIG_KEY="$(echo "$FIG_RAW" | tr '[:lower:]' '[:upper:]' | sed 's/[^A-Z0-9]//g')"
FIG_KEY="${FIG_KEY#FIG}"

mkdir -p "$OUTDIR"
if [[ -z "$OUT_PREFIX" ]]; then
  OUT_PREFIX="$OUTDIR/$FIG_KEY"
fi

# resources + signals
if [[ -z "${RESOURCES_DIR:-}" ]]; then
  RESOURCES_DIR="$ROOT_DIR/resources"
fi
if [[ -z "${SIGNALS_DIR:-}" ]]; then
  if [[ -d "$RESOURCES_DIR/signals" ]]; then
    SIGNALS_DIR="$RESOURCES_DIR/signals"
  else
    SIGNALS_DIR="$RESOURCES_DIR"
  fi
fi

# ----------------------------
# Optional conda activation
# ----------------------------
if [[ -n "$ENV_NAME" ]]; then
  if command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1090
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$ENV_NAME"
  elif command -v mamba >/dev/null 2>&1; then
    # shellcheck disable=SC1090
    source "$(mamba info --base)/etc/profile.d/conda.sh"
    conda activate "$ENV_NAME"
  else
    echo "Error: --env was provided but conda/mamba not found on PATH." >&2
    exit 1
  fi
fi

need "$PYTHON_BIN"

echo "[INFO] fig             : $FIG_RAW  (key=$FIG_KEY)"
echo "[INFO] intermediate-dir: $INTERMEDIATE_DIR"
echo "[INFO] outdir          : $OUTDIR"
echo "[INFO] out-prefix      : $OUT_PREFIX"
echo "[INFO] resources-dir   : $RESOURCES_DIR"
echo "[INFO] signals-dir     : $SIGNALS_DIR"
echo "[INFO] python          : $("$PYTHON_BIN" --version 2>&1)"
[[ -n "$PY_DIR" ]] && echo "[INFO] py-dir          : $PY_DIR"
[[ -n "$BIGWIG_OVERRIDE" ]] && echo "[INFO] bigwig          : $BIGWIG_OVERRIDE"

# ----------------------------
# Figure dispatcher
# ----------------------------
case "$FIG_KEY" in

  1B)
    INPUT_TSV="${INPUT_OVERRIDE:-$INTERMEDIATE_DIR/calls_vs_motif_specific.tsv}"
    [[ -f "$INPUT_TSV" ]] || { echo "Missing input: $INPUT_TSV" >&2; exit 1; }

    PY_SCRIPT="$(resolve_py_script zf_scatter_only_with_zfstats.py)"

    "$PYTHON_BIN" "$PY_SCRIPT" \
      -i "$INPUT_TSV" \
      --out-prefix "$OUT_PREFIX" \
      --min_gpc 1 \
      --include_ndr \
      --mol_threshold 1 \
      --exclude_nucleosome_up \
      --exclude_nucleosome_down \
      --hide_zf5 \
      --nuc_ndr_lo 20 \
      --nuc_ndr_hi 60 \
      --occupancy_denominator kept \
      --ymin -0.1 \
      --ymax 1.1 \
      --show_molecules \
      --scatter_size 10 \
      --export_mappable_zf_summary   "$OUTDIR/mappable_per_site_check.tsv" \
      --export_mappability_by_gpc    "$OUTDIR/mappability_by_gpc_check.tsv" \
      --export_zf_balance            "$OUTDIR/ZF_summ_check.tsv" \
      --export_scatter_stats         "$OUTDIR/ZF_scatter_check.tsv" \
      --export_read_category_counts  "$OUTDIR/global_check.tsv"
    ;;

  1C)
    CALLS_TSV="${INPUT_OVERRIDE:-$INTERMEDIATE_DIR/calls_expanded_CGwindow.tsv}"
    [[ -f "$CALLS_TSV" ]] || { echo "Missing input: $CALLS_TSV" >&2; exit 1; }

    MOTIF_TSV="${MOTIF_OVERRIDE:-$INTERMEDIATE_DIR/fimo_specific_ext65_nonoverlap.tsv}"
    [[ -f "$MOTIF_TSV" ]] || { echo "Missing motif file: $MOTIF_TSV" >&2; exit 1; }

    PY_SCRIPT="$(resolve_py_script zf_region_raster.py)"

    "$PYTHON_BIN" "$PY_SCRIPT" \
      -i "$CALLS_TSV" \
      -m "$MOTIF_TSV" \
      --calls-format auto \
      --region "chr16:8792017-8792147" \
      --assign-window 0 --analysis-span 200 \
      --span 18 --core-span 0 --bin 1 \
      --label-cpgs --lengthen-bp 2 --x-axis both \
      --black-is U \
      --order-by ctcf_categories --ctcf-dyn-min-k 2 \
      --nuc-flag-up --nuc-flag-down --nuc-ndr-lo 20 --nuc-ndr-hi 60 --nuc-flanks-when-no-core naked \
      --row-sep --row-sep-color 0.85 --row-sep-width 0.6 \
      --report --show-category-bar --row-category-stripe \
      --export "$OUTDIR/Fig1C.pdf" \
      --category-bar-width-pct 14 \
      --row-category-stripe-width-pct 20 \
      --category-label-fontsize 23 \
      --category-label-min-frac 0.04 \
      --row-px 40 --col-px 10 --no-profile-scatter --marker circle --circle-scale 4
    ;;

  1E)
 CALLS_TSV="${INPUT_OVERRIDE:-$INTERMEDIATE_DIR/calls_vs_motif_specific.tsv}"
    [[ -f "$CALLS_TSV" ]] || { echo "Missing input: $CALLS_TSV" >&2; exit 1; }

    MOTIF_TSV="${MOTIF_OVERRIDE:-$INTERMEDIATE_DIR/fimo_specific_ext65_nonoverlap.tsv}"
    [[ -f "$MOTIF_TSV" ]] || { echo "Missing motif file: $MOTIF_TSV" >&2; exit 1; }

    RAD21_URL="https://www.encodeproject.org/files/ENCFF241ZVM/@@download/ENCFF241ZVM.bigWig"
    RAD21_BW_DEFAULT="$INTERMEDIATE_DIR/resources/ENCFF241ZVM.bigWig"
    RAD21_BW="${BIGWIG_OVERRIDE:-$RAD21_BW_DEFAULT}"

    if [[ ! -f "$RAD21_BW" ]]; then
      echo "[INFO] RAD21 bigWig not found; downloading:"
      echo "       $RAD21_URL"
      mkdir -p "$(dirname "$RAD21_BW")"
      if command -v curl >/dev/null 2>&1; then
        curl -L --fail -o "$RAD21_BW" "$RAD21_URL"
      elif command -v wget >/dev/null 2>&1; then
        wget -O "$RAD21_BW" "$RAD21_URL"
      else
        echo "Error: need curl or wget to download bigWig." >&2
        exit 1
      fi
    fi
    [[ -s "$RAD21_BW" ]] || { echo "Error: bigWig empty or unreadable: $RAD21_BW" >&2; exit 1; }

    PY_Q="$(resolve_py_script assign_quartiles_by_bigwig_windowed.quantile_with_signal.py)"
    DECILE_PREFIX="$OUTDIR/HEK293T_RAD21_deciles"

    "$PYTHON_BIN" "$PY_Q" \
      -b "$RAD21_BW" \
      -m "$MOTIF_TSV" \
      -o "$DECILE_PREFIX" \
      --window 200 \
      --agg median \
      --n-quantiles 10 \
      --quantile-method rank \
      --missing-as-zero \
      --export-all-bed

    # Locate raw exported beds (pre-intersection)
    Q10_BED_RAW="$(pick_first_existing \
      "${DECILE_PREFIX}.median.quantile10.bed" \
    )" || { echo "Error: could not find quantile10 bed for prefix: $DECILE_PREFIX" >&2; exit 1; }

    Q1_BED_RAW="$(pick_first_existing \
      "${DECILE_PREFIX}.median.quantile01.bed" \
      "${DECILE_PREFIX}.median.quantile1.bed" \
    )" || { echo "Error: could not find quantile01/1 bed for prefix: $DECILE_PREFIX" >&2; exit 1; }

    # Intersect motif TSV with the quantile beds (creates *.fimo_intersect.bed outputs)
    PY_I="$(resolve_py_script intersect_motifs_with_bigwigs.v2.py)"

    "$PYTHON_BIN" "$PY_I" \
      --tsv "$MOTIF_TSV" \
      --beds "$Q10_BED_RAW"

    "$PYTHON_BIN" "$PY_I" \
      --tsv "$MOTIF_TSV" \
      --beds "$Q1_BED_RAW"

    # Now pick the produced intersected beds (support both spellings)
    Q10_BED="$(pick_first_existing \
      "${Q10_BED_RAW}.fimo_intersect.bed" \
      "${Q10_BED_RAW}.fimo_interesect.bed" \
    )" || { echo "Error: intersected Q10 bed not found (expected ${Q10_BED_RAW}.fimo_intersect.bed)" >&2; exit 1; }

    Q1_BED="$(pick_first_existing \
      "${Q1_BED_RAW}.fimo_intersect.bed" \
      "${Q1_BED_RAW}.fimo_interesect.bed" \
    )" || { echo "Error: intersected Q1 bed not found (expected ${Q1_BED_RAW}.fimo_intersect.bed)" >&2; exit 1; }

    PY_S="$(resolve_py_script zf_scatter_only_quantiles.py)"

    "$PYTHON_BIN" "$PY_S" \
      -i "$CALLS_TSV" \
      --regions-bed "$Q10_BED" \
      --min_gpc 1 \
      --include_ndr --drop_all_bound \
      --exclude_nucleosome_up \
      --exclude_nucleosome_down \
      --occupancy_denominator kept \
      --export_scatter_matrix "$OUTDIR/Rad21_HEK293T_q10_matrix_check.tsv" \
      --out-prefix "$OUTDIR/Rad21_HEK293T_q10_check"

    "$PYTHON_BIN" "$PY_S" \
      -i "$CALLS_TSV" \
      --regions-bed "$Q1_BED" \
      --min_gpc 1 \
      --include_ndr --drop_all_bound \
      --exclude_nucleosome_up \
      --exclude_nucleosome_down \
      --occupancy_denominator kept \
      --export_scatter_matrix "$OUTDIR/Rad21_HEK293T_q01_matrix_check.tsv" \
      --out-prefix "$OUTDIR/Rad21_HEK293T_q01_check"

    PY_C="$(resolve_py_script cluster_from_matrix.py)"

    "$PYTHON_BIN" "$PY_C" \
      --input "$OUTDIR/Rad21_HEK293T_q10_matrix_check.tsv" \
      --out   "$OUTDIR/Rad21_HEK293T_q10_clusters_from_matrix.tsv" \
      --compare-with "$OUTDIR/Rad21_HEK293T_q01_matrix_check.tsv" \
      --out-compare  "$OUTDIR/Rad21_HEK293T_q01_clusters_from_matrix.tsv" \
      --clusters "1-2,3-8,9-11" \
      --plot-out-prefix "$OUTDIR/Rad21_clusters_prism_like_median95" \
      --stats-out       "$OUTDIR/Rad21_MWU_prism_like_median95.tsv" \
      --group-labels "High Rad21,Low Rad21" \
      --group-colors "#3b82f6,#ef4444" \
      --stats-decimals 8
    ;;

  # ----------------------------
  # Fig.3B (IMPLEMENTED)
  #
  # Uses provided BEDs from --signals-dir / --resources-dir (no downloads).
  # Script: corr_ZF_ChIP.py
  # ----------------------------
  3B)
    INPUT_TSV="${INPUT_OVERRIDE:-$INTERMEDIATE_DIR/calls_vs_motif_specific.tsv}"
    [[ -f "$INPUT_TSV" ]] || { echo "Missing input: $INPUT_TSV" >&2; exit 1; }

    PY_SCRIPT="$(resolve_py_script corr_ZF_ChIP.py)"

    # Signal beds expected in resources/signals (or resources), provided by you.
    sig_basenames=(
      "REST_MMM.median.all.bed.fimo_interesect.bed"
      "SA1_MMM.median.all.bed.fimo_interesect.bed"
      "SA2_MMM.median.all.bed.fimo_interesect.bed"
      "PDS5A_MMM.median.all.bed.fimo_interesect.bed"
      "NIPBL_MMM.median.all.bed.fimo_interesect.bed"
      "RAD21_24DEG_MMM.median.all.bed.fimo_interesect.bed"
      "RAD21_DMSO_MMM.median.all.bed.fimo_interesect.bed"
      "CTCF_MMM.median.all.bed.fimo_interesect.bed"
    )

    labels=(REST STAG1 STAG2 PDS5A NIPBL RAD21_24h_deg RAD21 CTCF)

    signal_paths=()
    for bn in "${sig_basenames[@]}"; do
      p="$(find_signal_bed "$bn" || true)"
      [[ -n "${p:-}" ]] || { echo "Missing signal bed (looked under $SIGNALS_DIR and $RESOURCES_DIR): $bn" >&2; exit 1; }
      signal_paths+=( "$p" )
    done

    OUTPFX="$OUTDIR/dynamic_kept_fig3B_median_sorted"

    "$PYTHON_BIN" "$PY_SCRIPT" \
      -i "$INPUT_TSV" \
      --signals "${signal_paths[@]}" \
      --labels "${labels[@]}" \
      --plot \
      --r_vmin -0.12 --r_vmax 0.12 \
      --pneglog_vmax 7.7 \
      --min_gpc 1 \
      --mol_threshold 1 \
      --hide_zf5 \
      --nuc_ndr_lo 40 --nuc_ndr_hi 60 \
      --exclude_nucleosome_up --exclude_nucleosome_down \
      --pair_filter dynamic \
      --occupancy-denominator kept \
      --out-prefix "$OUTPFX"
    ;;

  3C)
 CALLS_TSV="${INPUT_OVERRIDE:-$INTERMEDIATE_DIR/calls_vs_motif_specific.tsv}"
    [[ -f "$CALLS_TSV" ]] || { echo "Missing input: $CALLS_TSV" >&2; exit 1; }

    MOTIF_TSV="${MOTIF_OVERRIDE:-$INTERMEDIATE_DIR/fimo_specific_ext65_nonoverlap.tsv}"
    [[ -f "$MOTIF_TSV" ]] || { echo "Missing motif file: $MOTIF_TSV" >&2; exit 1; }

    PDS5A_BW_DEFAULT="$INTERMEDIATE_DIR/resources/ChIP_PDS5A_4DNFIZYNQWRI.bw"
    PDS5A_BW="${BIGWIG_OVERRIDE:-$PDS5A_BW_DEFAULT}"

    PY_Q="$(resolve_py_script assign_quartiles_by_bigwig_windowed.quantile_with_signal.py)"
    DECILE_PREFIX="$OUTDIR/RPE1_PDS5A_deciles"

    "$PYTHON_BIN" "$PY_Q" \
      -b "$PDS5A_BW" \
      -m "$MOTIF_TSV" \
      -o "$DECILE_PREFIX" \
      --window 200 \
      --agg median \
      --n-quantiles 10 \
      --quantile-method rank \
      --missing-as-zero \
      --export-all-bed

    # Locate raw exported beds (pre-intersection)
    Q10_BED_RAW="$(pick_first_existing \
      "${DECILE_PREFIX}.median.quantile10.bed" \
    )" || { echo "Error: could not find quantile10 bed for prefix: $DECILE_PREFIX" >&2; exit 1; }

    Q1_BED_RAW="$(pick_first_existing \
      "${DECILE_PREFIX}.median.quantile01.bed" \
      "${DECILE_PREFIX}.median.quantile1.bed" \
    )" || { echo "Error: could not find quantile01/1 bed for prefix: $DECILE_PREFIX" >&2; exit 1; }

    # Intersect motif TSV with the quantile beds (creates *.fimo_intersect.bed outputs)
    PY_I="$(resolve_py_script intersect_motifs_with_bigwigs.v2.py)"

    "$PYTHON_BIN" "$PY_I" \
      --tsv "$MOTIF_TSV" \
      --beds "$Q10_BED_RAW"

    "$PYTHON_BIN" "$PY_I" \
      --tsv "$MOTIF_TSV" \
      --beds "$Q1_BED_RAW"

    # Now pick the produced intersected beds (support both spellings)
    Q10_BED="$(pick_first_existing \
      "${Q10_BED_RAW}.fimo_intersect.bed" \
      "${Q10_BED_RAW}.fimo_interesect.bed" \
    )" || { echo "Error: intersected Q10 bed not found (expected ${Q10_BED_RAW}.fimo_intersect.bed)" >&2; exit 1; }

    Q1_BED="$(pick_first_existing \
      "${Q1_BED_RAW}.fimo_intersect.bed" \
      "${Q1_BED_RAW}.fimo_interesect.bed" \
    )" || { echo "Error: intersected Q1 bed not found (expected ${Q1_BED_RAW}.fimo_intersect.bed)" >&2; exit 1; }

    PY_S="$(resolve_py_script zf_scatter_only_quantiles.py)"

    "$PYTHON_BIN" "$PY_S" \
      -i "$CALLS_TSV" \
      --regions-bed "$Q10_BED" \
      --min_gpc 1 \
      --include_ndr --keep_dynamic_only \
      --exclude_nucleosome_up \
      --exclude_nucleosome_down \
      --occupancy_denominator kept \
      --export_scatter_matrix "$OUTDIR/PDS5A_RPE1_q10_matrix_check.tsv" \
      --out-prefix "$OUTDIR/PDS5A_RPE1_q10_check"

    "$PYTHON_BIN" "$PY_S" \
      -i "$CALLS_TSV" \
      --regions-bed "$Q1_BED" \
      --min_gpc 1 \
      --include_ndr --keep_dynamic_only \
      --exclude_nucleosome_up \
      --exclude_nucleosome_down \
      --occupancy_denominator kept \
      --export_scatter_matrix "$OUTDIR/PDS5A_RPE1_q01_matrix_check.tsv" \
      --out-prefix "$OUTDIR/PDS5A_RPE1_q01_check"

    PY_C="$(resolve_py_script cluster_from_matrix.py)"

    "$PYTHON_BIN" "$PY_C" \
      --input "$OUTDIR/PDS5A_RPE1_q10_matrix_check.tsv" \
      --out   "$OUTDIR/RPDS5A_RPE1_q10_clusters_from_matrix.tsv" \
      --compare-with "$OUTDIR/PDS5A_RPE1_q01_matrix_check.tsv" \
      --out-compare  "$OUTDIR/PDS5A_RPE1_q01_clusters_from_matrix.tsv" \
      --clusters "1-2,3-8,9-11" \
      --plot-out-prefix "$OUTDIR/PDS5A_RPE1_clusters_prism_like_median95" \
      --stats-out       "$OUTDIR/PDS5A_RPE1_MWU_prism_like_median95.tsv" \
      --group-labels "High PDS5A,Low PDS5A" \
      --group-colors "#3b82f6,#ef4444" \
      --stats-decimals 8
    ;;
  # ----------------------------
  # Fig.S1B (IMPLEMENTED)
  # Inputs:
  #   calls_expanded_CGwindow.tsv
  #   fimo_specific_ext1kb_nonoverlap.bed
  # Script: metaplot_enhanced.py
  # ----------------------------
  S1B)
    CALLS_TSV="${INPUT_OVERRIDE:-$INTERMEDIATE_DIR/calls_expanded_CGwindow.tsv}"
    [[ -f "$CALLS_TSV" ]] || { echo "Missing calls TSV: $CALLS_TSV" >&2; exit 1; }

    MOTIF_BED="${MOTIF_OVERRIDE:-$INTERMEDIATE_DIR/fimo_specific_ext1kb_nonoverlap.bed}"
    [[ -f "$MOTIF_BED" ]] || { echo "Missing motif BED: $MOTIF_BED" >&2; exit 1; }

    PY_SCRIPT="$(resolve_py_script metaplot_enhanced.py)"

    "$PYTHON_BIN" "$PY_SCRIPT" \
      --calls "$CALLS_TSV" \
      --motif-beds "$MOTIF_BED" \
      --labels "Fig.S1B" \
      --output "$OUTDIR/CTCF_metaplot.pdf" \
      --threads 8 \
      --bin-size 5 \
      --smooth 5 \
      --window 1000 \
      --csv "$OUTDIR/nuc_1000bp_flank.csv"
    ;;
  # ----------------------------
  # Fig.S1C (IMPLEMENTED)
  # Inputs:
  #   calls_vs_motif_specific.tsv
  # Script: zf_scatter_only.py
  # ----------------------------
  S1C)
    INPUT_TSV="${INPUT_OVERRIDE:-$INTERMEDIATE_DIR/calls_vs_motif_specific.tsv}"
    [[ -f "$INPUT_TSV" ]] || { echo "Missing input TSV: $INPUT_TSV" >&2; exit 1; }

    PY_SCRIPT="$(resolve_py_script zf_scatter_only.py)"

    "$PYTHON_BIN" "$PY_SCRIPT" \
      -i "$INPUT_TSV" \
      --out-prefix "$OUT_PREFIX" \
      --min_gpc 1 \
      --include_ndr \
      --exclude_nucleosome_up \
      --exclude_nucleosome_down \
      --hide_zf5 \
      --nuc_ndr_lo 20 \
      --nuc_ndr_hi 60 \
      --occupancy_denominator kept \
      --dynamic_curve_out "$OUTDIR/dyn_curve" \
      --dynamic_curve_use kept \
      --dynamic_curve_mode eq \
      --dynamic_curve_kmax 5 \
      --dynamic_curve_min_measured 1
    ;;

  # ----------------------------
  # Fig.S1D (IMPLEMENTED)
  # Inputs:
  #   calls_expanded_CGwindow.tsv
  #   fimo_specific_ext65_nonoverlap.tsv
  # Script: zf_region_raster.py
  # Output: three example loci PDFs
  # ----------------------------
  S1D)
    CALLS_TSV="${INPUT_OVERRIDE:-$INTERMEDIATE_DIR/calls_expanded_CGwindow.tsv}"
    [[ -f "$CALLS_TSV" ]] || { echo "Missing input: $CALLS_TSV" >&2; exit 1; }

    MOTIF_TSV="${MOTIF_OVERRIDE:-$INTERMEDIATE_DIR/fimo_specific_ext65_nonoverlap.tsv}"
    [[ -f "$MOTIF_TSV" ]] || { echo "Missing motif file: $MOTIF_TSV" >&2; exit 1; }

    PY_SCRIPT="$(resolve_py_script zf_region_raster.py)"

    run_raster () {
      local region="$1"
      local outpdf="$2"

      "$PYTHON_BIN" "$PY_SCRIPT" \
        -i "$CALLS_TSV" \
        -m "$MOTIF_TSV" \
        --calls-format auto \
        --region "$region" \
        --assign-window 0 --analysis-span 200 \
        --span 200 --core-span 0 --bin 1 \
        --label-cpgs --lengthen-bp 2 --x-axis both \
        --black-is U \
        --order-by ctcf_categories --ctcf-dyn-min-k 2 \
        --nuc-flag-up --nuc-flag-down --nuc-ndr-lo 20 --nuc-ndr-hi 60 --nuc-flanks-when-no-core naked \
        --row-sep --row-sep-color 0.85 --row-sep-width 0.6 \
        --report --show-category-bar --row-category-stripe \
        --export "$outpdf" \
        --category-bar-width-pct 14 \
        --row-category-stripe-width-pct 20 \
        --category-label-fontsize 23 \
        --category-label-min-frac 0.04 \
        --row-px 40 --col-px 10 --no-profile-scatter --marker circle --circle-scale 4
    }

    run_raster "chr16:8792017-8792147"   "$OUTDIR/Fig_S1D_1.pdf"
    run_raster "chr7:142866423-142866553" "$OUTDIR/Fig_S1D_2.pdf"
    run_raster "chr17:7673364-7673494"   "$OUTDIR/Fig_S1D_3.pdf"
    ;;
  # ----------------------------
  # Fig.S1E (IMPLEMENTED)
  # Inputs:
  #   calls_expanded_CGwindow.tsv
  #   fimo_specific_ext65_nonoverlap.tsv
  # Script: zf_metaplot_groups_optimized_nucpar_v2.py
  # Outputs:
  #   global overlay PDF + profile outputs (as produced by the script)
  # ----------------------------
  S1E)
    CALLS_TSV="${INPUT_OVERRIDE:-$INTERMEDIATE_DIR/calls_expanded_CGwindow.tsv}"
    [[ -f "$CALLS_TSV" ]] || { echo "Missing input: $CALLS_TSV" >&2; exit 1; }

    MOTIF_TSV="${MOTIF_OVERRIDE:-$INTERMEDIATE_DIR/fimo_specific_ext65_nonoverlap.tsv}"
    [[ -f "$MOTIF_TSV" ]] || { echo "Missing motif file: $MOTIF_TSV" >&2; exit 1; }

    PY_SCRIPT="$(resolve_py_script zf_metaplot_groups_optimized_nucpar_v2.py)"

    "$PYTHON_BIN" "$PY_SCRIPT" \
      -i "$CALLS_TSV" \
      -m "$MOTIF_TSV" \
      --jobs 12 \
      --window 600 \
      --plot-xwin=-600:600 \
      --bin 5 \
      --exclude_nucleosome_up \
      --exclude_nucleosome_down \
      --nuc_ndr_lo 40 \
      --nuc_ndr_hi 60 \
      --global_plots \
      --export_global_overlay "$OUTDIR/phased_nucleosomes_Fig_S1E.pdf" \
      --profile
    ;;

  # ----------------------------
  # Fig.S1F (IMPLEMENTED)
  # Inputs:
  #   calls_vs_motif_specific.tsv
  #   fimo_specific_ext65_nonoverlap.tsv
  # Script: calc_zf_cond_prob.py
  # Outputs:
  #   conditional-probability plot + count tables in OUTDIR
  # ----------------------------
  S1F)
    CALLS_TSV="${INPUT_OVERRIDE:-$INTERMEDIATE_DIR/calls_vs_motif_specific.tsv}"
    [[ -f "$CALLS_TSV" ]] || { echo "Missing input: $CALLS_TSV" >&2; exit 1; }

    MOTIF_TSV="${MOTIF_OVERRIDE:-$INTERMEDIATE_DIR/fimo_specific_ext65_nonoverlap.tsv}"
    [[ -f "$MOTIF_TSV" ]] || { echo "Missing motif file: $MOTIF_TSV" >&2; exit 1; }

    PY_SCRIPT="$(resolve_py_script calc_zf_cond_prob.py)"

    "$PYTHON_BIN" "$PY_SCRIPT" \
      -i "$CALLS_TSV" \
      -m "$MOTIF_TSV" \
      --corr_all_bins \
      --min-gpc 1 \
      --mol-threshold 2 \
      --hide_zf5 \
      --exclude_nucleosome_up \
      --exclude_nucleosome_down \
      --nuc_ndr_lo 20 \
      --nuc_ndr_hi 60 \
      --cp_vmin 0.0 \
      --cp_vmax 1.0 \
      --export_cond_prob_bound   "$OUTDIR/conditional_probability_Fig_S1F.pdf" \
      --export_cond_counts_bound "$OUTDIR/n_i1_check.tsv" \
      --export_cell_counts_bound "$OUTDIR/cell_i1_check.tsv" \
      --drop_all_bound
    ;;

  # ----------------------------
  # Fig.21B (IMPLEMENTED)
  #
  # Uses provided BEDs from --signals-dir / --resources-dir (no downloads).
  # Script: corr_ZF_ChIP.py
  # ----------------------------
  S21B)
    INPUT_TSV="${INPUT_OVERRIDE:-$INTERMEDIATE_DIR/calls_vs_motif_specific.tsv}"
    [[ -f "$INPUT_TSV" ]] || { echo "Missing input: $INPUT_TSV" >&2; exit 1; }

    PY_SCRIPT="$(resolve_py_script corr_ZF_ChIP.py)"

    # Signal beds expected in resources/signals (or resources), provided by you.
    sig_basenames=(
      "REST_MMM.median.all.bed.fimo_interesect.bed"
      "SA2_MMM.median.all.bed.fimo_interesect.bed"
      "PDS5A_MMM.median.all.bed.fimo_interesect.bed"
      "NIPBL_MMM.median.all.bed.fimo_interesect.bed"
      "SA1_MMM.median.all.bed.fimo_interesect.bed"
      "RAD21_24DEG_MMM.median.all.bed.fimo_interesect.bed"
      "RAD21_DMSO_MMM.median.all.bed.fimo_interesect.bed"
      "CTCF_MMM.median.all.bed.fimo_interesect.bed"
    )

    labels=(REST STAG2 PDS5A NIPBL STAG1 RAD21_24h_deg RAD21 CTCF)

    signal_paths=()
    for bn in "${sig_basenames[@]}"; do
      p="$(find_signal_bed "$bn" || true)"
      [[ -n "${p:-}" ]] || { echo "Missing signal bed (looked under $SIGNALS_DIR and $RESOURCES_DIR): $bn" >&2; exit 1; }
      signal_paths+=( "$p" )
    done

    OUTPFX="$OUTDIR/21B_pvalues"

    "$PYTHON_BIN" "$PY_SCRIPT" \
      -i "$INPUT_TSV" \
      --signals "${signal_paths[@]}" \
      --labels "${labels[@]}" \
      --plot \
      --r_vmin -0.12 --r_vmax 0.12 \
      --pneglog_vmax 7.5 \
      --min_gpc 1 \
      --mol_threshold 1 \
      --hide_zf5 \
      --nuc_ndr_lo 40 --nuc_ndr_hi 60 \
      --exclude_nucleosome_up --exclude_nucleosome_down \
      --pair_filter dynamic \
      --occupancy-denominator kept \
      --out-prefix "$OUTPFX"
    ;;

  # ----------------------------
  # Fig.21CD (IMPLEMENTED)
  #
  # Uses provided BEDs from --signals-dir / --resources-dir (no downloads).
  # Script: corr_ZF_ChIP.py
  # ----------------------------
  S21CD)
    INPUT_TSV="${INPUT_OVERRIDE:-$INTERMEDIATE_DIR/calls_vs_motif_specific.tsv}"
    [[ -f "$INPUT_TSV" ]] || { echo "Missing input: $INPUT_TSV" >&2; exit 1; }

    PY_SCRIPT="$(resolve_py_script corr_ZF_ChIP.py)"

    # Signal beds expected in resources/signals (or resources), provided by you.
    sig_basenames=(
      "REST_MMM.median.all.bed.fimo_interesect.bed"
      "SA2_MMM.median.all.bed.fimo_interesect.bed"
      "PDS5A_MMM.median.all.bed.fimo_interesect.bed"
      "NIPBL_MMM.median.all.bed.fimo_interesect.bed"
      "SA1_MMM.median.all.bed.fimo_interesect.bed"
      "RAD21_24DEG_MMM.median.all.bed.fimo_interesect.bed"
      "RAD21_DMSO_MMM.median.all.bed.fimo_interesect.bed"
      "CTCF_MMM.median.all.bed.fimo_interesect.bed"
    )

    labels=(REST STAG2 PDS5A NIPBL STAG1 RAD21_24h_deg RAD21 CTCF)

    signal_paths=()
    for bn in "${sig_basenames[@]}"; do
      p="$(find_signal_bed "$bn" || true)"
      [[ -n "${p:-}" ]] || { echo "Missing signal bed (looked under $SIGNALS_DIR and $RESOURCES_DIR): $bn" >&2; exit 1; }
      signal_paths+=( "$p" )
    done

    OUTPFX="$OUTDIR/21CD_all"

    "$PYTHON_BIN" "$PY_SCRIPT" \
      -i "$INPUT_TSV" \
      --signals "${signal_paths[@]}" \
      --labels "${labels[@]}" \
      --plot \
      --r_vmin -0.12 --r_vmax 0.12 \
      --pneglog_vmax 7.5 \
      --min_gpc 1 \
      --mol_threshold 1 \
      --hide_zf5 \
      --nuc_ndr_lo 40 --nuc_ndr_hi 60 \
      --exclude_nucleosome_up --exclude_nucleosome_down \
      --pair_filter all \
      --occupancy-denominator kept \
      --out-prefix "$OUTPFX"
    ;;

  # ----------------------------
  # Fig.21EG (IMPLEMENTED)
  #
  # Uses provided BEDs from --signals-dir / --resources-dir (no downloads).
  # Script: corr_ZF_ChIP.py
  # ----------------------------
  S21EG)
    INPUT_TSV="${INPUT_OVERRIDE:-$INTERMEDIATE_DIR/calls_vs_motif_specific.tsv}"
    [[ -f "$INPUT_TSV" ]] || { echo "Missing input: $INPUT_TSV" >&2; exit 1; }

    PY_SCRIPT="$(resolve_py_script corr_ZF_ChIP.py)"

    # Signal beds expected in resources/signals (or resources), provided by you.
    sig_basenames=(
      "CTCF_CR_MMM.median.all.bed.fimo_interesect.bed"
      "CTCF_HEK_MMM.median.all.bed.fimo_interesect.bed"
      "CTCF_HeLa_MMM.median.all.bed.fimo_interesect.bed"
      "MNase_HEK_MMM.median.all.bed.fimo_interesect.bed"
      "Bethyl_NIPBL_HeLa2_MMM.median.all.bed.fimo_interesect.bed"
      "HA_NIPBL_HeLa1_MMM.median.all.bed.fimo_interesect.bed"
      "NIPBL_HFB_MMM.median.all.bed.fimo_interesect.bed"
      "RAD21_HeLa_MMM.median.all.bed.fimo_interesect.bed"
      "WAPL_RPE1_MMM.median.all.bed.fimo_interesect.bed"
    )

    labels=(CTCF_CR CTCF_HEK CTCF_HeLa MNase_HEK NIPBL_HeLa1 NIPBL_HeLa2 NIPBL_HFB RAD21_HeLa WAPL_RPE1)

    signal_paths=()
    for bn in "${sig_basenames[@]}"; do
      p="$(find_signal_bed "$bn" || true)"
      [[ -n "${p:-}" ]] || { echo "Missing signal bed (looked under $SIGNALS_DIR and $RESOURCES_DIR): $bn" >&2; exit 1; }
      signal_paths+=( "$p" )
    done

    OUTPFX="$OUTDIR/21EG_dynamic"

    "$PYTHON_BIN" "$PY_SCRIPT" \
      -i "$INPUT_TSV" \
      --signals "${signal_paths[@]}" \
      --labels "${labels[@]}" \
      --plot \
      --r_vmin -0.12 --r_vmax 0.12 \
      --pneglog_vmax 7.5 \
      --min_gpc 1 \
      --mol_threshold 1 \
      --hide_zf5 \
      --nuc_ndr_lo 40 --nuc_ndr_hi 60 \
      --exclude_nucleosome_up --exclude_nucleosome_down \
      --pair_filter dynamic \
      --occupancy-denominator kept \
      --out-prefix "$OUTPFX"
    ;;

  # ----------------------------
  # Fig.21FH (IMPLEMENTED)
  #
  # Uses provided BEDs from --signals-dir / --resources-dir (no downloads).
  # Script: corr_ZF_ChIP.py
  # ----------------------------
  S21FH)
    INPUT_TSV="${INPUT_OVERRIDE:-$INTERMEDIATE_DIR/calls_vs_motif_specific.tsv}"
    [[ -f "$INPUT_TSV" ]] || { echo "Missing input: $INPUT_TSV" >&2; exit 1; }

    PY_SCRIPT="$(resolve_py_script corr_ZF_ChIP.py)"

    # Signal beds expected in resources/signals (or resources), provided by you.
    sig_basenames=(
      "CTCF_CR_MMM.median.all.bed.fimo_interesect.bed"
      "CTCF_HEK_MMM.median.all.bed.fimo_interesect.bed"
      "CTCF_HeLa_MMM.median.all.bed.fimo_interesect.bed"
      "MNase_HEK_MMM.median.all.bed.fimo_interesect.bed"
      "Bethyl_NIPBL_HeLa2_MMM.median.all.bed.fimo_interesect.bed"
      "HA_NIPBL_HeLa1_MMM.median.all.bed.fimo_interesect.bed"
      "NIPBL_HFB_MMM.median.all.bed.fimo_interesect.bed"
      "RAD21_HeLa_MMM.median.all.bed.fimo_interesect.bed"
      "WAPL_RPE1_MMM.median.all.bed.fimo_interesect.bed"
    )

    labels=(CTCF_CR CTCF_HEK CTCF_HeLa MNase_HEK NIPBL_HeLa1 NIPBL_HeLa2 NIPBL_HFB RAD21_HeLa WAPL_RPE1)

    signal_paths=()
    for bn in "${sig_basenames[@]}"; do
      p="$(find_signal_bed "$bn" || true)"
      [[ -n "${p:-}" ]] || { echo "Missing signal bed (looked under $SIGNALS_DIR and $RESOURCES_DIR): $bn" >&2; exit 1; }
      signal_paths+=( "$p" )
    done

    OUTPFX="$OUTDIR/21FH_all"

    "$PYTHON_BIN" "$PY_SCRIPT" \
      -i "$INPUT_TSV" \
      --signals "${signal_paths[@]}" \
      --labels "${labels[@]}" \
      --plot \
      --r_vmin -0.12 --r_vmax 0.12 \
      --pneglog_vmax 7.5 \
      --min_gpc 1 \
      --mol_threshold 1 \
      --hide_zf5 \
      --nuc_ndr_lo 40 --nuc_ndr_hi 60 \
      --exclude_nucleosome_up --exclude_nucleosome_down \
      --pair_filter all \
      --occupancy-denominator kept \
      --out-prefix "$OUTPFX"
    ;;
    
  # ----------------------------
  # Fig.S21I–N (IMPLEMENTED)
  #
  # Generates deepTools computeMatrix TABs from bigWigs and plots a 2×3 overlay panel
  # (CTCF in black + factor in color; single y-axis, scaled overlay; no twin axis).
  #
  # Inputs:
  #   - regions: fimo_specific_ext65_nonoverlap.tsv (or BED6 with strand)
  #   - bigWigs provided under --signals-dir / --resources-dir:
  #       CTCF_CUTRUN.bw
  #       HA_NIPBL.bw
  #       PDS5A.bw
  #       Rad21.bw
  #       SA2.bw
  #       SA1.bw
  #       Rad21_dNIPBL.bw
  #
  # Tools:
  #   - computeMatrix (deepTools)
  #
  # Script:
  #   - plot_ctcf_cohesin_overlay_panels_twin.py (imports orient_footprint.py)
  #
  # Output:
  #   - Fig_S21I-N_overlay.pdf
  # ----------------------------
  S21IN)

      need computeMatrix

      # Regions (BED6 with strand) for computeMatrix + plotting - BED6 with computeMatrix insures proper orientation handling of CTCF motif
        # Regions (BED6 with strand) for computeMatrix + plotting
        MOTIF_SRC="${MOTIF_OVERRIDE:-$INTERMEDIATE_DIR/fimo_specific_ext65_nonoverlap.bed}"
        [[ -f "$MOTIF_SRC" ]] || { echo "Missing regions/motif file: $MOTIF_SRC" >&2; exit 1; }

        REGIONS_BED="$OUTDIR/regions_for_matrix.bed"

        # Treat .bed6 as a BED too
        if [[ "$MOTIF_SRC" == *.bed || "$MOTIF_SRC" == *.bed6 ]]; then
          REGIONS_BED="$MOTIF_SRC"
        else
          echo "[INFO] making BED6 from TSV: $MOTIF_SRC" >&2
          make_bed_from_motif_tsv "$MOTIF_SRC" "$REGIONS_BED"
        fi

        # ------------------------------------------------------------------
        # Force: negative X = N-side
        # (JASPAR MA0139.1 5'->3' is C->N, so swap strands before computeMatrix)
        # ------------------------------------------------------------------
        REGIONS_BED_N="$OUTDIR/regions_for_matrix.NsideNeg.bed"
        awk 'BEGIN{OFS="\t"}
             NF<6 {print "ERROR: expected BED6 (strand in col6): " $0 > "/dev/stderr"; exit 2}
             $6=="+" {$6="-"; print; next}
             $6=="-" {$6="+"; print; next}
             {print "ERROR: invalid strand (col6 must be + or -): " $0 > "/dev/stderr"; exit 2}' \
          "$REGIONS_BED" > "$REGIONS_BED_N"


        REGIONS_BED="$REGIONS_BED_N"
        echo "[INFO] regions (N-side negative): $REGIONS_BED" >&2


      

      # Expected bigWigs (put these files under --resources-dir/{bigwigs,bw,signals} or --signals-dir)
      # Adjust basenames here if your filenames differ.
      bw_ctcf_bn="CUTRUN_CTCF_HEK293T_Rudnizky.bigWig"
      bw_nipbl_bn="ChIP_NIPBL_HA_4DNFIGNDYFED.bw"
      bw_pds5a_bn="ChIP_PDS5A_4DNFIZYNQWRI.bw"
      bw_rad21_bn="ChIP_Rad21_DMSO_4DNFIWDSKQU7.bw"
      bw_sa2_bn="ChIP_SA2_4DNFIW3TOY91.bw"
      bw_sa1_bn="ChIP_SA1_4DNFIFM7T5ID.bw"
      bw_rad21_dn_bn="ChIP_Rad21_24h_dTAG_4DNFI1IY1AJA.bw"

      bw_ctcf="$(find_bigwig "$bw_ctcf_bn" || true)"
      bw_nipbl="$(find_bigwig "$bw_nipbl_bn" || true)"
      bw_pds5a="$(find_bigwig "$bw_pds5a_bn" || true)"
      bw_rad21="$(find_bigwig "$bw_rad21_bn" || true)"
      bw_sa2="$(find_bigwig "$bw_sa2_bn" || true)"
      bw_sa1="$(find_bigwig "$bw_sa1_bn" || true)"
      bw_rad21_dn="$(find_bigwig "$bw_rad21_dn_bn" || true)"

      for v in bw_ctcf bw_nipbl bw_pds5a bw_rad21 bw_sa2 bw_sa1 bw_rad21_dn; do
        [[ -n "${!v:-}" ]] || { echo "Missing required bigWig ($v). Expected basenames: $bw_ctcf_bn $bw_nipbl_bn $bw_pds5a_bn $bw_rad21_bn $bw_sa2_bn $bw_sa1_bn $bw_rad21_dn_bn" >&2; exit 1; }
      done

      TABDIR="$OUTDIR/computeMatrix_tabs"
      mkdir -p "$TABDIR"

      WIN=500
      BIN=1
      THREADS=12

    make_tab() {
      local bw="$1"
      local name="$2"
      local tab="$TABDIR/${name}.tab"
      local mat="$TABDIR/${name}.mat.gz"

      if [[ ! -s "$tab" ]]; then
        echo "[computeMatrix] $name" >&2
        computeMatrix reference-point \
          -S "$bw" \
          -R "$REGIONS_BED" \
          --referencePoint center \
          -b "$WIN" -a "$WIN" \
          --binSize "$BIN" \
          -p "$THREADS" \
          -o "$mat" \
          --outFileNameMatrix "$tab" \
          1>/dev/null
      else
        echo "[computeMatrix] reuse existing: $tab" >&2
      fi

      printf '%s\n' "$tab"
    }


      tab_ctcf="$(make_tab "$bw_ctcf" "CTCF")"
      tab_nipbl="$(make_tab "$bw_nipbl" "NIPBL")"
      tab_pds5a="$(make_tab "$bw_pds5a" "PDS5A")"
      tab_rad21="$(make_tab "$bw_rad21" "Rad21")"
      tab_sa2="$(make_tab "$bw_sa2" "SA2")"
      tab_sa1="$(make_tab "$bw_sa1" "SA1")"
      tab_rad21_dn="$(make_tab "$bw_rad21_dn" "Rad21_dNIPBL")"

      PY_PLOT="$(resolve_py_script plot_ctcf_cohesin_overlay_panels_twin.py)"
      # ensure orient_footprint.py (imported by the plot script) is discoverable
      export PYTHONPATH="$(dirname "$PY_PLOT"):${PYTHONPATH:-}"

      OUTPDF="$OUTDIR/Fig_S21I-N_overlay.pdf"

    "$PYTHON_BIN" "$PY_PLOT" \
      --regions "$REGIONS_BED" \
      --ref-tab "$tab_ctcf" \
      --track "NIPBL:orangered:$tab_nipbl" \
      --track "PDS5A:blue:$tab_pds5a" \
      --track "Rad21:deepskyblue:$tab_rad21" \
      --track "SA2:orangered:$tab_sa2" \
      --track "SA1:blue:$tab_sa1" \
      --track "Rad21 ΔNIPBL:deepskyblue:$tab_rad21_dn" \
      --letters I,J,K,L,M,N \
      --xlim-bp -500 500 \
      --tick-every 200 \
      --binsize 1 \
      --ncols 3 \
      --twin-y \
      --ref-ylim 0 2 \
      --ref-ylabel "CTCF CUT&RUN (RPM)" \
      --y2-label-template "ChIP {label} (RPKM)" \
      --xlabel "bp from the CTCF motif center" \
      --out "$OUTPDF"


      echo "[DONE] wrote $OUTPDF"
      ;;
  # ----------------------------
  # Fig.S20D (IMPLEMENTED)
  #
  # Generates deepTools computeMatrix TABs from bigWigs and plots a 2×3 overlay panel
  # (CTCF in black + factor in color; single y-axis, scaled overlay; no twin axis).
  #
  # Inputs:
  #   - regions: fimo_specific_ext65_nonoverlap.tsv (or BED6 with strand)
  #   - bigWigs provided under --signals-dir / --resources-dir:
  #       CUTRUN_CTCF_HEK293T_Rudnizky.bigWig
  #
  #       MNase_HEK293T_Rudnizky.bigWig
  #
  #
  #
  #
  #
  # Tools:
  #   - computeMatrix (deepTools)
  #
  # Script:
  #   - plot_ctcf_cohesin_overlay_panels_twin.py (imports orient_footprint.py)
  #
  # Output:
  #   - Fig_20D_overlay.pdf
  # ----------------------------
  S20D)
      need computeMatrix

      # Regions (BED6 with strand) for computeMatrix + plotting - BED6 with computeMatrix insures proper orientation handling of CTCF motif
        # Regions (BED6 with strand) for computeMatrix + plotting
        MOTIF_SRC="${MOTIF_OVERRIDE:-$INTERMEDIATE_DIR/fimo_specific_ext65_nonoverlap.bed}"
        [[ -f "$MOTIF_SRC" ]] || { echo "Missing regions/motif file: $MOTIF_SRC" >&2; exit 1; }

        REGIONS_BED="$OUTDIR/regions_for_matrix.bed"

        # Treat .bed6 as a BED too
        if [[ "$MOTIF_SRC" == *.bed || "$MOTIF_SRC" == *.bed6 ]]; then
          REGIONS_BED="$MOTIF_SRC"
        else
          echo "[INFO] making BED6 from TSV: $MOTIF_SRC" >&2
          make_bed_from_motif_tsv "$MOTIF_SRC" "$REGIONS_BED"
        fi

        # ------------------------------------------------------------------
        # Force: negative X = N-side
        # (JASPAR MA0139.1 5'->3' is C->N, so swap strands before computeMatrix)
        # ------------------------------------------------------------------
        REGIONS_BED_N="$OUTDIR/regions_for_matrix.NsideNeg.bed"
        awk 'BEGIN{OFS="\t"}
             NF<6 {print "ERROR: expected BED6 (strand in col6): " $0 > "/dev/stderr"; exit 2}
             $6=="+" {$6="-"; print; next}
             $6=="-" {$6="+"; print; next}
             {print "ERROR: invalid strand (col6 must be + or -): " $0 > "/dev/stderr"; exit 2}' \
          "$REGIONS_BED" > "$REGIONS_BED_N"


        REGIONS_BED="$REGIONS_BED_N"
        echo "[INFO] regions (N-side negative): $REGIONS_BED" >&2


      

      # Expected bigWigs (put these files under --resources-dir/{bigwigs,bw,signals} or --signals-dir)
      # Adjust basenames here if your filenames differ.
      bw_ctcf_bn="CUTRUN_CTCF_HEK293T_Rudnizky.bigWig"
      bw_mnase_bn="MNase_HEK293T_Rudnizky.bigWig"

      bw_ctcf="$(find_bigwig "$bw_ctcf_bn" || true)"
      bw_mnase="$(find_bigwig "$bw_mnase_bn" || true)"

      for v in bw_ctcf bw_mnase; do
        [[ -n "${!v:-}" ]] || { echo "Missing required bigWig ($v). Expected basenames: $bw_ctcf_bn $bw_mnase_bn" >&2; exit 1; }
      done

      TABDIR="$OUTDIR/computeMatrix_tabs"
      mkdir -p "$TABDIR"

      WIN=1200
      BIN=1
      THREADS=12

    make_tab() {
      local bw="$1"
      local name="$2"
      local tab="$TABDIR/${name}.tab"
      local mat="$TABDIR/${name}.mat.gz"

      if [[ ! -s "$tab" ]]; then
        echo "[computeMatrix] $name" >&2
        computeMatrix reference-point \
          -S "$bw" \
          -R "$REGIONS_BED" \
          --referencePoint center \
          -b "$WIN" -a "$WIN" \
          --binSize "$BIN" \
          -p "$THREADS" \
          -o "$mat" \
          --outFileNameMatrix "$tab" \
          1>/dev/null
      else
        echo "[computeMatrix] reuse existing: $tab" >&2
      fi

      printf '%s\n' "$tab"
    }


      tab_ctcf="$(make_tab "$bw_ctcf" "CTCF")"
      tab_mnase="$(make_tab "$bw_mnase" "MNase")"


      PY_PLOT="$(resolve_py_script plot_ctcf_cohesin_overlay_panels_v2.py)"
      # ensure orient_footprint.py (imported by the plot script) is discoverable
      export PYTHONPATH="$(dirname "$PY_PLOT"):${PYTHONPATH:-}"

      OUTPDF="$OUTDIR/Fig_S20D_overlay.pdf"

    "$PYTHON_BIN" "$PY_PLOT" \
      --regions "$REGIONS_BED" \
      --ref-tab "$tab_ctcf" \
      --track "MNase:red:$tab_mnase" \
      --letters D \
      --xlim-bp -1200 1200 \
      --tick-every 400 \
      --binsize 1 \
      --ncols 3 \
      --twin-y \
      --error sd --error-which both --error-alpha 0.25 \
      --ref-ylim 0 2 \
      --ref-ylabel "CTCF CUT&RUN (RPM)" \
      --y2-label-template "MNase {label} (RPKM)" \
      --xlabel "bp from the CTCF motif center" \
      --out "$OUTPDF"


      echo "[DONE] wrote $OUTPDF"
      ;;
  # ----------
    
    
*)
    echo "Unknown figure key: $FIG_KEY" >&2
    echo "Run: $0 --list" >&2
    exit 1
    ;;
esac

echo "[DONE] $FIG_KEY outputs in: $OUTDIR"


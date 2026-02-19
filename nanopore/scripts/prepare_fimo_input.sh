#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'
export LC_ALL=C

need() { command -v "$1" >/dev/null 2>&1 || { echo "Missing: $1" >&2; exit 1; }; }

FIMO_BIN="${FIMO_BIN:-fimo}"
#make sure you have bedtools: https://bedtools.readthedocs.io/en/latest/, samtools:https://www.htslib.org/, FIMO: https://meme-suite.org/meme/doc/fimo.html
#python3 for helper function

need python3
need bedtools
need samtools
need sort
need awk
need gunzip
need grep
need cut
need tail
need "$FIMO_BIN"

# =============================================================================
# prepare_fimo_input.sh - extract unambigous bp information for single-molecule accessibility using nanopolish CpG methylation output.
# ---------------------
#   1. Extract consensus CUT&RUN summit regions
#   2. Download and prepare ENCODE CTCF peaks
#   3. Download and index hg38.fa reference
#   4. Download CTCF motif from JASPAR ELIXIR
#   5. Generate FASTA for FIMO and run motif scan
#   6. Clean up, extract motif positions, apply +1 bp correction, extend 1kb
#   7. Extract nanopolish methylation calls and realign
#   8. Organize outputs
# Usage: chmod +x prepare_fimo_input.sh
#./prepare_fimo_input.sh BED_DIR --nanopolish NANO_FILE [--clean-temp]
#   --nanopolish: path to nanopolish TSV
#   --clean-temp: remove temporary files after successful run
# Note: Python helper scripts must reside in 'python_scripts/' under the script dir.
#Example:./prepare_fimo_input.sh "/Users/sergeirudnizky/Manuscripts/CTCF_dynamics_2025/Science_submission/Final_submission/Code/Shell/CTCF_peaks"  --nanopolish "/Users/sergeirudnizky/GpC_uncut_full/uncut_full_processed_tsv/header_gpc_filtered_header_200bp_cpggpc_nanonome_NB16_HEK293T_ctrl_fraction_methylation.tsv" =============================================================================

# Resolve script directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_SCRIPTS_DIR="$SCRIPT_DIR/python_scripts"

# Check arguments
if [ "$#" -lt 2 ] || [ "$#" -gt 4 ]; then
  echo "Usage: sh $(basename "$0") BED_DIR --nanopolish NANO_FILE [--clean-temp]" >&2
  exit 1
fi

# Required BED_DIR
BED_DIR="$(cd "$1" && pwd)"
shift

# Initialize options
CLEAN_TMP=0
NANO_TSV=""

# Parse options
while [ "$#" -gt 0 ]; do
  case "$1" in
    --clean-temp)
      CLEAN_TMP=1; shift ;;
    --nanopolish)
      NANO_TSV="$(cd "$(dirname "$2")" && pwd)/$(basename "$2")"; shift 2 ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

# Validate nanopolish TSV (600GB, ~x50 coverage,HEK293T cells)
if [ -z "$NANO_TSV" ] || [ ! -f "$NANO_TSV" ]; then
  echo "Error: must specify existing --nanopolish NANO_FILE" >&2
  exit 1
fi

# Prepare directories (local temp under repo/script dir)
TMP_ROOT="${TMP_ROOT:-$SCRIPT_DIR/tmp}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/data}"

mkdir -p "$TMP_ROOT" "$OUTPUT_DIR"

TMP_DIR="$(mktemp -d "$TMP_ROOT/prepare_fimo_input.XXXXXX")"
trap '[[ "${CLEAN_TMP:-0}" -eq 1 ]] && rm -rf "$TMP_DIR"' EXIT

echo "[INFO] Switching to temp dir: $TMP_DIR"
cd "$TMP_DIR"

# Define resources - hg38 genome/CTCF motif/CTCF ChIP-seq from HEK293T cells
REF_FASTA="$TMP_DIR/hg38.fa"
#CTCF consensus from JASPAR
MOTIF_ID="MA0139.1"
MOTIF_FILE="$TMP_DIR/${MOTIF_ID}.meme"
ENCODE_URL="https://www.encodeproject.org/files/ENCFF314ZAL/@@download/ENCFF314ZAL.bed.gz"
#ENCFF314ZAL.bed is a bed narrowPeak optimal IDR thresholded peaks   GRCh38    2016-12-16

# 1) Prepare reference genome
echo "[1/8] Preparing reference genome..."
if [ ! -f "$REF_FASTA" ]; then
  wget -q https://hgdownload.soe.ucsc.edu/goldenpath/hg38/bigZips/hg38.fa.gz -O "$TMP_DIR/hg38.fa.gz"
  gunzip -f "$TMP_DIR/hg38.fa.gz"
fi
[ -f "$REF_FASTA.fai" ] || samtools faidx "$REF_FASTA"
echo "    -> $REF_FASTA"

# 2) Download CTCF consensus from JASPAR
echo "[2/8] Downloading CTCF motif $MOTIF_ID..."
if [ ! -f "$MOTIF_FILE" ]; then
  wget -q "https://jaspar.elixir.no/api/v1/matrix/${MOTIF_ID}.meme" -O "$MOTIF_FILE"
  [ -s "$MOTIF_FILE" ] || { echo "Error: motif download failed" >&2; exit 1; }
fi
echo "    -> $MOTIF_FILE"

# 3) Extract summit coordinates - CTCF CUT&RUN - generated from Ha lab CUT&RUN replicates, using SEACR stringent mode
echo "[3/8] Extracting summit coordinates from $BED_DIR..."
beds=( "$BED_DIR"/*stringent.bed )
[ ${#beds[@]} -gt 0 ] || { echo "No *stringent.bed files in $BED_DIR" >&2; exit 1; }
for bed in "${beds[@]}"; do
  fbase="$(basename "$bed")"
  out="${fbase%.bed}.summit.sorted.bed"
  awk 'BEGIN{OFS="\t"}{split($6,a,":");split(a[2],b,"-");s=int((b[1]+b[2])/2);print $1,s-1,s,$4}' \
    "$bed" | sort -k1,1 -k2,2n > "$out"
  echo "    -> $out"
done

# 4) Consensus summits - if peak appears in at least 2 biological CUT&RUN replicates =reliable
echo "[4/8] Computing consensus summits..."
bedtools multiinter -i *.summit.sorted.bed > all_summits.multi.bed
awk '$4>=2' all_summits.multi.bed > consensus_2rep_summits.bed
cat *.summit.sorted.bed > all_summits_combined.bed
sort -k1,1 -k2,2n all_summits_combined.bed > all_summits_combined.sorted.bed
bedtools map -a consensus_2rep_summits.bed \
  -b all_summits_combined.sorted.bed -c 4 -o max > consensus_2rep_summits_maxsig.bed

# 5) Intersect ChIP-seq ENCODE peaks with our reliable CUT&RUN peaks (25 bp spanning) - ENCODE database
echo "[5/8] Intersecting with ENCODE CTCF peaks..."
wget -q -L "$ENCODE_URL" -O ENCFF314ZAL.bed.gz
gunzip -c ENCFF314ZAL.bed.gz | sort -k1,1 -k2,2n > ENCODE_CTCF.sorted.bed
awk 'BEGIN{OFS="\t"}{start=$2-25;if(start<0)start=0;print $1,start,$3+25,$4}' consensus_2rep_summits.bed \
  | sort -k1,1 -k2,2n > cutrun_window.bed
bedtools intersect -u -a cutrun_window.bed -b ENCODE_CTCF.sorted.bed > summits_windowed_in_chipseq.bed
echo "    -> summits_windowed_in_chipseq.bed"

# 6) FIMO scan - scan for CTCF motif within the fetched 'peak' sequences
echo "[6/8] Running FIMO..."
bedtools getfasta -fi "$REF_FASTA" -bed summits_windowed_in_chipseq.bed -fo ctcf_peaks.fa
FIMO_BIN="${FIMO_BIN:-fimo}"
"$FIMO_BIN" --oc fimo_out --thresh 1e-4 "$MOTIF_FILE" ctcf_peaks.fa

#remove characters from FIMO generated TSV except of genomic info
python3 "$PYTHON_SCRIPTS_DIR/fimo_cleanup.py" --input fimo_out/fimo.tsv --output fimo_cleaned.tsv
echo "    -> fimo_cleaned.tsv"

# 6b) +1 bp shift - correction for genomic position of CTCF motif - relative to motif center
echo "[6b/8] Shifting FIMO coords by +1..."
awk -F'\t' 'BEGIN{OFS=FS} NR==1{print;next}{$4+=1;$5+=1;print}' fimo_cleaned.tsv > fimo_cleaned_shifted.tsv
echo "    -> fimo_cleaned_shifted.tsv"

echo "[CHECK] Verifying motif sequence matches for fimo_cleaned_shifted.tsv..."

# Ensure required files exist
[ -f "fimo_cleaned_shifted.tsv" ] || { echo "File not found: fimo_cleaned_shifted.tsv" >&2; exit 1; }
[ -f "hg38.fa" ] || { echo "Reference not found: hg38.fa" >&2; exit 1; }

##------------------------- QC: verify that the shift didn't mess up the positions of C ----------------------------------------
## 1. BED for sequence extraction
#tail -n +2 "fimo_cleaned_shifted.tsv" | \
#awk -F'\t' 'BEGIN{OFS="\t"}{print $3, $4-1, $5, $10, ".", $6}' > fimo_regions.bed
#
## 2. Get genome sequence (strand-aware)
#bedtools getfasta -fi "hg38.fa" -bed fimo_regions.bed -name -tab -s > fimo_extracted.seq
#
## 3. Case-insensitive motif vs. genome check
#awk -F'\t' '{
#  split($1, arr, "::")   # arr[1]=motif
#  motif = toupper(arr[1])
#  got   = toupper($2)
#  if (motif != got) {
#    print "LINE", NR, "MISMATCH:", motif, "vs", got
#  }
#}' fimo_extracted.seq > fimo_mismatches.txt
#
## 4. Print summary of mismatches
#total=$(wc -l < fimo_extracted.seq)
#mismatch=$(wc -l < fimo_mismatches.txt)
#pct=$(awk "BEGIN{if ($total>0) print $mismatch/$total*100; else print 0}")
#printf "Total sites: %d\nMismatches: %d (%.2f%%)\n" \
#  "$total" "$mismatch" "$pct"
##------------------------- QC: verify that the shift didn't mess up the positions of C - should be 0% mismatches ---------------

# 6c) Extend ±1kb - to get flanking region for nucleosome positioning - positioned nucleosomes flank CTCF sites
echo "[6c/8] Extending to ±1kb..."
awk -F'\t' 'BEGIN{OFS=FS} NR==1{print;next}{s=$4-991;if(s<1)s=1;e=$5+991;$4=s;$5=e;print}' \
  fimo_cleaned_shifted.tsv > fimo_cleaned_1kb.tsv
echo "    -> fimo_cleaned_1kb.tsv"

# 6d) Extract final BED via Python - chr, start, end, strand
echo "[6d/8] Extracting final BED positions..."
python3 "$PYTHON_SCRIPTS_DIR/extract_fimo_positions.py" --input fimo_cleaned_1kb.tsv --output fimo_positions_sorted.bed
echo "    -> fimo_positions_sorted.bed"

# 6e) Sort final BED window
echo "[6e/8] Sorting final BED to fimo.summit.1kb_window.bed..."
sort -k1,1 -k2,2n fimo_positions_sorted.bed > fimo.summit.1kb_window.bed
echo "    -> fimo.summit.1kb_window.bed"

# 7) Slow step - From nanopolish output file (HEK293T/hg38) - intersect with desired bed to fetch genomic location of interest (CTCF sites +/- 1kb) in nanopolish generated file. The file is large genome-wide 600GB tsv, can be provided upon request. gpc_calls_intersect.tsv ~16.9 GB

tail -n +2 "$NANO_TSV" | \
awk -F'\t' 'BEGIN{OFS="\t"}{print $1,$3,$4,$0}' \
| bedtools intersect -a - -b fimo.summit.1kb_window.bed -wa \
| cut -f4-15 > gpc_calls_intersect.tsv


# 7) Ensure gpc_calls_intersect.tsv is available
if [ ! -f gpc_calls_intersect.tsv ]; then
  if [ -f "$OUTPUT_DIR/gpc_calls_intersect.tsv" ]; then
    echo "[7/8] Copying gpc_calls_intersect.tsv from output dir..."
    cp "$OUTPUT_DIR/gpc_calls_intersect.tsv" ./
  else
    echo "Error: gpc_calls_intersect.tsv not found in TMP_DIR or OUTPUT_DIR" >&2
    exit 1
  fi
fi

# 7a) Filter for single CG per k-mer, otherwise per zinc finger call can be ambigious
awk -F $'\t' '$10 == 1' \
    gpc_calls_intersect.tsv \
  > gpc_calls_intersect_chipseq_cutrun_fimo_all_1kb_ext_singletons.tsv

## QC optional- verify that all num_motifs == 1
#awk -F $'\t' '{
#  if ($10 != 1) {
#    printf("Error: non-singleton motif count %d at line %d\n", $10, NR) > "/dev/stderr"
#    exit 1
#  }
#}
#END {
#  print "All", NR, "records have exactly one motif."
#}' gpc_calls_intersect_chipseq_cutrun_fimo_all_1kb_ext_singletons.tsv


# 7b) Assign GpC per genomic coordinate - initial correction for GC genomic coodrinate - but will be fully corrected in 7d (legacy)
awk -F $'\t' 'BEGIN{OFS="\t"} {
  pos  = $3 + 2
  cpos = ($2 == "-") ? pos+1 : pos
  print $0, pos, cpos
}' \
  gpc_calls_intersect_chipseq_cutrun_fimo_all_1kb_ext_singletons.tsv \
  > gpc_calls_intersect_chipseq_cutrun_fimo_all_1kb_ext_singletons_C_pos.tsv


# 7c) Add GpC methylation call based on log likelyhood ratio (LLR) from nanopolish - threshold from Winston Timp lab
awk -F $'\t' 'BEGIN { OFS = "\t" } {
  llr = $6
  if      (llr >=  1.0) call = "M"
  else if (llr <= -1.0) call = "U"
  else                  call = "N"
  print $0, call
}' \
  gpc_calls_intersect_chipseq_cutrun_fimo_all_1kb_ext_singletons_C_pos.tsv \
  > gpc_calls_intersect_chipseq_cutrun_fimo_all_1kb_ext_singletons_C_pos_mcall.tsv

 echo "    -> gpc_calls_intersect_chipseq_cutrun_fimo_all_1kb_ext_singletons_C_pos_mcall.tsv"

# 7d) Realigning C positions and k-mer - C positions now should completely align with genomic positions
awk -F $'\t' 'BEGIN { OFS = "\t" } {
  seq    = $12
  start0 = $3
  p      = index(seq, "M"); if (p == 0) p = 1
  corr   = start0 + (p - 3)
  target = seq; gsub(/M/, "C", target)
  print $0, corr, target
}' \
  gpc_calls_intersect_chipseq_cutrun_fimo_all_1kb_ext_singletons_C_pos_mcall.tsv \
  > gpc_calls_corrected.tsv

echo "    -> gpc_calls_corrected.tsv"


## 8) ------- QC Optional - verify corrected C positions - we extract nucleotide sequence from hg38 and they should all be C -------
#
## 8a) Build a 1-bp BED at each corrected C position
# awk -F $'\t' 'BEGIN{OFS="\t"} {
#  chr   = $1
#  pos   = $16 + 0          # corrected_C_pos is field 16
#  start = pos - 1
#  if (start < 0) start = 0
#  end   = pos              # half-open interval
#  print chr, start, end    # ONLY these three columns
# }' gpc_calls_corrected.tsv > coords.bed
#
# echo "Extracting bases for $(wc -l < coords.bed) positions..."
#
## 8b) Fetch bases from the reference (tab-delimited: coord <tab> base)
# bedtools getfasta -fi hg38.fa -bed coords.bed -tab > coords.fa
#
## 8c) Count fraction of 'C' at those positions
# awk -F $'\t' 'BEGIN {
#    total = 0; c_cnt = 0;
# }
# {
#    base = toupper($2);
#    total++;
#    if (base == "C") c_cnt++;
# }
# END {
#    pct = (total>0 ? c_cnt/total*100 : 0);
#    printf("Total bases: %d\nC count:     %d\n%% C:        %.2f%%\n", total, c_cnt, pct);
# }' coords.fa
#
## cleanup
# rm -f coords.bed coords.fa
#
# echo "Verification complete"
## 8) ------- QC Optional - verify corrected C positions - we extract nucleotide sequence from hg38 and they should all be C -------

# 9) Summarize key columns (1–13, corrected pos, status) - initial reorganization for taking relevant columns
echo "[9/10] Summarizing key columns (1–13, corrected pos, status)..."
awk -F $'\t' 'BEGIN { OFS = "\t" } {
  #print columns 1–13
  for(i=1; i<=13; i++) printf("%s%s", $i, OFS)
  # col14 ← corrected pos (orig col16)
  printf("%s%s", $16, OFS)
  # col15 ← status (orig col15)
  print $15
}' gpc_calls_corrected.tsv \
  > gpc_calls_intersect_chipseq_cutrun_fimo_all_1kb_ext_singletons_C_pos_corr_mcall.tsv
  
echo "[9.1/10] Reorder columns..."
# ──────────
# 9.1) Simplify & reorder nanopolish-simplified file - chromosome, C position, strand, molecule_id, LLR, LLR_met, LLR_unmet, methylation_status (for LLR assigned in 7c)
# ──────────
awk -F $'\t' 'BEGIN { OFS = "\t" }
{
  # chr, CG pos, strand, read_id, llr_ratio, llr_met, llr_unmet, status
  print $1, $14, $2, $5, $6, $7, $8, $15
}' \
  gpc_calls_intersect_chipseq_cutrun_fimo_all_1kb_ext_singletons_C_pos_corr_mcall.tsv > gpc_calls_relevant_cols.tsv

# ──────────
# 9.2) Sort Nanopolish file
# ──────────
sort -k1,1 -k2,2n gpc_calls_relevant_cols.tsv > gpc_calls_sorted.tsv

# ──────────
# 9.3) Expand CG pos to a 3-bp window
# ──────────
awk 'BEGIN{FS=OFS="\t"}
{
  start = $2 - 1
  end   = $2 + 1
  # $1..$8 are the input fields
  print $1, start, end, $2, $3, $4, $5, $6, $7, $8
}' \
  gpc_calls_sorted.tsv \
> calls_expanded_CGwindow.tsv

echo "[9.3/10] calls_expanded_CGwindow.tsv was generated ..."

# ──────────
# 9.4) Rearrange & trim FIMO file
awk -F $'\t' 'BEGIN{OFS="\t"} { print $3, $4, $5, $6, $10 }' \
  fimo_cleaned_shifted.tsv > fimo_relevant_cols.tsv

# 9.4a) Sort it
sort -k1,1 -k2,2n fimo_relevant_cols.tsv > fimo_sorted.tsv

# 9.5) Add motif-center coordinate (reads fimo_sorted.tsv)
awk -F $'\t' 'BEGIN{OFS="\t"} {
  chr=$1; start=$2; end=$3; strand=$4; seq=$5
  specific = (strand=="-") ? start+10 : start+8
  print chr, start, end, strand, seq, specific
}' fimo_sorted.tsv > fimo_with_specific_nt.tsv


# ──────────
# 9.6) Deduplicate & filter to standard chromosomes
# ──────────
sort fimo_with_specific_nt.tsv | uniq > fimo_unique.tsv

grep -E '^chr([1-9]|1[0-9]|2[0-2]|X|Y)\t' fimo_unique.tsv \
  | grep -v '^chrY\t' \
> fimo_standardchrs.tsv


# ──────────
# 9.7) Extend coordinates ±65 bp around motif center
# ──────────
awk -F $'\t' 'BEGIN { OFS = "\t" }
{
  chr    = $1
  center = $6
  start  = center - 65
  if (start < 0) start = 0
  end    = center + 65

  print chr, start, end, $4, $5, center
}' \
  fimo_standardchrs.tsv \
> fimo_specific_ext65.tsv

# Optional - check if there is a mismatch in the file before bedtools intersect
awk -F'\t' 'NF!=10 { print "line", NR, "has", NF, "fields"; exit }' \
  calls_expanded_CGwindow.tsv

bedtools intersect \
  -a calls_expanded_CGwindow.tsv \
  -b fimo_specific_ext65.tsv \
  -wa -wb \
> calls_vs_motif_specific.tsv

echo "calls_vs_motif_specific.tsv was generated ..."
# 1) sort by motif-chr (col 11) then numeric on motif-center (col 16)
sort -t$'\t' -k11,11 -k16,16n calls_vs_motif_specific.tsv > intersect.sorted.tsv

# 1) Sort your motif windows by chr (col 1) then center (col 6)
sort -t$'\t' -k1,1 -k6,6n fimo_specific_ext65.tsv > sorted_ext65.tsv

# 2) Remove overlapping CTCF motifs - any two motif windows whose centers are < 36 bp apart
awk '
  BEGIN { FS = OFS = "\t" }
  {
    chr = $1        # motif chr
    pos = $6        # motif center
    key = chr ":" pos

    data[NR] = $0
    keys[NR] = key
    ch[NR]   = chr
    ps[NR]   = pos

    if (NR>1 && ch[NR]==ch[NR-1] && (ps[NR]-ps[NR-1])<36) {
      bad[key]                    = 1
      bad[ch[NR-1] ":" ps[NR-1]]  = 1
    }
  }
  END {
    for(i=1; i<=NR; i++) {
      if (!(keys[i] in bad))
        print data[i]
    }
  }
' sorted_ext65.tsv > fimo_specific_ext65_nonoverlap.tsv

# 2) Extend bed for 1kb flanking
echo "fimo_specific_ext65_nonoverlap.tsv was generated ..."

awk -F $'\t' 'BEGIN { OFS = "\t" }
{
  chr    = $1
  center = $6
  start  = center - 935
  if (start < 0) start = 0
  end    = center + 935

  print chr, start, end, $4, $5, center
}' \
  fimo_specific_ext65_nonoverlap.tsv \
> fimo_specific_ext1kb_nonoverlap.bed

echo "fimo_specific_ext1kb_nonoverlap.bed was generated ..."

# Extract bed for n-terminal proximal nucleosome
awk -F $'\t' 'BEGIN { OFS = "\t"; w=65 }
{
  chr    = $1
  motif_center = $6
  strand = $4
  nuc_center = (strand == "+") ? motif_center - 155 : motif_center + 155
  start = nuc_center - int(w/2)
  end   = nuc_center + int(w/2)
  if (start < 0) start = 0
  print chr, start, end, $4, $5, nuc_center
}' fimo_standardchrs.tsv > NUC_N.bed

# Extract bed for c-terminal proximal nucleosome
awk -F $'\t' 'BEGIN { OFS = "\t"; w=65 }
{
  chr    = $1
  motif_center = $6
  strand = $4
  nuc_center = (strand == "+") ? motif_center + 155 : motif_center - 155
  start = nuc_center - int(w/2)
  end   = nuc_center + int(w/2)
  if (start < 0) start = 0
  print chr, start, end, $4, $5, nuc_center
}' fimo_standardchrs.tsv > NUC_C.bed

# Extract bed for n-terminal nucleosome depleted region (NDR)
awk -F $'\t' 'BEGIN { OFS = "\t"; w=65 }
{
  chr    = $1
  motif_center = $6
  strand = $4
  ndr_center = (strand == "+") ? motif_center - 50 : motif_center + 50
  start = ndr_center - int(w/2)
  end   = ndr_center + int(w/2)
  if (start < 0) start = 0
  print chr, start, end, $4, $5, ndr_center
}' fimo_standardchrs.tsv > NDR_N.bed

# Extract bed for c-terminal NDR
awk -F $'\t' 'BEGIN { OFS = "\t"; w=65 }
{
  chr    = $1
  motif_center = $6
  strand = $4
  ndr_center = (strand == "+") ? motif_center + 50 : motif_center - 50
  start = ndr_center - int(w/2)
  end   = ndr_center + int(w/2)
  if (start < 0) start = 0
  print chr, start, end, $4, $5, ndr_center
}' fimo_standardchrs.tsv > NDR_C.bed


# Intersect BEDs with filtered nanopolish
bedtools intersect \
  -a calls_expanded_CGwindow.tsv \
  -b NDR_N.bed \
  -wa -wb \
> calls_vs_NDR_N.tsv

bedtools intersect \
  -a calls_expanded_CGwindow.tsv \
  -b NUC_N.bed  \
  -wa -wb \
> calls_vs_NUC_N.tsv

bedtools intersect \
  -a calls_expanded_CGwindow.tsv \
  -b NDR_C.bed \
  -wa -wb \
> calls_vs_NDR_C.tsv

bedtools intersect \
  -a calls_expanded_CGwindow.tsv \
  -b NUC_C.bed  \
  -wa -wb \
> calls_vs_NUC_C.tsv


# 10) Organize outputs
echo "[10/10] Moving outputs to: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

# Output list


outputs=(
#TSV with GpC 1kb spanning motif
  calls_expanded_CGwindow.tsv
#TSV with GpC 65bp spanning motif
  calls_vs_motif_specific.tsv
#bed files with coordinates of motifs, centred
  fimo_specific_ext65_nonoverlap.tsv
  fimo_specific_ext1kb_nonoverlap.bed
  nucleosomes_left_160bp.bed
  nucleosomes_right_158bp.bed
  NUC_N.bed
  NUC_C.bed
  NDR_N.bed
  NDR_C.bed
  calls_vs_NUC_N.tsv
  calls_vs_NUC_C.tsv
  calls_vs_NDR_N.tsv
  calls_vs_NDR_C.tsv
)

for f in "${outputs[@]}"; do
  if [ -e "$f" ]; then
    mv "$f" "$OUTPUT_DIR"/
  else
    echo "[WARN] Skipping missing file: $f"
  fi
done

# Optional cleanup
echo "[INFO] CLEAN_TMP=$CLEAN_TMP"
if [ "$CLEAN_TMP" -eq 1 ]; then
  rm -rf "$TMP_DIR"
  echo "[INFO] temp directory cleared"
fi

# Completion message
printf "\nPipeline complete! Results in: %s\n" "$OUTPUT_DIR"

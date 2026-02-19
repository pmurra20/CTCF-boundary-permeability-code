# `prepare_fimo_input.sh` — README

This script prepares **FIMO-scanned CTCF motif windows** and **nanopolish-derived single-molecule GpC accessibility calls** in standardized TSV/BED formats that downstream nanopore-figure scripts consume (e.g., `nanopore_fig.sh`).

It is designed for a **large, genome-wide nanopolish TSV** (~600 GB) and will:
- build a **high-confidence CTCF site set** (CUT&RUN summits supported by ≥2 replicates **and** overlapping ENCODE CTCF peaks),
- run **FIMO** to locate the **CTCF motif** in those peak windows,
- create **motif-centered windows** (±65 bp and ±1 kb) and several **nucleosome/NDR proxy windows**,
- intersect nanopolish calls with those windows and produce compact, analysis-ready outputs.

---

## Need to provide

### Required
1) **`BED_DIR`** (positional argument)  
A directory containing one or more SEACR “stringent” CUT&RUN peak BEDs matching:

- `*stringent.bed`

These are used to extract per-replicate summit coordinates and derive a **consensus summit set**.

2) **`--nanopolish NANO_TSV`**  
Path to your **nanopolish methylation calls** TSV (GpC/cytosine methylation output).  
This is expected to be huge (e.g., ~600 GB) and will be streamed via `tail | awk | bedtools intersect`.

### Optional
- `--clean-temp`  
Delete the per-run temporary directory after a successful run.

---

## Auto-download

The script downloads into its temp directory:

- **hg38 reference genome** (`hg38.fa.gz` from UCSC) and builds `hg38.fa.fai` via `samtools faidx`
- **CTCF motif** from JASPAR (ELIXIR API), ID: `MA0139.1` in MEME format
- **ENCODE CTCF peaks** (narrowPeak BED.gz): `ENCFF314ZAL` (GRCh38)

These are used to standardize the site set and to run FIMO.

---

## Dependencies

### Command-line tools (required)
The script checks these are available on `PATH`:

- `python3`
- `bedtools`
- `samtools`
- `awk`, `sort`, `grep`, `cut`, `tail`, `gunzip`
- `fimo` (from MEME Suite), or override via `FIMO_BIN`

**Notes**
- `wget` is used for downloads (the script calls it directly). Make sure `wget` is installed.
- Sorting assumes GNU/BSD `sort` is fine.

### Python helper scripts (required)
The script expects these helper scripts to exist at:

`<script_dir>/python_scripts/`

Required files:
- `python_scripts/fimo_cleanup.py`  
  Cleans the raw `fimo.tsv` down to a minimal, genomic-coordinates table.
- `python_scripts/extract_fimo_positions.py`  
  Extracts final BED-style coordinates from the cleaned/extended FIMO table.

Python packages: these helpers commonly use:
- `pandas` (and standard library)

If you want a simple environment:
```bash
python3 -m pip install pandas
```

---

## Environment variables (optional)

You can control output + temp locations and tool path:

- `FIMO_BIN`  
  Path/name of the FIMO executable (default: `fimo`)
  ```bash
  FIMO_BIN=/path/to/fimo ./prepare_fimo_input.sh ...
  ```

- `TMP_ROOT`  
  Where to place temp directories (default: `<script_dir>/tmp`)

- `OUTPUT_DIR`  
  Final output directory (default: `<script_dir>/data`)

Example:
```bash
TMP_ROOT=/fast_scratch/tmp OUTPUT_DIR=/project/data \
  ./prepare_fimo_input.sh /path/to/BED_DIR --nanopolish /path/to/nanopolish.tsv
```

---

## Usage

```bash
chmod +x prepare_fimo_input.sh

./prepare_fimo_input.sh BED_DIR --nanopolish NANO_FILE [--clean-temp]
```

### Example
```bash
./prepare_fimo_input.sh \
  "/Users/sergeirudnizky/.../Code/Shell/CTCF_peaks" \
  --nanopolish "/Users/sergeirudnizky/.../nanopolish_calls.tsv" \
  --clean-temp
```

---

## Pipeline walkthrough (what each block does)

### 1) Reference genome (hg38)
Downloads `hg38.fa.gz` from UCSC (if missing), gunzips to `hg38.fa`, and indexes with:
- `samtools faidx hg38.fa`

### 2) CTCF motif (JASPAR)
Downloads the CTCF motif **`MA0139.1`** in MEME format for FIMO scanning.

### 3) CTCF CUT&RUN summit extraction (per replicate)
For each `*stringent.bed` peak file, extracts a summit position and writes:
- `*.summit.sorted.bed` (sorted)

### 4) Consensus summits across replicates
Uses `bedtools multiinter` and retains summits appearing in **≥2** replicates:
- `consensus_2rep_summits.bed`  
Then maps the **max score** across replicates for annotation:
- `consensus_2rep_summits_maxsig.bed`

### 5) Cross-filter with ENCODE CTCF peaks
Downloads CTCF ENCODE peaks (`ENCFF314ZAL.bed.gz`) and intersects with CTCF CUT&RUN windows (±25 bp around summit):
- `summits_windowed_in_chipseq.bed`

This creates a more conservative, cross-dataset CTCF site set.

### 6) FIMO motif scan in those windows
Extracts FASTA for the peak windows and runs FIMO:
- `ctcf_peaks.fa`
- `fimo_out/fimo.tsv` (raw MEME output)

Then cleans:
- `fimo_cleaned.tsv` (via `fimo_cleanup.py`)

#### 6b) +1 bp shift
Applies a +1 bp adjustment to match the intended coordinate convention:
- `fimo_cleaned_shifted.tsv`

(There is optional QC code commented out to verify extracted motif sequences match the genome.)

#### 6c) Extend ±1 kb
Creates 1 kb-flank windows around motif coordinates:
- `fimo_cleaned_1kb.tsv`

#### 6d–6e) Extract/sort final windows
Produces:
- `fimo_positions_sorted.bed`
- `fimo.summit.1kb_window.bed`

### 7) Intersect nanopolish calls with motif ±1 kb windows
Streams the nanopolish TSV, reshapes columns, and intersects with the ±1 kb motif windows to keep only relevant calls:
- `gpc_calls_intersect.tsv`  *(still large; ~10–20 GB typical depending on depth/site count)*

### 7a) Singleton motif filter
Keeps calls where the nanopolish record maps to exactly **one** motif instance (to avoid ambiguity):
- `gpc_calls_intersect_chipseq_cutrun_fimo_all_1kb_ext_singletons.tsv`

### 7b–7d) Coordinate correction + methylation call
- Adds initial C position columns (legacy)
- Assigns methylation status from LLR (`M/U/N`) using thresholds ±1.0
- **Realigns** the “C” position by searching the k-mer sequence for the methylated base marker and generating:
  - `gpc_calls_corrected.tsv`

(There is optional QC code commented out to confirm corrected positions are indeed cytosines in hg38.)

### 9) Normalize outputs into standardized files
Builds compact, analysis-ready tables:

- `calls_expanded_CGwindow.tsv`  
  A 3-bp window around corrected C position, with key metadata.
- `calls_vs_motif_specific.tsv`  
  Nanopolish calls intersected with motif-centered ±65 bp windows.

Also processes the FIMO table into motif windows:
- `fimo_specific_ext65.tsv` (±65 bp around motif “center”)
- `fimo_specific_ext65_nonoverlap.tsv` (drops motifs whose centers are <36 bp apart)
- `fimo_specific_ext1kb_nonoverlap.bed` (±935 bp around center; ~1 kb flank)

### Nucleosome/NDR proxy windows
Creates strand-aware coordinate windows relative to motif center (typical offsets):
- `NUC_N.bed`, `NUC_C.bed`  (±~32 bp around ±155 bp)
- `NDR_N.bed`, `NDR_C.bed`  (±~32 bp around ±50 bp)

And intersections with the nanopolish calls:
- `calls_vs_NUC_N.tsv`, `calls_vs_NUC_C.tsv`
- `calls_vs_NDR_N.tsv`, `calls_vs_NDR_C.tsv`

### 10) Output organization
Moves selected outputs into `OUTPUT_DIR` (default: `<script_dir>/data`).

---

## Outputs (what you should expect in `OUTPUT_DIR`)

Primary downstream inputs:
- `calls_expanded_CGwindow.tsv`
- `calls_vs_motif_specific.tsv`
- `fimo_specific_ext65_nonoverlap.tsv`
- `fimo_specific_ext1kb_nonoverlap.bed`

Additional useful BED/TSV products(optional for nucleosome analysis):
- `NUC_N.bed`, `NUC_C.bed`, `NDR_N.bed`, `NDR_C.bed`
- `calls_vs_NUC_N.tsv`, `calls_vs_NUC_C.tsv`, `calls_vs_NDR_N.tsv`, `calls_vs_NDR_C.tsv`

**Note:** The script currently lists two files in `outputs=(...)`:
- `nucleosomes_left_160bp.bed`
- `nucleosomes_right_158bp.bed`

…but these files are **not generated** anywhere in the shown script body. You will either want to:
1) remove them from the `outputs` list, **or**
2) add the generation step (if you intended to create those).

---

## Notes

- The nanopolish intersection step is the major bottleneck and will be I/O heavy.
  - Put `TMP_ROOT` on fast local disk if possible.
- If your nanopolish TSV header layout differs, the `awk/cut` field assumptions may need adjustment.
- If you run on macOS, ensure `bedtools` and `samtools` are from a consistent install (Homebrew recommended).
- If `fimo` isn’t found, set `FIMO_BIN` explicitly.

---

## Minimal one-liner to run

```bash
./prepare_fimo_input.sh /path/to/CTCF_peaks_dir \
  --nanopolish /path/to/nanopolish_calls.tsv \
  --clean-temp
```

# nanopore_fig.sh — Figure Generation Driver

This repository includes `nanopore_fig.sh`, a **single entrypoint** for generating nanopore-based figures from standardized intermediate files produced by your preprocessing pipeline.

It is designed to:
- Run **one figure** (e.g., `Fig.1E`, `S1F`, `S19IN`) reproducibly with a single command.
- Optionally run **all configured figures** in one go (`--all`) or a **comma-separated subset** (`--fig "Fig.1E,S1F,S19IN"`).
- Keep outputs organized with consistent per-figure folders and output prefixes.

---

## What the script does

`nanopore_fig.sh` is a Bash dispatcher that:
1. Validates required inputs (`--intermediate-dir`, `--outdir`, `--fig`).
2. Locates Python analysis scripts (via `--py-dir` or common repo locations).
3. Locates external resources (signal BEDs / bigWigs) under `--resources-dir` / `--signals-dir`.
4. Runs the correct **Python pipeline** (and sometimes deepTools `computeMatrix`) for the requested figure ID.
5. Writes figure PDFs and exported TSV “check” outputs into the requested output directory.

---

## Requirements

### System
- Bash
- `python3` (or set `PYTHON_BIN`)
- Common CLI tools as needed by specific figures:
  - `curl` or `wget` (only if a figure downloads a bigWig automatically, e.g., Fig.1E RAD21 example)
  - `computeMatrix` from **deepTools** (required for `S19IN`, `S20D`)

### Python
The Python figure scripts have their own dependencies. In practice, most runs require the following:

**Core (TSV parsing + plotting)**
- `numpy`
- `pandas`
- `matplotlib`

**Statistics (e.g., Mann–Whitney U / p-values in clustering summaries)**
- `scipy`

**Genomics helpers (used by some figures)**
- `pyBigWig` *(reading bigWig signal for quantile assignment / windowed signal extraction)*
- `pybedtools` + the `bedtools` executable *(if any script performs BED/interval operations via bedtools wrappers)*

**Optional / convenience (depends on the exact scripts you run)**
- `tqdm` *(progress bars, if enabled)*
- `statsmodels` or `scikit-learn` *(only if a specific script imports them)*

**Recommended install (conda)**
```bash
conda create -n nanopore_fig -c conda-forge -c bioconda \
  python=3.10 numpy pandas matplotlib scipy pybigwig pybedtools bedtools deeptools
conda activate nanopore_fig
```

**Minimal install (pip)**
```bash
pip install numpy pandas matplotlib scipy pyBigWig pybedtools
```

If you hit `ImportError: No module named ...`, install the missing module into the same environment and re-run.

---

## Inputs / directory conventions

### `--intermediate-dir` (REQUIRED)
This folder should contain prepare_fimo_input.sh outputs, e.g.:

- `calls_vs_motif_specific.tsv`
- `calls_expanded_CGwindow.tsv`
- `fimo_specific_ext65_nonoverlap.tsv` (or `.bed` variants)
- `fimo_specific_ext1kb_nonoverlap.bed`

### `--resources-dir` (optional, default: `<repo>/resources`)
Used for:
- cached downloads (e.g., downloaded bigWigs)
- provided BEDs / bigWigs used by correlation or overlay figures

### `--signals-dir` (optional)
Where to look for provided signal BEDs / bigWigs. If not provided:
- uses `<resources-dir>/signals` if it exists, else `<resources-dir>`

The script searches a few common subfolders, including:
- `<resources-dir>/signals`
- `<resources-dir>/bw`
- `<resources-dir>/bigwigs`
- `<resources-dir>/beds`

It also tolerates the common typo:
- `fimo_interesect` vs `fimo_intersect`

---

## Usage

### List configured figures
```bash
./scripts/run/nanopore_fig.sh --list
```

### Run one figure
```bash
./scripts/run/nanopore_fig.sh \
  --fig Fig.1E \
  --intermediate-dir "/abs/path/to/stage1_output" \
  --outdir "/abs/path/to/results/1E" \
  --py-dir "/abs/path/to/python_scripts" \
  --resources-dir "/abs/path/to/resources"
```

### Run a comma-separated subset
```bash
./scripts/run/nanopore_fig.sh \
  --fig "Fig.1E,S1F,S19IN" \
  --intermediate-dir "/abs/path/to/stage1_output" \
  --outdir "/abs/path/to/results/nanopore" \
  --py-dir "/abs/path/to/python_scripts" \
  --resources-dir "/abs/path/to/resources"
```

### Run *all* configured figures
In `--all` mode, `--outdir` is treated as a **base directory** and the script writes each figure into:
`<outdir>/<FIG_KEY>/...`

```bash
./scripts/run/nanopore_fig.sh \
  --all \
  --intermediate-dir "/abs/path/to/stage1_output" \
  --outdir "/abs/path/to/results/nanopore" \
  --py-dir "/abs/path/to/python_scripts" \
  --resources-dir "/abs/path/to/resources"
```

### Keep going even if one figure fails
```bash
./scripts/run/nanopore_fig.sh \
  --all --keep-going \
  --intermediate-dir "/abs/path/to/stage1_output" \
  --outdir "/abs/path/to/results/nanopore" \
  --py-dir "/abs/path/to/python_scripts" \
  --resources-dir "/abs/path/to/resources"
```

---

## Important behavior notes

### Figure IDs → FIG_KEY
The script normalizes the figure label to a key by:
- uppercasing
- removing non-alphanumerics
- removing a leading `FIG`

Examples:
- `Fig.1E` → `1E`
- `S19IN` → `S19IN`

### Output naming
- `--outdir` is where outputs go for that run.
- If `--out-prefix` is not provided, the default is:
  - `OUT_PREFIX=<outdir>/<FIG_KEY>`

Many Python scripts write multiple outputs using `--out-prefix`, plus explicit files written to `OUTDIR`.

### Batch mode overrides
When running `--all` or `--fig` with a comma-separated list, per-figure overrides like:
- `--input`, `--motif`, `--bigwig`, `--py`, `--out-prefix`
are typically **ignored** (because they are ambiguous across multiple figures).  
Run figures individually if you need per-figure overrides.

---

## Configured figures (current)

The dispatcher has explicit cases for these figure keys:

- `1B`, `1C`, `1E`
- `3B`, `3C`
- `S1B`, `S1C`, `S1D`, `S1E`, `S1F`
- `S19B`, `S19CD`, `S19EG`, `S19FH`, `S19IN`
- `S20D`

---

## What each figure runs (high level)

### Fig.1B
- Input: `calls_vs_motif_specific.tsv`
- Script: `zf_scatter_only_with_zfstats.py`
- Outputs: scatter/violin + multiple exported TSV “check” tables.

### Fig.1C
- Inputs: `calls_expanded_CGwindow.tsv`, `fimo_specific_ext65_nonoverlap.tsv`
- Script: `zf_region_raster.py`
- Output: `Fig1C.pdf`

### Fig.1E
- Inputs: `calls_vs_motif_specific.tsv`, `fimo_specific_ext65_nonoverlap.tsv`
- Steps:
  1) Assign RAD21 quantiles by bigWig signal (may auto-download RAD21 bigWig).
  2) Intersect motif TSV with quantile beds.
  3) Generate scatter matrices for Q10 and Q1.
  4) Cluster + MWU stats + Prism-like plot.
- Scripts:
  - `assign_quartiles_by_bigwig_windowed.quantile_with_signal.py`
  - `intersect_motifs_with_bigwigs.v2.py`
  - `zf_scatter_only_quantiles.py`
  - `cluster_from_matrix.py`

### Fig.3B / S19B / S19CD / S19EG / S19FH
- Input: `calls_vs_motif_specific.tsv`
- Script: `corr_ZF_ChIP.py`
- Uses **provided** signal BEDs under `--signals-dir` / `--resources-dir`.

### Fig.3C
- Similar to Fig.1E but using a provided PDS5A bigWig (RPE1) and `--keep_dynamic_only`.

### S1B
- Inputs: `calls_expanded_CGwindow.tsv`, `fimo_specific_ext1kb_nonoverlap.bed`
- Script: `metaplot_enhanced.py`
- Output: `CTCF_metaplot.pdf` + CSV.

### S1C
- Input: `calls_vs_motif_specific.tsv`
- Script: `zf_scatter_only.py`
- Output: scatter + dynamic curve outputs.

### S1D
- Inputs: `calls_expanded_CGwindow.tsv`, `fimo_specific_ext65_nonoverlap.tsv`
- Script: `zf_region_raster.py`
- Outputs: three example locus PDFs.

### S1E
- Inputs: `calls_expanded_CGwindow.tsv`, `fimo_specific_ext65_nonoverlap.tsv`
- Script: `zf_metaplot_groups_optimized_nucpar_v2.py`
- Output: `phased_nucleosomes_Fig_S1E.pdf` + profiles.

### S1F
- Inputs: `calls_vs_motif_specific.tsv`, `fimo_specific_ext65_nonoverlap.tsv`
- Script: `calc_zf_cond_prob.py`
- Outputs: conditional probability PDF + count tables.

### S19IN / S20D (deepTools overlay panels)
- Tools: `computeMatrix` (deepTools)
- Inputs:
  - motif regions (BED6 preferred; TSV can be converted to BED6)
  - bigWigs under resources/signals
- Script: `plot_ctcf_cohesin_overlay_panels_*.py`
- Output:
  - `Fig_S19I-N_overlay.pdf` (S19IN)
  - `Fig_S20D_overlay.pdf` (S20D)

These cases also include strand-flipping logic so that “negative X = N-side” relative to the CTCF motif orientation.

---

## Troubleshooting

### “Missing input: …”
Make sure `--intermediate-dir` is the folder that actually contains the Stage-1 TSV/BED outputs.

### “Error: cannot find <script>.py”
Provide:
- `--py-dir /abs/path/to/python_scripts`
or use `--py /abs/path/to/script.py` to override.

### “Missing required bigWig …”
Put the required bigWig files in one of:
- `--signals-dir`
- `--resources-dir`
- `--resources-dir/bw`
- `--resources-dir/bigwigs`
- `--resources-dir/signals`

### computeMatrix not found
Install deepTools (conda is easiest):
```bash
conda install -c bioconda deeptools
```

### macOS bash compatibility
If you encounter errors around bash-4-only parameter expansions, switch to a bash-3.2-safe alternative (`tr`) or run bash ≥ 4 (e.g., via Homebrew).

---

## Repro tips

- Prefer absolute paths for `--intermediate-dir`, `--outdir`, `--py-dir`, and `--resources-dir`.
- Commit the dispatcher and Python scripts together so the pipeline is version-locked.
- Consider capturing `pip freeze` / `conda env export` for the environment used to generate final figures.

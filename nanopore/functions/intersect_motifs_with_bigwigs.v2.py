#!/usr/bin/env python3
"""
Intersect/merge generated BED files with a motif TSV to produce H2A.Z-like outputs.
H2A.Z-like format (5 columns, no header):
    chr    start    end    strand    signal

Typical use with outputs from assign_quartiles_by_bigwig_windowed.py:
    python intersect_motifs_with_bigwigs.py \
        --tsv fimo_specific_ext65_nonoverlap.tsv \
        --beds ctcf_RAD21.mean.all.bed ctcf_RAD21.median.all.bed ctcf_RAD21.max.all.bed \
        --output-dir out \
        --missing-as-zero

Key behavior (UPDATED)
----------------------
By default this script now writes ONLY rows that have a real signal value after the merge,
i.e. it DROPS NaN / missing signal rows. This matches the decile use-case where you expect
~2,400 entries rather than the full motif universe (~20k+).

To preserve the previous behavior (keep rows even if signal is missing), pass:
    --keep-missing

Options:
- --signal-col auto|<index> : Which BED column to use as signal (0-based index). Default: auto = last numeric.
- --missing-as-zero         : If signal is NaN/missing after merge, fill with 0.0 (implies keeping all rows).
- --keep-missing            : Keep rows with missing signal (previous behavior).
- --match coords|center|overlap:
    coords : join on (chr,start,end) [default]
    center : join on (chr, center), where center is computed as floor((start+end)/2) in the BED and TSV must have a numeric center column (default index 5; override with --tsv-center-col)
    overlap: interval overlap (requires pyranges; falls back to coords if unavailable)

Outputs (per input BED):
    <output-dir>/<BED-basename><suffix>   (default suffix: .fimo_interesect.bed)
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="Intersect BEDs with motif TSV to get H2A.Z-like outputs.")
    p.add_argument("--tsv", required=True, help="Motif TSV with columns: chr, start, end, strand, ... (no header).")
    p.add_argument("--beds", nargs="+", required=True, help="One or more BED files to merge.")
    p.add_argument("--output-dir", default="", help="Directory for outputs (default: alongside each BED).")
    p.add_argument("--suffix", default=".fimo_interesect.bed", help="Suffix for output filenames.")
    p.add_argument("--signal-col", default="auto", help="BED column index for signal (0-based) or 'auto'. Default: auto")
    p.add_argument("--missing-as-zero", action="store_true", help="Fill missing/NaN signals with 0.0 (keeps all rows).")
    p.add_argument("--keep-missing", action="store_true",
                   help="Keep rows with missing signal (previous behavior). Default: drop missing rows.")
    p.add_argument("--match", choices=["coords", "center", "overlap"], default="coords",
                   help="How to match BED to TSV. Default: coords")
    p.add_argument("--tsv-center-col", type=int, default=5,
                   help="TSV center column index (used only for --match center). Default: 5")
    return p.parse_args()


def read_tsv(tsv_path: str) -> pd.DataFrame:
    tsv = pd.read_csv(tsv_path, sep="\t", header=None, dtype={0: str, 1: int, 2: int, 3: str})
    tsv["__order__"] = np.arange(len(tsv), dtype=int)  # preserve original order
    return tsv


def read_bed(bed_path: str) -> pd.DataFrame:
    return pd.read_csv(bed_path, sep="\t", header=None, dtype={0: str, 1: int, 2: int})


def pick_signal_column(bed: pd.DataFrame, arg_signal_col: str) -> int:
    if arg_signal_col != "auto":
        idx = int(arg_signal_col)
        if idx < 0 or idx >= bed.shape[1]:
            raise ValueError(f"--signal-col index {idx} out of range for BED with {bed.shape[1]} columns")
        return idx

    # Auto: choose the last numeric column
    numeric_cols = [i for i in range(bed.shape[1]) if pd.api.types.is_numeric_dtype(bed[i])]
    if not numeric_cols:
        # try coercion on the last column as a common case
        last = bed.shape[1] - 1
        coerced = pd.to_numeric(bed[last], errors="coerce")
        if np.isfinite(coerced.to_numpy()).any():
            bed[last] = coerced
            return last
        raise ValueError("Could not auto-detect a numeric signal column in BED. Specify --signal-col.")
    return numeric_cols[-1]


def _finalize_out(m: pd.DataFrame, sig: pd.Series, fill_zero: bool, keep_missing: bool) -> pd.DataFrame:
    # Normalize to numeric
    sig = pd.to_numeric(sig, errors="coerce")

    if fill_zero:
        sig = sig.fillna(0.0)
    elif not keep_missing:
        keep = np.isfinite(sig.to_numpy())
        m = m.loc[keep].copy()
        sig = sig.loc[keep].copy()

    out = pd.DataFrame({
        0: m[0],
        1: m[1],
        2: m[2],
        3: m[3],
        4: sig.astype(float),
    })
    out = out.join(m["__order__"]).sort_values("__order__").drop(columns="__order__")
    return out


def merge_coords(tsv: pd.DataFrame, bed: pd.DataFrame, sig_idx: int, fill_zero: bool, keep_missing: bool) -> pd.DataFrame:
    m = tsv[[0, 1, 2, 3, "__order__"]].merge(
        bed[[0, 1, 2, sig_idx]], on=[0, 1, 2], how="left", suffixes=("", "_bed")
    )
    return _finalize_out(m, m[sig_idx], fill_zero, keep_missing)


def merge_center(tsv: pd.DataFrame, bed: pd.DataFrame, sig_idx: int, fill_zero: bool,
                 keep_missing: bool, tsv_center_col: int) -> pd.DataFrame:
    tsv_center = tsv.copy()
    if tsv_center_col >= tsv_center.shape[1]:
        raise ValueError(f"--tsv-center-col {tsv_center_col} out of range for TSV with {tsv_center.shape[1]} columns.")
    tsv_center["__center__"] = tsv_center[tsv_center_col].astype(int)

    bed_center = bed.copy()
    bed_center["__center__"] = ((bed_center[1] + bed_center[2]) // 2).astype(int)

    m = tsv_center[[0, 1, 2, 3, "__center__", "__order__"]].merge(
        bed_center[[0, "__center__", sig_idx]], on=[0, "__center__"], how="left"
    )
    return _finalize_out(m, m[sig_idx], fill_zero, keep_missing)


def merge_overlap(tsv: pd.DataFrame, bed: pd.DataFrame, sig_idx: int, fill_zero: bool, keep_missing: bool) -> pd.DataFrame:
    try:
        import pyranges as pr
    except Exception:
        return merge_coords(tsv, bed, sig_idx, fill_zero, keep_missing)

    gr_tsv = pr.PyRanges(
        chromosomes=tsv[0].astype(str).values,
        starts=tsv[1].astype(int).values,
        ends=tsv[2].astype(int).values
    )

    bed_sig = bed[[0, 1, 2, sig_idx]].copy()
    bed_sig.columns = ["Chromosome", "Start", "End", "Signal"]
    gr_bed = pr.PyRanges(bed_sig)

    joined = gr_tsv.join(gr_bed, how="left").df  # may produce multiple overlaps
    joined["__key__"] = joined["Chromosome"] + ":" + joined["Start"].astype(str) + "-" + joined["End"].astype(str)
    agg = joined.groupby("__key__", as_index=False)["Signal"].max()

    tsv2 = tsv.copy()
    tsv2["__key__"] = tsv2[0].astype(str) + ":" + tsv2[1].astype(str) + "-" + tsv2[2].astype(str)
    m = tsv2.merge(agg, on="__key__", how="left")

    # ensure the column exists even if no overlaps
    sig = m["Signal"] if "Signal" in m.columns else pd.Series([np.nan] * len(m), index=m.index)
    return _finalize_out(m.rename(columns={0: 0, 1: 1, 2: 2, 3: 3}), sig, fill_zero, keep_missing)


def to_h2az_like(tsv_path: str, bed_path: str, out_dir: str, suffix: str,
                 signal_col: str, fill_zero: bool, keep_missing: bool,
                 match: str, tsv_center_col: int) -> Path:
    tsv = read_tsv(tsv_path)
    bed = read_bed(bed_path)
    sig_idx = pick_signal_column(bed, signal_col)

    if match == "coords":
        out = merge_coords(tsv, bed, sig_idx, fill_zero, keep_missing)
    elif match == "center":
        out = merge_center(tsv, bed, sig_idx, fill_zero, keep_missing, tsv_center_col)
    else:
        out = merge_overlap(tsv, bed, sig_idx, fill_zero, keep_missing)

    bed_base = Path(bed_path).name
    out_name = bed_base + suffix
    out_dir_p = Path(out_dir) if out_dir else Path(bed_path).parent
    out_dir_p.mkdir(parents=True, exist_ok=True)
    out_path = out_dir_p / out_name

    out.to_csv(out_path, sep="\t", header=False, index=False)
    return out_path


def main():
    args = parse_args()

    written = []
    for bed in args.beds:
        out_path = to_h2az_like(
            tsv_path=args.tsv,
            bed_path=bed,
            out_dir=args.output_dir,
            suffix=args.suffix,
            signal_col=args.signal_col,
            fill_zero=args.missing_as_zero,
            keep_missing=args.keep_missing,
            match=args.match,
            tsv_center_col=args.tsv_center_col,
        )
        written.append(str(out_path))

    for p in written:
        print(f"Wrote: {p}")


if __name__ == "__main__":
    main()

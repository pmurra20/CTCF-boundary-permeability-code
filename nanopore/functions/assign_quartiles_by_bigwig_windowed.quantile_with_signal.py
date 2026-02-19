#!/usr/bin/env python3
"""
Assign quantile labels to motif sites based on BigWig signal around each motif
center, with configurable window size and aggregation (mean/median/max).

Outputs:
- Per-selected aggregation, separate BED files for each quantile (unless --no-quantiles).
- A TSV per aggregation with raw per-site statistics: <prefix>.<agg>.signals.tsv.
- Optionally, a BED per aggregation with all sites and the single signal column via --export-all-bed: <prefix>.<agg>.all.bed.
- Optionally, a combined TSV (all aggs together) unless --no-combined-signals is set.

Example:
    python assign_quartiles_by_bigwig_windowed.py \
        --bigwig RAD21_GM12878.bigWig \
        --motifs fimo.summit.bed \
        --output-prefix ctcf_RAD21_quants \
        --window 50 \
        --agg mean median max \
        --n-quantiles 4 \
        --quantile-method qcut \
        --export-all-bed  # writes <prefix>.<agg>.all.bed for each agg

Notes:
- window is a *half*-window in bp. With --window 50, we read [center-50, center+50).
- Aggregations:
    * mean   : np.nanmean over values in the window
    * median : np.nanmedian over values in the window
    * max    : np.nanmax over values in the window
- Missing values: pyBigWig may return None or NaN for missing regions.
  By default, these remain NaN and those sites are excluded from quantiling.
  Use --missing-as-zero to treat missing as 0.0 instead.
- Quantiling:
    * qcut  : equal-population bins (can fail when there are many ties)
    * rank  : rank-based bins (robust to ties)
"""

import argparse
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pyBigWig


def parse_args():
    p = argparse.ArgumentParser(
        description="Assign quantiles by BigWig signal at/around motif centers."
    )
    p.add_argument("-b", "--bigwig", required=True, help="Input BigWig file (ChIP signal).")
    p.add_argument("-m", "--motifs", required=True,
                   help="BED file of motif intervals (chr, start, end, ...).")
    p.add_argument("-o", "--output-prefix", required=True, help="Prefix for output files.")
    p.add_argument("-w", "--window", type=int, default=0,
                   help="Half-window (bp) around center; 0 = single base at center. Default: 0")
    p.add_argument("--agg", nargs="+", default=["mean"],
                   choices=["mean", "median", "max"],
                   help="One or more aggregations to compute. Default: mean")
    p.add_argument("--n-quantiles", type=int, default=4,
                   help="Number of quantiles (e.g., 4 for quartiles, 5 for quintiles). Default: 4")
    p.add_argument("--quantile-method", choices=["qcut", "rank"], default="qcut",
                   help="Method to create quantile bins. Default: qcut")
    p.add_argument("--missing-as-zero", action="store_true",
                   help="Treat missing signal values as 0.0 (instead of NaN).")
    p.add_argument("--write-no-data-bed", action="store_true",
                   help="If set, write a BED of sites with no data for a given aggregation.")
    p.add_argument("--export-all-bed", action="store_true",
                   help="Write a BED per aggregation with all sites and its signal column (no quantiles).")
    p.add_argument("--no-combined-signals", action="store_true",
                   help="Do not write the combined <prefix>.signals.tsv file.")
    p.add_argument("--no-quantiles", action="store_true",
                   help="Skip quantile assignment and per-quantile BED outputs.")
    return p.parse_args()


def _safe_interval(chrom: str, start: int, end: int, chr_sizes: Dict[str, int]) -> Tuple[int, int]:
    """Clamp interval to [0, chrom_length). Return (start, end) or (0, 0) if invalid."""
    if chrom not in chr_sizes:
        return 0, 0
    L = chr_sizes[chrom]
    s = max(0, start)
    e = min(L, end)
    if e <= s:
        return 0, 0
    return s, e


def _agg_from_values(vals: List[float], agg: str) -> float:
    """Aggregate over a list/array of floats with potential Nones/NaNs."""
    if vals is None or len(vals) == 0:
        return np.nan
    arr = np.array(vals, dtype=float)
    # Normalize None to NaN
    arr = np.where(arr == None, np.nan, arr)  # noqa: E711
    if agg == "mean":
        v = np.nanmean(arr)
    elif agg == "median":
        v = np.nanmedian(arr)
    elif agg == "max":
        v = np.nanmax(arr)
    else:
        raise ValueError(f"Unknown agg: {agg}")
    return float(v) if np.isfinite(v) else np.nan


def _compute_signal_for_site(
    bw: pyBigWig.pyBigWig, chrom: str, center: int, halfwin: int, agg: str, chr_sizes: Dict[str, int]
) -> float:
    """Fetch values in [center-halfwin, center+halfwin) and aggregate."""
    if halfwin <= 0:
        s, e = _safe_interval(chrom, center, center + 1, chr_sizes)
    else:
        s, e = _safe_interval(chrom, center - halfwin, center + halfwin, chr_sizes)
    if e <= s:
        return np.nan
    try:
        vals = bw.values(chrom, s, e)
    except RuntimeError:
        return np.nan
    return _agg_from_values(vals, agg=agg)


def _assign_quantiles(series: pd.Series, n_quantiles: int, method: str) -> pd.Series:
    """Assign 1..n_quantiles to series values. NaNs preserved as NaN."""
    s = series.copy()
    # Work only on finite values
    mask = np.isfinite(s.values)
    out = pd.Series(np.full(len(s), np.nan), index=s.index, dtype=float)

    if mask.sum() == 0:
        return out

    s_valid = s[mask]

    if method == "qcut":
        try:
            labels = list(range(1, n_quantiles + 1))
            out.loc[mask] = pd.qcut(s_valid, q=n_quantiles, labels=labels).astype(float).values
        except ValueError:
            # Fallback to rank if too many ties
            method = "rank"

    if method == "rank":
        # Rank 1..N, break ties by average; map to quantile bins
        ranks = s_valid.rank(method="average", na_option="keep")
        bins = np.ceil(ranks * n_quantiles / ranks.max()).astype(int).clip(1, n_quantiles)
        out.loc[mask] = bins.values.astype(float)

    return out


def main():
    args = parse_args()

    # Open BigWig
    try:
        bw = pyBigWig.open(args.bigwig)
    except Exception as e:
        sys.exit(f"Error opening BigWig '{args.bigwig}': {e}")

    # Chrom sizes for bounds
    chr_sizes = bw.chroms()

    # Read motifs (first 3 columns as chr, start, end)
    motifs = pd.read_csv(
        args.motifs, sep="\t", header=None, usecols=[0, 1, 2], names=["chr", "start", "end"]
    )
    if motifs.empty:
        sys.exit("No motifs found in the provided BED.")

    # Compute center (integer midpoint)
    motifs["center"] = ((motifs["start"] + motifs["end"]) // 2).astype(int)

    # Compute signals for each requested aggregation
    for agg in args.agg:
        signals: List[float] = []
        for chrom, center in zip(motifs["chr"].values, motifs["center"].values):
            sig = _compute_signal_for_site(
                bw, str(chrom), int(center), args.window, agg, chr_sizes
            )
            signals.append(sig)
        col = f"signal_{agg}"
        motifs[col] = signals

    # Close BigWig early
    bw.close()

    # argparse converts --missing-as-zero to missing_as_zero
    missing_as_zero = args.missing_as_zero

    # Columns used repeatedly
    signal_cols_all = [f"signal_{a}" for a in args.agg]

    # Write per-aggregation signals TSVs
    for agg in args.agg:
        col = f"signal_{agg}"
        sig_cols = ["chr", "start", "end", "center", col]
        sig_file = f"{args.output_prefix}.{agg}.signals.tsv"
        out_df = motifs[sig_cols].copy()
        if missing_as_zero:
            out_df[col] = out_df[col].fillna(0.0)
        out_df.to_csv(sig_file, sep="\t", index=False)
        print(f"[{agg}] Wrote signals to: {sig_file}")

    # Optionally also write the combined signals TSV unless suppressed
    if not args.no_combined_signals:
        signal_cols = ["chr", "start", "end", "center"] + signal_cols_all
        sig_file = f"{args.output_prefix}.signals.tsv"
        motifs[signal_cols].to_csv(sig_file, sep="\t", index=False)
        print(f"Wrote combined signals to: {sig_file}")

    # Optionally export a per-aggregation ALL-sites BED (no quantiles)
    if args.export_all_bed:
        for agg in args.agg:
            col = f"signal_{agg}"
            all_bed = f"{args.output_prefix}.{agg}.all.bed"
            out_df = motifs[["chr", "start", "end", "center", col]].copy()
            if missing_as_zero:
                out_df[col] = out_df[col].fillna(0.0)
            out_df.to_csv(all_bed, sep="\t", index=False, header=False)
            print(f"[{agg}] Wrote ALL-sites BED to: {all_bed}")

    # Skip quantiles if requested
    if args.no_quantiles:
        return

    # For each aggregation, assign quantiles and write BEDs
    nQ = args.n_quantiles
    for agg in args.agg:
        col = f"signal_{agg}"
        s = motifs[col].copy()

        if missing_as_zero:
            s = s.fillna(0.0)

        q_series = _assign_quantiles(s, n_quantiles=nQ, method=args.quantile_method)
        qcol = f"quantile_{agg}"
        motifs[qcol] = q_series

        # Optionally write "no data" BED (NaN quantile for this agg)
        if args.write_no_data_bed:
            nodata = motifs[~np.isfinite(q_series.values)]
            if not nodata.empty:
                nodata_bed = f"{args.output_prefix}.{agg}.nodata.bed"
                nodata[["chr", "start", "end"]].to_csv(
                    nodata_bed, sep="\t", index=False, header=False
                )
                print(f"Wrote no-data BED for '{agg}' to: {nodata_bed}")

        # Write per-quantile BEDs
        # NOTE: We write the same columns as the *.all.bed export (chr,start,end,center,signal)
        # so downstream tools can auto-detect the signal column reliably.
        for q in range(1, nQ + 1):
            sel = motifs[np.isclose(motifs[qcol].values, float(q))]
            bed_out = f"{args.output_prefix}.{agg}.quantile{q}.bed"
            # chr, start, end, center, signal
            if sel.empty:
                # Emit empty file to keep downstream logic simple
                with open(bed_out, "w") as f:
                    pass
            else:
                sel_out = sel[["chr", "start", "end", "center", col]].copy()
                if missing_as_zero:
                    sel_out[col] = sel_out[col].fillna(0.0)
                sel_out.to_csv(bed_out, sep="\t", index=False, header=False)
            print(f"[{agg}] Wrote quantile {q} BED to: {bed_out}")

    # Also write a per-agg summary TSV of quantile counts
    counts = []
    for agg in args.agg:
        qcol = f"quantile_{agg}"
        if qcol not in motifs.columns:
            continue
        cnt = motifs[qcol].value_counts(dropna=False).sort_index()
        for label, n in cnt.items():
            label_str = "NaN" if not np.isfinite(label) else int(label)
            counts.append({"aggregation": agg, "bin": label_str, "count": int(n)})
    if counts:
        summary = pd.DataFrame(counts)
        sum_file = f"{args.output_prefix}.quantile_counts.tsv"
        summary.to_csv(sum_file, sep="\t", index=False)
        print(f"Wrote quantile counts to: {sum_file}")


if __name__ == "__main__":
    main()

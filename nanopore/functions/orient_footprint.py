#!/usr/bin/env python3
"""
orient_footprint.py — strand-aware orientation & publication plots

Features
--------
- Robust loader for deepTools matrix TAB (pads ragged rows with NaN).
- Auto-detects strand column (BED6 or TSV: chr start end strand motif center).
- Uses --threads to set BLAS/OMP threading env vars.
- Uses --dtype (float32/64) to reduce RAM.
- Flips negative-strand rows so all rows share the same PWM/motif orientation.
  (Whether that corresponds to a protein-centric convention like N→C vs C→N is
  PWM-definition dependent; if needed, mirror at the plotting stage.)
- Optional sorting by center signal (mean/max/median within ±bp window).
- Heatmap auto color scaling (like plotHeatmap when --zMin/--zMax omitted).
- Axis limits: --xlim-bp, --ylim-profile, --row-range.
- X ticks every N bp with --tick-every (applies to BOTH heatmap and profile, respects --xlim-bp).
- Profile mean ± SD/SEM; export TSV with bp, mean, sd, N.
- Arial font with pdf.fonttype=42 for Illustrator-editable text.
"""

import argparse, os, sys
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter

VERSION = "orient_footprint.py v1.6 (2025-08-22)"

# ---------- Utilities ----------

def set_threads(n_threads: int):
    if not n_threads or n_threads <= 0:
        return
    os.environ['OMP_NUM_THREADS'] = str(n_threads)
    os.environ['OPENBLAS_NUM_THREADS'] = str(n_threads)
    os.environ['MKL_NUM_THREADS'] = str(n_threads)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(n_threads)
    os.environ['NUMEXPR_NUM_THREADS'] = str(n_threads)

def load_matrix_tab_robust(tab_path: str, dtype: str = 'float64') -> np.ndarray:
    """
    Read a deepTools --outFileNameMatrix .tab robustly:
    - Skip lines starting with '#'
    - Split by TAB
    - Keep only tokens convertible to float
    - Pad rows to the maximum numeric length with NaN
    """
    rows_numeric = []
    max_len = 0
    with open(tab_path, 'r') as f:
        for ln in f:
            if not ln.strip() or ln[0] == '#':
                continue
            tokens = ln.rstrip('\n').split('\t')
            numeric = []
            for tok in tokens:
                try:
                    numeric.append(float(tok))
                except ValueError:
                    continue
            if not numeric:
                continue
            rows_numeric.append(numeric)
            if len(numeric) > max_len:
                max_len = len(numeric)
    if not rows_numeric:
        raise ValueError("No numeric rows parsed from matrix tab file.")
    mat = np.full((len(rows_numeric), max_len),
                  np.nan,
                  dtype=(np.float32 if dtype == 'float32' else np.float64))
    for i, row in enumerate(rows_numeric):
        L = len(row)
        mat[i, :L] = row
    return mat

def read_rows(regions_path: str):
    rows = []
    with open(regions_path, 'r') as f:
        for ln in f:
            if not ln.strip() or ln.startswith(('#','track','browser')):
                continue
            rows.append(ln.rstrip('\n').split('\t'))
    return rows

def detect_strand_col(rows):
    if not rows:
        return None
    first = rows[0]
    if len(first) >= 6 and first[5] in ('+','-'):
        return 5
    if len(first) >= 4 and first[3] in ('+','-'):
        return 3
    for i, v in enumerate(first):
        if v in ('+','-'):
            return i
    return None

def read_strands(regions_path: str, strand_col: int = None) -> np.ndarray:
    rows = read_rows(regions_path)
    if strand_col is None:
        idx = detect_strand_col(rows)
    else:
        idx = int(strand_col) - 1
    strands = []
    for parts in rows:
        s = parts[idx] if (idx is not None and idx < len(parts)) else '+'
        strands.append(s)
    return np.array(strands)

def orient_matrix_by_strand(mat: np.ndarray, strands: np.ndarray):
    if mat.shape[0] != len(strands):
        raise ValueError("Matrix rows ({}) != regions ({})".format(mat.shape[0], len(strands)))
    oriented = mat.copy()
    neg = (strands == '-')
    oriented[neg] = oriented[neg, ::-1]
    return oriented, int(neg.sum())

# ---------- Plotting ----------

def _setup_fonts(font: str = 'Arial'):
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['font.sans-serif'] = [font]

def plot_heatmap(mat: np.ndarray, out_pdf: str, cmap: str = 'Blues',
                 vmin=None, vmax=None, xlabel='Distance (bp)', ylabel='Motifs',
                 font='Arial', tick_every=None, binsize=1, dpi=600,
                 xlim_bp=None, row_range=None):
    _setup_fonts(font)

    # Apply row-range cropping for display
    if row_range is not None:
        rs, re = row_range
        rs = max(0, rs)
        re = min(mat.shape[0], re)
        mat = mat[rs:re, :]

    # Apply xlim cropping for display
    left_bp_display = None
    if xlim_bp is not None:
        n_bins_full = mat.shape[1]
        half_bp_full = (n_bins_full * binsize) // 2
        left_bp, right_bp = xlim_bp
        left_col  = int(np.clip((left_bp  + half_bp_full) / binsize, 0, n_bins_full-1))
        right_col = int(np.clip((right_bp + half_bp_full) / binsize, 0, n_bins_full-1))
        if right_col <= left_col:
            right_col = min(n_bins_full-1, left_col+1)
        mat = mat[:, left_col:right_col+1]
        left_bp_display = left_bp  # remember for tick mapping

    fig, ax = plt.subplots(figsize=(3.2, 9.5))
    im = ax.imshow(mat, aspect='auto', interpolation='nearest', cmap=cmap,
                   vmin=vmin, vmax=vmax, origin='upper')
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    # Ticks (respect xlim)
    n_bins = mat.shape[1]
    if tick_every:
        if xlim_bp is not None and left_bp_display is not None:
            # Visible window: [left_bp, right_bp], columns run left->right with step = binsize
            right_bp_display = left_bp_display + (n_bins - 1) * binsize
            ticks = np.arange(left_bp_display, right_bp_display + 1, tick_every)
            tick_pos = ((ticks - left_bp_display) / binsize).astype(int)
            tick_pos = np.clip(tick_pos, 0, n_bins - 1)
        else:
            half_bp = (n_bins * binsize) // 2
            ticks = np.arange(-half_bp, half_bp + 1, tick_every)
            tick_pos = ((ticks + half_bp) / binsize).astype(int)
            tick_pos = np.clip(tick_pos, 0, n_bins - 1)
        ax.set_xticks(tick_pos)
        ax.set_xticklabels([str(t) for t in ticks])

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Signal')
    fig.tight_layout()
    fig.savefig(out_pdf, dpi=dpi, bbox_inches='tight')
    try:
        fig.savefig(out_pdf.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    except Exception:
        pass
    plt.close(fig)

def plot_profile(mat: np.ndarray, out_pdf: str, binsize=1, xlabel='Distance (bp)',
                 title=None, font='Arial', error='sd',
                 xlim_bp=None, ylim_profile=None, tick_every=None):
    _setup_fonts(font)

    mean = np.nanmean(mat, axis=0)
    n = np.sum(~np.isnan(mat), axis=0)
    sd = np.nanstd(mat, axis=0, ddof=1)

    x = np.arange(mat.shape[1]) * binsize
    x = x - np.mean(x)

    # Crop x-range for display (and for exported TSV to match the plot)
    if xlim_bp is not None:
        half_bp = int(round(np.max(np.abs(x))))
        left_bp, right_bp = xlim_bp
        left_col  = int(np.clip((left_bp  + half_bp) / binsize, 0, len(x)-1))
        right_col = int(np.clip((right_bp + half_bp) / binsize, 0, len(x)-1))
        if right_col <= left_col:
            right_col = min(len(x)-1, left_col+1)
        sl = slice(left_col, right_col+1)
        x = x[sl]; mean = mean[sl]; sd = sd[sl]; n = n[sl]

    fig, ax = plt.subplots(figsize=(3.2, 2.6))
    ax.plot(x, mean, lw=2)
    if error == 'sd':
        spread = sd
    elif error == 'sem':
        spread = sd / np.sqrt(np.maximum(n, 1))
    else:
        spread = None
    if spread is not None:
        ax.fill_between(x, mean - spread, mean + spread, alpha=0.3, linewidth=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(f'Signal (mean ± {error.upper()})')
    if title:
        ax.set_title(title)
    if ylim_profile is not None:
        ax.set_ylim(ylim_profile[0], ylim_profile[1])

    # Force ticks every N bp, respecting any crop
    if tick_every:
        xmin, xmax = int(round(x.min())), int(round(x.max()))
        ticks = np.arange(xmin, xmax + 1, tick_every)
        # locator expects positions in data coords (bp), not index
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_major_formatter(FixedFormatter([str(t) for t in ticks]))

    fig.tight_layout()
    fig.savefig(out_pdf, dpi=600, bbox_inches='tight')
    plt.close(fig)

    df = pd.DataFrame({'bp': x, 'mean': mean, 'sd': sd, 'N': n})
    return df

# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--matrix-tab', required=True, help='computeMatrix tab file')
    ap.add_argument('--regions', required=True, help='BED/TSV with strand')
    ap.add_argument('--strand-col', type=int, default=None, help='1-based strand column; auto-detected if omitted')
    ap.add_argument('--out-prefix', required=True)
    ap.add_argument('--font', default='Arial')
    ap.add_argument('--binsize', type=int, default=1)
    ap.add_argument('--xlabel', default='Distance (bp)')
    ap.add_argument('--ylabel', default='Motifs')
    ap.add_argument('--title', default=None)
    ap.add_argument('--cmap', default='Blues')
    ap.add_argument('--tick-every', type=int, default=None, help='X tick step in bp (e.g., 250)')
    ap.add_argument('--error', choices=['sd','sem','none'], default='sd')
    ap.add_argument('--threads', type=int, default=None, help='CPU threads to use (sets OMP/MKL/BLAS env vars)')
    ap.add_argument('--dtype', choices=['float64','float32'], default='float64', help='Matrix dtype in memory')
    ap.add_argument('--sort-by-center', choices=['none','mean','max','median'], default='none',
                    help='Sort rows by signal near the center (choose statistic).')
    ap.add_argument('--center-window-bp', type=int, default=1,
                    help='Half-window around center in bp for sorting (e.g., 50 uses [-50,+50]).')
    ap.add_argument('--ascending', action='store_true',
                    help='Ascending sort (default is descending).')
    # Axis limiting & row controls
    ap.add_argument('--xlim-bp', nargs=2, type=int, default=None,
                    help='Crop plotted window to [left right] bp around center.')
    ap.add_argument('--ylim-profile', nargs=2, type=float, default=None,
                    help='Profile y-axis limits [ymin ymax].')
    ap.add_argument('--row-range', nargs=2, type=int, default=None,
                    help='Show only rows [start end) after sorting/orientation.')
    ap.add_argument('--version', action='store_true', help='Print version and exit.')
    args = ap.parse_args()

    if args.version:
        print(VERSION)
        sys.exit(0)

    print(VERSION, file=sys.stderr)
    set_threads(args.threads)

    mat = load_matrix_tab_robust(args.matrix_tab, dtype=args.dtype)
    strands = read_strands(args.regions, args.strand_col)

    oriented, nneg = orient_matrix_by_strand(mat, strands)

    # Optional sorting by center signal
    if args.sort_by_center != 'none':
        n_bins = oriented.shape[1]
        center_idx = n_bins // 2
        hw_bins = max(0, int(round(args.center_window_bp / max(args.binsize, 1))))
        lo = max(0, center_idx - hw_bins)
        hi = min(n_bins, center_idx + hw_bins + 1)
        window = oriented[:, lo:hi]
        if args.sort_by_center == 'mean':
            score = np.nanmean(window, axis=1)
        elif args.sort_by_center == 'max':
            score = np.nanmax(window, axis=1)
        elif args.sort_by_center == 'median':
            score = np.nanmedian(window, axis=1)
        else:
            score = np.nanmean(window, axis=1)
        order = np.argsort(score)
        if not args.ascending:
            order = order[::-1]
        oriented = oriented[order, :]

    np.save(args.out_prefix + '.oriented.npy', oriented)

    # Plot heatmap (auto vmin/vmax like plotHeatmap defaults)
    # Plot heatmap (auto vmin/vmax like plotHeatmap defaults)
    plot_heatmap(oriented, args.out_prefix + '.heatmap.pdf', cmap=args.cmap,
                 xlabel=args.xlabel, ylabel=args.ylabel, font=args.font,
                 tick_every=args.tick_every, binsize=args.binsize,
                 xlim_bp=args.xlim_bp, row_range=args.row_range)


    # Plot profile and export numbers
    prof = plot_profile(oriented, args.out_prefix + '.profile.pdf',
                        binsize=args.binsize, xlabel=args.xlabel, title=args.title,
                        font=args.font, error=args.error,
                        xlim_bp=args.xlim_bp, ylim_profile=args.ylim_profile,
                        tick_every=args.tick_every)
    prof.to_csv(args.out_prefix + '.profile.tsv', sep='\t', index=False)

    print("Flipped {} reverse-strand rows out of {} total.".format(nneg, len(strands)))
    print("Wrote: {}.heatmap.pdf/.png, {}.profile.pdf, {}.profile.tsv, {}.oriented.npy".format(
        args.out_prefix, args.out_prefix, args.out_prefix, args.out_prefix))

if __name__ == '__main__':
    main()


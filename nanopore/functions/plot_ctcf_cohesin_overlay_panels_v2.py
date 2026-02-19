#!/usr/bin/env python3
"""plot_ctcf_cohesin_overlay_panels.py

Make a multi-panel figure where each panel overlays:
  - a reference profile (typically CTCF CUT&RUN) in black
  - one ChIP/CUT&RUN track in a specified color

Inputs are deepTools computeMatrix --outFileNameMatrix .tab files.
Orientation is strand-aware using the provided regions BED/TSV (must contain +/−).

Typical workflow
----------------
1) Generate one matrix TAB per bigWig (same regions + same window/binSize):

   computeMatrix reference-point \
     -S TRACK.bw -R regions.bed --referencePoint center \
     -b 500 -a 500 --binSize 1 -p 12 \
     -o TRACK.mat.gz --outFileNameMatrix TRACK.tab

2) Plot:

   python plot_ctcf_cohesin_overlay_panels.py \
     --regions regions.bed \
     --ref-tab CTCF.tab \
     --track NIPBL:orangered:HA_NIPBL.tab \
     --track PDS5A:blue:PDS5A.tab \
     --track Rad21:deepskyblue:Rad21.tab \
     --out MNase_like_overlay.pdf

Scaling
-------
No twin y-axes are used. By default, each colored track is scaled so its peak
matches the reference peak (scale=refmax). This preserves shapes and allows
clean overlays.

Optional error shading
----------------------
You can shade mean ± SD across regions for the reference, the track, or both:

   --error sd --error-which both --error-alpha 0.25

NOTE: If you also use --smooth, the SD is computed per-bin across rows and then
smoothed for visualization.

"""

import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Reuse robust loaders from orient_footprint.py (same folder)
from orient_footprint import (
    load_matrix_tab_robust,
    read_strands,
    orient_matrix_by_strand,
)


def _setup_fonts(font: str = 'Arial'):
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['font.sans-serif'] = [font]


def _moving_average(y: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return y
    win = int(win)
    k = np.ones(win, dtype=float) / win
    # pad with edge values to avoid shrinkage
    ypad = np.pad(y, (win//2, win-1-win//2), mode='edge')
    return np.convolve(ypad, k, mode='valid')


def mean_profile_from_tab(tab_path: str,
                          regions_path: str,
                          binsize: int = 1,
                          dtype: str = 'float32',
                          xlim_bp=None,
                          smooth: int = 1,
                          orient_by_strand: bool = False,
                          return_sd: bool = False):
    mat = load_matrix_tab_robust(tab_path, dtype=dtype)

    # IMPORTANT:
    # If your regions file is BED6 (6th column is strand), deepTools computeMatrix
    # already orients rows on the minus strand.
    # In that common case, DO NOT flip again here.
    if orient_by_strand:
        strands = read_strands(regions_path, strand_col=None)
        mat, _ = orient_matrix_by_strand(mat, strands)

    mean = np.nanmean(mat, axis=0)
    sd = np.nanstd(mat, axis=0) if return_sd else None

    x = np.arange(mat.shape[1], dtype=float) * binsize
    x = x - np.mean(x)

    if xlim_bp is not None:
        left, right = xlim_bp
        m = (x >= left) & (x <= right)
        x = x[m]
        mean = mean[m]
        if sd is not None:
            sd = sd[m]

    mean = _moving_average(mean, smooth)
    if sd is not None:
        sd = _moving_average(sd, smooth)

    return x, mean, sd


def parse_track_spec(s: str):
    """Parse a --track spec.

    Supported:
      PANEL|AXIS:COLOR:TABFILE
      PANEL|AXIS:COLOR:TABFILE:YMIN:YMAX
      PANEL|AXIS:COLOR:TABFILE:YMIN:YMAX:TICK1,TICK2,...

    If "|" is not provided, PANEL == AXIS.
    """
    parts = s.split(':')
    if len(parts) not in (3, 5, 6):
        raise ValueError(
            "--track must be PANEL|AXIS:COLOR:TABFILE"
            " [or add :YMIN:YMAX[:TICKS]]"
        )

    label_field, color, tab = parts[0], parts[1], parts[2]
    if '|' in label_field:
        panel_label, axis_label = label_field.split('|', 1)
    else:
        panel_label, axis_label = label_field, label_field

    ylim = None
    ticks = None
    if len(parts) in (5, 6):
        ylim = (float(parts[3]), float(parts[4]))
    if len(parts) == 6:
        ticks = [float(x) for x in parts[5].split(',') if x != '']

    return panel_label, axis_label, color, tab, ylim, ticks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--regions', required=True, help='BED/TSV with strand; must match computeMatrix region order')
    ap.add_argument('--ref-tab', required=True, help='Reference computeMatrix TAB (plotted in black)')
    ap.add_argument('--track', action='append', required=True,
                    help='Track spec: LABEL:COLOR:TABFILE (repeat for multiple panels)')
    ap.add_argument('--out', required=True, help='Output PDF (a PNG will also be written)')

    ap.add_argument('--binsize', type=int, default=1)
    ap.add_argument('--xlim-bp', nargs=2, type=int, default=[-500, 500])
    ap.add_argument('--tick-every', type=int, default=200)
    ap.add_argument('--smooth', type=int, default=1)
    ap.add_argument('--font', default='Arial')

    ap.add_argument('--orient-by-strand', action='store_true',
                    help='Flip minus-strand rows using the strand column in --regions. '
                         'Use ONLY if your computeMatrix was generated from regions without strand '
                         '(i.e., computeMatrix did NOT already orient the matrix).')

    ap.add_argument('--mirror-x', action='store_true',
                    help='Mirror profiles left<->right (swap -bp and +bp). Useful if you want to '
                         'display the opposite biological orientation (e.g., N→C vs C→N) after '
                         'computeMatrix strand-orientation.')

    ap.add_argument('--scale', choices=['none', 'max', 'refmax'], default='refmax',
                    help='Scale colored tracks (single-axis mode only): none=raw, max=each to max=1, refmax=match reference peak')
    ap.add_argument('--ylim', nargs=2, type=float, default=None,
                    help='Shared y-limits (single-axis mode only)')

    ap.add_argument('--ncols', type=int, default=3)
    ap.add_argument('--letters', default=None,
                    help='Optional comma-separated panel letters (e.g. I,J,K,L,M,N)')
    ap.add_argument('--ylabel', default='Signal (scaled)')
    ap.add_argument('--xlabel', default='bp from the CTCF motif center')

    # Paper-like dual-axis mode (matches the reference figure):
    # left axis = reference (CTCF), right axis = factor.
    ap.add_argument('--twin-y', action='store_true',
                    help='Use a right y-axis per panel (paper-like). Ignores --scale/--ylim for tracks.')
    ap.add_argument('--ref-ylabel', default='CTCF CUT&RUN (RPM)')
    ap.add_argument('--ref-ylim', nargs=2, type=float, default=[0.0, 2.0])
    ap.add_argument('--ref-yticks', nargs='*', type=float, default=None)
    ap.add_argument('--y2-label-template', default='ChIP {label} (RPKM)')
    ap.add_argument('--vline0', action='store_true', help='Draw a vertical line at 0 bp')

    # Error shading (mean ± SD)
    ap.add_argument('--error', choices=['none', 'sd'], default='none',
                    help='Add shaded error around the mean. Currently supports: sd (mean ± SD)')
    ap.add_argument('--error-which', choices=['ref', 'track', 'both'], default='both',
                    help='Which profile(s) get shaded error')
    ap.add_argument('--error-alpha', type=float, default=0.25,
                    help='Alpha for shaded error region')

    args = ap.parse_args()

    _setup_fonts(args.font)

    want_ref_err = (args.error != 'none') and (args.error_which in ('ref', 'both'))
    x, ref, ref_err = mean_profile_from_tab(
        args.ref_tab, args.regions,
        binsize=args.binsize,
        dtype='float32',
        xlim_bp=args.xlim_bp,
        smooth=args.smooth,
        orient_by_strand=args.orient_by_strand,
        return_sd=want_ref_err
    )

    if args.mirror_x:
        ref = ref[::-1]
        if ref_err is not None:
            ref_err = ref_err[::-1]

    ref_peak = float(np.nanmax(ref)) if np.isfinite(ref).any() else 1.0

    tracks = []
    for spec in args.track:
        panel_label, axis_label, color, tab, y2_ylim, y2_ticks = parse_track_spec(spec)

        want_track_err = (args.error != 'none') and (args.error_which in ('track', 'both'))
        xt, y, y_err = mean_profile_from_tab(
            tab, args.regions,
            binsize=args.binsize,
            dtype='float32',
            xlim_bp=args.xlim_bp,
            smooth=args.smooth,
            orient_by_strand=args.orient_by_strand,
            return_sd=want_track_err
        )

        if args.mirror_x:
            y = y[::-1]
            if y_err is not None:
                y_err = y_err[::-1]

        if len(xt) != len(x) or np.nanmax(np.abs(xt - x)) > 1e-6:
            raise RuntimeError(f"X grids do not match between {args.ref_tab} and {tab}. Ensure same window/binSize.")

        # Scaling only applies to the single-axis overlay mode.
        if not args.twin_y:
            if args.scale == 'max':
                m = float(np.nanmax(y)) if np.isfinite(y).any() else 0.0
                if m > 0:
                    y = y / m
                    if y_err is not None:
                        y_err = y_err / m
            elif args.scale == 'refmax':
                m = float(np.nanmax(y)) if np.isfinite(y).any() else 0.0
                if m > 0:
                    s = ref_peak / m
                    y = y * s
                    if y_err is not None:
                        y_err = y_err * s

        tracks.append((panel_label, axis_label, color, y, y_err, y2_ylim, y2_ticks))

    n = len(tracks)
    ncols = max(1, int(args.ncols))
    nrows = (n + ncols - 1) // ncols

    # The reference figure shows x/y tick labels on every panel, so don't share axes.
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.6 * nrows), sharex=False, sharey=False)
    axes = np.array(axes).reshape(-1)

    letters = None
    if args.letters:
        letters = [t.strip() for t in args.letters.split(',') if t.strip()]

    for i, (ax, (panel_label, axis_label, color, y, y_err, y2_ylim, y2_ticks)) in enumerate(zip(axes, tracks)):
        # reference (always on the left axis)
        ax.plot(x, ref, color='black', lw=2, zorder=3)
        if args.error != 'none' and args.error_which in ('ref', 'both') and ref_err is not None:
            ax.fill_between(x, ref - ref_err, ref + ref_err, color='black',
                            alpha=args.error_alpha, linewidth=0, zorder=1)

        ax.set_xlim(args.xlim_bp[0], args.xlim_bp[1])

        if args.twin_y:
            # Match the paper figure: fixed left axis for CTCF + per-panel right axis for factor.
            ax.set_ylim(args.ref_ylim[0], args.ref_ylim[1])
            if args.ref_yticks is not None and len(args.ref_yticks) > 0:
                ax.set_yticks(args.ref_yticks)
            else:
                if abs(args.ref_ylim[0] - 0.0) < 1e-9 and abs(args.ref_ylim[1] - 2.0) < 1e-9:
                    ax.set_yticks([0.0, 0.5, 1.0, 1.5, 2.0])

            ax2 = ax.twinx()
            ax2.plot(x, y, color=color, lw=2, zorder=2)
            if args.error != 'none' and args.error_which in ('track', 'both') and y_err is not None:
                ax2.fill_between(x, y - y_err, y + y_err, color=color,
                                 alpha=args.error_alpha, linewidth=0, zorder=1)

            if y2_ylim is not None:
                ax2.set_ylim(y2_ylim[0], y2_ylim[1])
            if y2_ticks is not None:
                ax2.set_yticks(y2_ticks)

            ax2.tick_params(axis='y', colors=color, direction='out', length=4, width=1, labelsize=14)
            ax2.spines['right'].set_color(color)
            ax2.yaxis.label.set_color(color)
            ax2.set_ylabel(args.y2_label_template.format(label=axis_label), fontsize=18)

            ax.tick_params(axis='y', colors='black', labelsize=14)
            ax.tick_params(axis='x', labelsize=14)
        else:
            # legacy: single-axis overlay
            ax.plot(x, y, color=color, lw=2, zorder=2)
            if args.error != 'none' and args.error_which in ('track', 'both') and y_err is not None:
                ax.fill_between(x, y - y_err, y + y_err, color=color,
                                alpha=args.error_alpha, linewidth=0, zorder=1)

            if args.ylim is not None:
                ax.set_ylim(args.ylim[0], args.ylim[1])
            ax.tick_params(direction='out', length=4, width=1)

        if args.vline0:
            ax.axvline(0, color='k', lw=1, alpha=0.35, zorder=1)

        # panel label
        ax.text(0.03, 0.92, panel_label, transform=ax.transAxes,
                ha='left', va='top', color=color, fontsize=22, fontweight='bold')

        if letters and i < len(letters):
            ax.text(-0.18, 1.05, letters[i], transform=ax.transAxes,
                    ha='left', va='bottom', color='black', fontsize=44, fontweight='bold')

        # x ticks
        if args.tick_every:
            ticks = np.arange(args.xlim_bp[0], args.xlim_bp[1] + 1, args.tick_every)
            ax.set_xticks(ticks)

    # hide unused axes
    for ax in axes[n:]:
        ax.axis('off')

    # Matplotlib <3.4 compatibility: no fig.supxlabel/supylabel.
    fig.subplots_adjust(left=0.10, right=0.92, bottom=0.10, top=0.95, wspace=0.38, hspace=0.32)
    fig.text(0.5, 0.04, args.xlabel, ha='center', va='center', fontsize=24)
    if args.twin_y:
        fig.text(0.03, 0.5, args.ref_ylabel, ha='center', va='center', rotation=90, fontsize=24)
    else:
        fig.text(0.03, 0.5, args.ylabel, ha='center', va='center', rotation=90, fontsize=18)

    fig.savefig(args.out, dpi=600, bbox_inches='tight')
    if args.out.lower().endswith('.pdf'):
        try:
            fig.savefig(args.out[:-4] + '.png', dpi=300, bbox_inches='tight')
        except Exception:
            pass
    plt.close(fig)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Rows = reads; columns = fixed bp bins from -span..+span (relative to motif center).

Heatmap (bottom):
  - gray  = no GpC / not assigned
  - white = M (methylated)
  - black = U (unmethylated)   [default black; change with --black-is]
  - --lengthen-bp visually extends each observed CpG box left/right in the HEATMAP ONLY.
  - --row-sep draws horizontal separator lines between reads.

Profile (top):
  - Fraction unmethylated (U) mean per bin, with markers + ±SD error bars.
  - The connecting line draws only across bins that have signal (count>0),
    connecting the signal bins themselves (even if not adjacent).

Reads are filtered to SPAN the motif (≥1 CpG ≤ -core-span AND ≥ +core-span).
Optional ordering: --order-by center_access with --center-width (bp) to group rows by
  fully accessible at center, completely inaccessible, partial, then no-center-calls.

Optionally annotate X with measured CpG bins:
  --label-cpgs [--label-every N --label-fontsize 8 --label-rotate 90]
  
  To generate Fig.3E: python zf_region_raster.py \
  -i calls_expanded_CGwindow.tsv \
  -m fimo_specific_ext65_nonoverlap.tsv \
  --calls-format auto \
  --region chr16:8792017-8792147 \
  --assign-window 0 --analysis-span 200 \
  --span 18 --core-span 0 --bin 1 \
  --label-cpgs --lengthen-bp 2 --x-axis both \
  --black-is U \
  --order-by ctcf_categories --ctcf-dyn-min-k 2 \
  --nuc-flag-up --nuc-flag-down --nuc-ndr-lo 20 --nuc-ndr-hi 60 --nuc-flanks-when-no-core naked \
  --row-sep --row-sep-color 0.85 --row-sep-width 0.6 \
  --report --show-category-bar --row-category-stripe \
  --export chr16_8792017-8792147_.pdf \
  --category-bar-width-pct 14 \
  --row-category-stripe-width-pct 20 \
  --category-label-fontsize 23 \
  --category-label-min-frac 0.04 \
  --row-px 40 --col-px 10 --no-profile-scatter --marker circle --circle-scale 4
"""

import argparse, re, sys
from pathlib import Path
from typing import Tuple, Optional, List, Sequence

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

# ── Matplotlib defaults ──────────────────────────────────────────────────────
mpl.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.family'] = 'Arial'


# ────────────────────────── helpers ──────────────────────────
from collections import Counter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def _parse_category_colors(s: str):
    # defaults + user overrides
    cmap = {
        'ctcf_static':   '#0057B8',  # blue
        'ctcf_dynamic':  '#D32F2F',  # red
        'nucleosomal':   '#7f7f7f',  # dark gray
        'naked':         '#c0c0c0',  # light gray
    }
    try:
        for tok in s.split(','):
            if not tok.strip(): continue
            k, v = tok.split(':', 1)
            cmap[k.strip()] = v.strip()
    except Exception:
        pass
    return cmap

from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import ListedColormap
from collections import Counter

from collections import Counter
import numpy as np
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.patheffects as pe

from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from collections import Counter
import numpy as np

def _add_category_bar(
    ax_heat,
    categories,
    colors,
    *,
    show_stripe=False,
    # labeling
    label_min_frac=0.06,
    pct_fontsize=10,
    # sizing (percentages relative to the heatmap width)
    bar_width_pct=10.0,
    stripe_width_pct=2.0,
):
    """
    Draw a stacked fraction bar to the RIGHT of the heatmap (aligned to row groups)
    and an optional per-row category stripe INSIDE the heatmap at the right edge.
    categories: list[str] aligned to final read_ids (values in colors.keys()).
    """
    if not categories:
        return

    # consistent order with ctcf_categories row ordering
    order = [k for k in ('ctcf_static', 'ctcf_dynamic', 'nucleosomal', 'naked') if k in colors]

    counts = Counter([c for c in categories if c in colors])
    total  = float(sum(counts.values())) or 1.0

    # --- stacked bar: appended axis so it stays inside the figure ---
    divider = make_axes_locatable(ax_heat)
    ax_bar = divider.append_axes(
        'right',
        size=f'{bar_width_pct:.1f}%',
        pad=0.15,         # gap between heatmap and bar (inches)
    )

    start = 0.0
    for k in order:
        frac = counts.get(k, 0) / total
        if frac <= 0:
            continue
        ax_bar.bar(0.5, frac, bottom=start, width=0.9,
                   color=colors[k], edgecolor='none')
        if frac >= label_min_frac:
            y = start + frac/2.0
            ax_bar.text(0.5, y, f"{frac*100:.0f}%",
                        ha='center', va='center', fontsize=pct_fontsize)
        start += frac

    ax_bar.set_xlim(0, 1)
    ax_bar.set_ylim(0, 1)
    ax_bar.axis('off')
    ax_bar.invert_yaxis()   # <-- key: top chunk corresponds to top rows

    # --- per-row stripe: stays INSIDE the heatmap ---
    if show_stripe:
        ax_stripe = inset_axes(
            ax_heat,
            width=f"{stripe_width_pct:.1f}%",
            height="100%",
            bbox_to_anchor=(0.995, 0, 0.005, 1.0),  # inside the right edge
            bbox_transform=ax_heat.transAxes,
            loc='lower left',
            borderpad=0
        )
        idx_map = {k: i for i, k in enumerate(order)}
        data = np.array([idx_map.get(c, 0) for c in categories], dtype=int).reshape(-1, 1)
        cmap = ListedColormap([colors[k] for k in order])
        ax_stripe.imshow(
            data, aspect='auto', cmap=cmap,
            vmin=0, vmax=len(order)-1,
            origin='upper', interpolation='nearest'
        )
        ax_stripe.set_xticks([])
        ax_stripe.set_yticks([])
        for sp in ax_stripe.spines.values():
            sp.set_visible(False)

def _map_zf_from_relpos(rel_pos: pd.Series) -> pd.Series:
    """CTCF ZF index from strand-aware rel_pos: 1..11 are core bins (3bp each)."""
    return (6 + np.floor_divide(rel_pos.astype(int), 3)).astype(int)

def _nucleosome_flags_per_read(sub: pd.DataFrame, lo: int, hi: int, flag_up: bool, flag_dn: bool) -> pd.Series:
    """
    Return boolean per read_id: nucleosome-like if (requested side has ≥1 CpG AND all are U).
    Uses strand-aware rel_pos already in `sub`.
    """
    if not (flag_up or flag_dn) or sub.empty:
        return pd.Series(False, index=sub['read_id'].astype(str).unique())

    df = sub[sub['status'].isin(['M','U'])].copy()
    df['read_id'] = df['read_id'].astype(str)

    up_mask = df['rel_pos'].between(-hi, -lo) if flag_up else pd.Series(False, index=df.index)
    dn_mask = df['rel_pos'].between(lo, hi)   if flag_dn else pd.Series(False, index=df.index)

    nuc_up = pd.Series(False, index=df['read_id'].unique())
    nuc_dn = pd.Series(False, index=df['read_id'].unique())

    if flag_up:
        up = df[up_mask]
        if not up.empty:
            g = up.groupby('read_id')['status'].apply(lambda s: (len(s) > 0) and np.all(s.values == 'U'))
            nuc_up.loc[g.index] = g.values
    if flag_dn:
        dn = df[dn_mask]
        if not dn.empty:
            g = dn.groupby('read_id')['status'].apply(lambda s: (len(s) > 0) and np.all(s.values == 'U'))
            nuc_dn.loc[g.index] = g.values

    # nucleosome-like if either requested side is nucleosomal
    out = (nuc_up.reindex_like(nuc_up).fillna(False) | nuc_dn.reindex_like(nuc_up).fillna(False))
    out.name = 'is_nuc'
    return out



    
def parse_region(s: str) -> Tuple[str, int, int]:
    m = re.match(r'^(\S+):(\d+)-(\d+)$', s.replace(',', ''))
    if not m:
        raise ValueError("Bad --region format. Use chr:start-end")
    chrom, a, b = m.group(1), int(m.group(2)), int(m.group(3))
    if a > b: a, b = b, a
    return chrom, a, b


def read_motif_bed(path: str) -> pd.DataFrame:
    m = pd.read_csv(path, sep='\t', header=None, engine='c',
                    usecols=[0, 1, 2, 3],
                    dtype={0:'category',1:'int64',2:'int64',3:'category'})
    m.columns = ['chr_motif','start_motif','end_motif','strand_motif']
    # optional specific_pos (col 5)
    try:
        sp = pd.read_csv(path, sep='\t', header=None, engine='c', usecols=[4])
        sp = pd.to_numeric(sp.iloc[:,0], errors='coerce')
        m['specific_pos'] = (sp if len(sp)==len(m) else pd.Series([pd.NA]*len(m))).astype('Int64')
    except Exception:
        m['specific_pos'] = pd.Series([pd.NA]*len(m), dtype='Int64')
    center = (m['start_motif'] + m['end_motif']) // 2
    m['specific_pos'] = m['specific_pos'].fillna(center).astype('int64')
    return m


def detect_calls_format(path: str) -> str:
    head = pd.read_csv(path, sep='\t', header=None, nrows=3, engine='c')
    ncol = head.shape[1]
    if ncol >= 16: return 'intersect'
    if ncol == 10: return 'wide'
    return 'wide' if ncol <= 12 else 'intersect'


def find_target_motif(motif_df: pd.DataFrame, chrom: str, start: int, end: int,
                      idx_override: Optional[int]) -> pd.Series:
    cand = motif_df[(motif_df['chr_motif'].astype(str)==str(chrom)) &
                    (motif_df['start_motif']<=end) &
                    (motif_df['end_motif']>=start)].reset_index(drop=True)
    if cand.empty:
        raise RuntimeError("No motif overlaps the region.")
    if idx_override is not None:
        if not (0 <= idx_override < len(cand)):
            raise IndexError(f"--motif-index {idx_override} out of range [0,{len(cand)-1}]")
        return cand.iloc[idx_override]
    mid = (start+end)//2
    i = (cand['specific_pos']-mid).abs().idxmin()
    return cand.loc[i]


def _order_by_ctcf_categories(
    sub: pd.DataFrame,
    read_ids_in_base_order: List[str],
    *,
    dyn_min_k: int,
    nuc_lo: int,
    nuc_hi: int,
    flag_up: bool,
    flag_dn: bool,
    nuc_flanks_when_no_core: str = "naked",   # 'naked' (default) or 'nucleosomal'
):
    """
    Classify each read into one of four groups, then return:
      order_idx, breaks, counts, cats_df

    Categories:
      0 'ctcf_static'   : CTCF-bound AND among measurable ZFs in [1..11],
                          measurable_count >= dyn_min_k AND all measurable are bound
      1 'ctcf_dynamic'  : CTCF-bound but NOT static (includes uninformative with < dyn_min_k measurable ZFs)
      2 'nucleosomal'   : Nucleosome-like by flank window rule. Overrides others EXCEPT
                          the special case: if no core ZF (1..11) is bound and
                          nuc_flanks_when_no_core == 'naked', then classify as 'naked' instead.
      3 'naked'         : No core ZF (1..11) bound
    Ordering: static → dynamic → nucleosomal → naked (stable by base order within groups).

    Returns:
      order_idx : ndarray of indices into read_ids_in_base_order (new row order)
      breaks    : ndarray [b1, b2, b3] cumulative boundaries between groups
      counts    : tuple (n_static, n_dynamic, n_nucleosomal, n_naked)
      cats_df   : DataFrame with columns ['read_id','category'] (for all reads)
    """
    if sub.empty or len(read_ids_in_base_order) == 0:
        n = len(read_ids_in_base_order)
        return (
            np.arange(n, dtype=int),
            np.array([0, 0, 0], dtype=int),
            (0, 0, 0, n),
            pd.DataFrame({"read_id": read_ids_in_base_order, "category": ["naked"] * n}),
        )

    df = sub[sub["status"].isin(["M", "U"])].copy()
    df["read_id"] = df["read_id"].astype(str)

    # Map to ZF indices and keep core 1..11
    df["ZF"] = _map_zf_from_relpos(df["rel_pos"])
    core = df[df["ZF"].between(1, 11)]

    # Build a stable rank from the base order
    base_rank = {rid: i for i, rid in enumerate(read_ids_in_base_order)}

    # Per-(read,ZF): bound if ANY U observed at that ZF
    if core.empty:
        per_zf = pd.DataFrame(columns=["read_id", "ZF", "bound"])
    else:
        per_zf = (
            core.groupby(["read_id", "ZF"])["status"]
            .apply(lambda s: int("U" in set(s)))
            .rename("bound")
            .reset_index()
        )

    # Start with all reads known to us
    all_reads = pd.Index(read_ids_in_base_order, dtype="object")

    # CTCF-any: any ZF(1..11) bound?
    if per_zf.empty:
        ctcf_any = pd.Series(0, index=all_reads, name="ctcf_any")  # none bound
    else:
        ctcf_any = (
            per_zf.groupby("read_id")["bound"].max().reindex(all_reads, fill_value=0).rename("ctcf_any")
        )

    # Measurable ZFs in 1..11 (use ALL core ZFs, not 2..11)
    if per_zf.empty:
        z_sum_cnt = pd.DataFrame(
            {"sum": np.zeros(len(all_reads), dtype=int), "count": np.zeros(len(all_reads), dtype=int)},
            index=all_reads,
        )
    else:
        z_all = per_zf[per_zf["ZF"].between(1, 11)]
        z_sum_cnt = (
            z_all.groupby("read_id")["bound"]
            .agg(sum="sum", count="count")
            .reindex(all_reads, fill_value=0)
        )

    # Nucleosome-like flank flags (applied on the windowed positions)
    nuc = _nucleosome_flags_per_read(df, lo=nuc_lo, hi=nuc_hi, flag_up=flag_up, flag_dn=flag_dn)
    nuc = nuc.reindex(all_reads, fill_value=False)

    # Category assignment
    code2name = {0: "ctcf_static", 1: "ctcf_dynamic", 2: "nucleosomal", 3: "naked"}
    cats_code = {}
    extra_key = {}  # used to push informative dynamic (>=dyn_min_k measurable) before uninformative

    for rid in read_ids_in_base_order:
        any_core = int(ctcf_any.get(rid, 0)) == 1
        # Special-case for flanks-only nucleosome look:
        if nuc.get(rid, False):
            if not any_core and nuc_flanks_when_no_core == "naked":
                # looks nucleosomal in flanks, but no core bound → call 'naked'
                cats_code[rid] = 3
                extra_key[rid] = 0
                continue
            else:
                # nucleosomal overrides in all other cases
                cats_code[rid] = 2
                extra_key[rid] = 0
                continue

        if any_core:
            c = int(z_sum_cnt.loc[rid, "count"])
            s = int(z_sum_cnt.loc[rid, "sum"])
            if c >= dyn_min_k and c > 0 and s == c:
                cats_code[rid] = 0  # static across all measurable ZFs
                extra_key[rid] = 0
            else:
                cats_code[rid] = 1  # dynamic (includes uninformative < dyn_min_k)
                extra_key[rid] = 0 if c >= dyn_min_k else 1
        else:
            cats_code[rid] = 3  # naked
            extra_key[rid] = 0

    # Final order by category, then extra_key, then base order (stable)
    order = sorted(read_ids_in_base_order, key=lambda r: (cats_code[r], extra_key[r], base_rank[r]))
    order_idx = np.array([base_rank[r] for r in order], dtype=int)

    # Counts & breaks
    counts = [sum(1 for r in order if cats_code[r] == k) for k in (0, 1, 2, 3)]
    b1 = counts[0]
    b2 = b1 + counts[1]
    b3 = b2 + counts[2]
    breaks = np.array([b1, b2, b3], dtype=int)

    cats_df = pd.DataFrame({"read_id": read_ids_in_base_order,
                            "category": [code2name[cats_code[r]] for r in read_ids_in_base_order]})

    return order_idx, breaks, tuple(counts), cats_df


# ────────────────────────── streaming I/O ──────────────────────────

def stream_intersect_for_motif(path: str, target: pd.Series,
                               left: int, right: int,
                               chunksize: int=2_000_000) -> pd.DataFrame:
    colnames = ['chr_call','start_call','end_call','call_pos','strand_call',
                'read_id','llr_ratio','llr_met','llr_unmet','status',
                'chr_motif','start_motif','end_motif','strand_motif',
                'motif_seq','specific_pos']
    usecols = list(range(16))
    dtypes = {0:'category',1:'int64',2:'int64',3:'int64',4:'category',
              5:'category',6:'float32',7:'float32',8:'float32',9:'category',
              10:'category',11:'int64',12:'int64',13:'category',
              14:'category',15:'int64'}
    out = []
    for chunk in pd.read_csv(path, sep='\t', header=None, engine='c',
                             usecols=usecols, dtype=dtypes, chunksize=chunksize):
        chunk.columns = colnames
        m = ((chunk['chr_motif'].astype(str)==str(target['chr_motif'])) &
             (chunk['start_motif']==int(target['start_motif'])) &
             (chunk['end_motif']==int(target['end_motif'])) &
             (chunk['strand_motif'].astype(str)==str(target['strand_motif'])) &
             (chunk['call_pos']>=left) & (chunk['call_pos']<=right))
        sub = chunk[m]
        if not sub.empty: out.append(sub)
    if not out: return pd.DataFrame(columns=colnames)
    return pd.concat(out, ignore_index=True)


def stream_wide_for_region(path: str, chrom: str, left: int, right: int,
                           chunksize: int=2_000_000) -> pd.DataFrame:
    cols = list(range(10))
    dtypes = {0:'category',1:'int64',2:'int64',3:'int64',4:'category',
              5:'category',6:'float32',7:'float32',8:'float32',9:'category'}
    out = []
    for chunk in pd.read_csv(path, sep='\t', header=None, engine='c',
                             usecols=cols, dtype=dtypes, chunksize=chunksize):
        sub = chunk[(chunk.iloc[:,0].astype(str)==str(chrom)) &
                    (chunk.iloc[:,3]>=left) & (chunk.iloc[:,3]<=right)]
        if not sub.empty: out.append(sub)
    if not out: return pd.DataFrame(columns=cols)
    return pd.concat(out, ignore_index=True)


def map_wide_to_motif(calls_wide: pd.DataFrame, motif_df: pd.DataFrame,
                      chrom: str, assign_window: int) -> pd.DataFrame:
    """
    Assign 10-col 'wide' calls to the nearest motif center on the same chr.

    Behavior change:
      - If assign_window <= 0  → no tolerance limit (assign to the nearest motif;
        nothing is dropped due to distance).
    """
    # No calls → return empty with expected columns
    if calls_wide.empty:
        return pd.DataFrame(columns=[
            'chr_call','start_call','end_call','call_pos','strand_call','read_id',
            'llr_ratio','llr_met','llr_unmet','status',
            'chr_motif','start_motif','end_motif','strand_motif','specific_pos'
        ])

    calls = calls_wide.copy()
    calls.columns = [
        'chr_call','start_call','end_call','call_pos','strand_call',
        'read_id','llr_ratio','llr_met','llr_unmet','status'
    ]

    # Left table: calls on this chromosome, sorted by position
    left = calls[calls['chr_call'].astype(str) == str(chrom)].sort_values('call_pos', kind='mergesort')

    # Right table: motifs on this chr, sorted by specific_pos
    right = motif_df[motif_df['chr_motif'].astype(str) == str(chrom)].copy()
    if right.empty:
        # No motif on this chr → nothing can be assigned
        return pd.DataFrame(columns=list(calls.columns) + [
            'chr_motif','start_motif','end_motif','strand_motif','specific_pos'
        ])
    right = right.sort_values('specific_pos', kind='mergesort')

    # Tolerance: None means "no limit" → assign to nearest motif always
    tol = None if assign_window is None or assign_window <= 0 else assign_window

    merged = pd.merge_asof(
        left, right,
        left_on='call_pos', right_on='specific_pos',
        direction='nearest', tolerance=tol
    )

    # Keep only rows that matched a motif row
    merged = merged.dropna(subset=['specific_pos'])
    return merged


# ────────────────────────── selection & matrix ──────────────────────────

def prepare_calls(sub: pd.DataFrame, target_center: int,
                  plot_span: int, bin_size: int,
                  core_span: int, min_cpg_per_read: int, report: bool):
    """
    Prepare calls for downstream analysis.

    Intended use now: call this with the **analysis half-window** (plot_span),
    not the display window. Later, crop to the display window for plotting only.

    Steps:
      - keep only M/U
      - compute strand-aware rel_pos and rel_bin
      - restrict to ±plot_span (analysis window)
      - enforce spanning (±core_span) and per-read min CpGs (if requested)
    """
    stats = {'calls_in_window': 0, 'reads_in_window': 0,
             'reads_spanning': 0, 'reads_after_min': 0}

    if sub.empty:
        return sub, stats

    # Keep only binary states
    sub = sub[sub['status'].isin(['M', 'U'])]
    if sub.empty:
        return sub, stats

    # Strand-aware relative position to motif center
    # (flip so + strand has decreasing coords upstream → consistent orientation)
    sf = {'+': -1, '-': 1}
    sub = sub.copy()
    sub['rel_pos'] = (sub['call_pos'] - int(target_center)) * sub['strand_motif'].map(sf).fillna(1).astype(int)

    # ANALYSIS window filter (use a large window here; plotting will crop later)
    sub = sub[(sub['rel_pos'] >= -plot_span) & (sub['rel_pos'] <= plot_span)]
    if sub.empty:
        return sub, stats

    # Fixed-size binning on the analysis window
    sub['rel_bin'] = (np.floor_divide(sub['rel_pos'], bin_size) * bin_size).astype(int)
    sub['read_id'] = sub['read_id'].astype(str)

    stats['calls_in_window'] = len(sub)
    stats['reads_in_window'] = sub['read_id'].nunique()

    # Spanning filter (disable by setting core_span=0)
    if core_span > 0:
        g = sub.groupby('read_id')['rel_pos']
        keep = g.apply(lambda s: (s.min() <= -core_span) and (s.max() >= core_span))
        keep_reads = set(keep[keep].index.tolist())
        sub = sub[sub['read_id'].isin(keep_reads)]
    stats['reads_spanning'] = sub['read_id'].nunique()

    # Minimum CpGs per read in the analysis window
    if min_cpg_per_read > 0 and not sub.empty:
        counts = sub.groupby('read_id').size()
        good_reads = set(counts[counts >= min_cpg_per_read].index.tolist())
        sub = sub[sub['read_id'].isin(good_reads)]
    stats['reads_after_min'] = sub['read_id'].nunique()

    if report:
        print(f"[SELECT] calls_in_window={stats['calls_in_window']:,}  "
              f"reads_in_window={stats['reads_in_window']:,}  "
              f"reads_spanning={stats['reads_spanning']:,}  "
              f"reads_after_min={stats['reads_after_min']:,}")
    return sub, stats



def build_matrix_rows_symmetric(sub: pd.DataFrame, status_black: str,
                                span: int, bin_size: int,
                                order_by: str='first_pos'):
    """
    ROWS = reads; COLUMNS = fixed bins from -span..+span step bin_size (symmetric).
    Values: 1 = status_black, 0 = other state, -1 = N/A.
    """
    if sub.empty:
        return np.zeros((0,0), dtype=np.int8), np.array([], dtype=int), []

    bins = np.arange(-span, span + 1, bin_size, dtype=int)
    sb = str(status_black)
    df = sub[['read_id','rel_bin','status']].copy()
    df['val'] = np.where(df['status'] == sb, 1, 0).astype(np.int8)

    if order_by == 'read_id':
        read_ids = sorted(df['read_id'].unique())
    else:
        first_pos = df.groupby('read_id')['rel_bin'].min().sort_values()
        read_ids = list(first_pos.index)

    tbl = df.pivot_table(index='read_id', columns='rel_bin', values='val', aggfunc='max')
    tbl = tbl.reindex(index=read_ids, columns=bins)
    mat = tbl.fillna(-1).to_numpy(dtype=np.int8)
    return mat, bins, read_ids


# ────────────────────────── center-access ordering ──────────────────────────

def reorder_by_center_access(
    mat: np.ndarray, bins: np.ndarray, *, black_is: str, center_width: int
) -> Tuple[np.ndarray, np.ndarray, Tuple[int,int,int,int]]:
    """
    Reorder rows into groups based on CpGs in ±center_width bp:
      fully accessible (all observed == U), completely inaccessible (all observed == M),
      partial (mix), no-center-calls (none observed).
    Uses mat encoding: 1 == status_black; 0 == other; -1 == N/A.
    Accessibility is U, so compare to (1 if black_is=='U' else 0).
    """
    nrow = mat.shape[0]
    if nrow == 0:
        return np.arange(0), np.array([], dtype=int), (0,0,0,0)

    center_mask = (np.abs(bins) <= center_width)
    idx = np.where(center_mask)[0]
    if idx.size == 0:
        return np.arange(nrow), np.array([], dtype=int), (0,0,0,0)

    sub = mat[:, idx]
    obs = (sub >= 0)
    any_obs = obs.any(axis=1)

    acc_val = 1 if black_is == 'U' else 0
    inacc_val = 0 if black_is == 'U' else 1

    all_acc = any_obs & np.where(obs, sub == acc_val, True).all(axis=1)
    all_inacc = any_obs & np.where(obs, sub == inacc_val, True).all(axis=1)
    partial = any_obs & ~(all_acc | all_inacc)
    no_center = ~any_obs

    i_acc      = np.flatnonzero(all_acc)
    i_inacc    = np.flatnonzero(all_inacc)
    i_partial  = np.flatnonzero(partial)
    i_none     = np.flatnonzero(no_center)

    order = np.r_[i_acc, i_inacc, i_partial, i_none]
    b1 = len(i_acc); b2 = b1 + len(i_inacc); b3 = b2 + len(i_partial)
    breaks = np.array([b1, b2, b3], dtype=int)
    counts = (len(i_acc), len(i_inacc), len(i_partial), len(i_none))
    return order, breaks, counts


# ────────────────────────── U-fraction profile (mean ± SD) ───────────────────

def compute_fraction_U_profile(sub: pd.DataFrame, read_ids: List[str], bins: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute per-bin fraction of U across the plotted reads + per-bin SD + per-bin counts.
    Returns (mean, std, counts) aligned to 'bins'.
    """
    if sub.empty or len(read_ids) == 0:
        zero_f = np.zeros_like(bins, dtype=float)
        zero_i = np.zeros_like(bins, dtype=int)
        return zero_f, zero_f, zero_i
    df = sub[['read_id','rel_bin','status']].copy()
    df = df[df['read_id'].isin(read_ids)]
    df['uval'] = (df['status'] == 'U').astype(float)  # 1 for U, 0 for M
    piv = df.pivot_table(index='read_id', columns='rel_bin', values='uval', aggfunc='mean')
    piv = piv.reindex(index=read_ids, columns=bins)
    mean = piv.mean(axis=0, skipna=True).to_numpy(dtype=float)
    std  = piv.std(axis=0, skipna=True, ddof=0).to_numpy(dtype=float)
    cnt  = piv.count(axis=0).to_numpy(dtype=int)
    return mean, std, cnt


# ────────────────────────── visual lengthening of boxes ──────────────────────

def lengthen_boxes(mat: np.ndarray, L_bins: int) -> np.ndarray:
    """
    Visually extend observed states horizontally by up to L_bins on each side,
    filling only N/A cells (-1). Profile is NOT affected (only the heatmap).
    Tie-breaking between two different nearby states: choose the NEAREST;
    if equidistant, prefer LEFT.
    """
    if L_bins <= 0 or mat.size == 0:
        return mat
    nrow, ncol = mat.shape
    out = mat.copy()
    idxs = np.arange(ncol)
    for r in range(nrow):
        row = out[r, :]
        called = (row >= 0)
        if not np.any(called):
            continue

        # nearest called index to the LEFT of each position
        left = np.where(called, idxs, -1).astype(np.int64)
        np.maximum.accumulate(left, out=left)

        # nearest called index to the RIGHT of each position
        right = np.where(called, idxs, ncol).astype(np.int64)
        rev = right[::-1]
        np.minimum.accumulate(rev, out=rev)
        right = rev[::-1]

        # original statuses at called positions
        statuses = np.full(ncol, -2, dtype=np.int8)
        statuses[called] = row[called]

        unk = (row == -1)
        dl = np.where(left >= 0, idxs - left, np.inf)
        dr = np.where(right < ncol, right - idxs, np.inf)
        dmin = np.minimum(dl, dr)

        within = (unk & (dmin <= L_bins))
        choose_right = (dr < dl)  # ties -> left

        sel = np.where(choose_right, right, left)  # index of chosen source
        fill_idx = np.where(within)[0]
        row[fill_idx] = statuses[sel[fill_idx]]
        out[r, :] = row
    return out


# ────────────────────────── plotting (profile + heat) ────────────────────────

def plot_profile_and_heat(
    mat, bins, read_ids, *,
    status_black='U', outpath=None, title=None,
    group_breaks=None, group_counts=None,
    measured_bins=None, label_every=1, label_fontsize=8, label_rotate=90,
    prof_mean=None, prof_std=None, prof_cnt=None,
    show_profile_scatter=True,
    profile_mark_every=1, profile_marker_size=3.0, profile_cap_size=2.0,
    row_px=6.0, col_px=6.0, dpi=96, lengthen_bp=0,
    row_sep=False, row_sep_color='0.75', row_sep_width=0.6,
    category_for_rows=None, category_colors=None,
    show_category_bar=False, show_row_category_stripe=False,
    category_bar_width_pct=10.0,
    row_category_stripe_width_pct=2.0,
    category_label_fontsize=11,
    category_label_min_frac=0.03,
    x_axis='bp',
    marker='square',
    circle_scale=0.9,
):



    """
    TOP: fraction U mean with markers + ±SD at bins where cnt>0.
         Line connects only the signal bins (in index order).
    BOTTOM: per-read heatmap. Row height via --row-px, column width via --col-px.
            If --lengthen-bp > 0, extend observed boxes horizontally in the heatmap only.
            If --row-sep, draw horizontal separators between read rows.
    """
    if mat.size == 0:
        print("No data to plot.")
        return
    nrow, ncol = mat.shape
    orig_mat = mat.copy()

    # Visual lengthening (convert bp to bins from the bin step)
    if len(bins) >= 2:
        step_bp = int(abs(bins[1] - bins[0]))
    else:
        step_bp = 1
    L_bins = int(round(max(0, lengthen_bp) / max(1, step_bp)))
    vis_mat = lengthen_boxes(mat, L_bins) if L_bins > 0 else mat

    # Auto-fit (respect Matplotlib pixel limits)
    MAX_W_PX = 62000
    MAX_H_PX = 62000
    DPI = int(dpi)
    target_px_per_col = float(col_px)
    target_px_per_row = float(row_px)
    px_per_col = min(target_px_per_col, MAX_W_PX / max(1, ncol))
    px_per_row = min(target_px_per_row, MAX_H_PX / max(1, nrow))
    heat_h_px = int(np.clip(np.ceil(nrow * px_per_row), 300, MAX_H_PX - 220))
    prof_h_px = 180
    fig_w_px  = int(np.clip(np.ceil(ncol * px_per_col), 800, MAX_W_PX))
    fig_h_px  = int(np.clip(heat_h_px + prof_h_px, 500, MAX_H_PX))
    fig_w_in, fig_h_in = fig_w_px / DPI, fig_h_px / DPI

    # Map values: -1->0 (grey), 0->1 (white), 1->2 (black)
    cmap = ListedColormap(['0.6', '1.0', '0.0'])
    plot_mat = np.clip(vis_mat, -1, 1).astype(np.int8) + 1

    # Legend labels
    if status_black == 'U':
        black_lbl = 'U (unmethylated)'
        white_lbl = 'M (methylated)'
    else:
        black_lbl = 'M (methylated)'
        white_lbl = 'U (unmethylated)'

    fig, (axp, ax) = plt.subplots(
        2, 1,
        gridspec_kw={'height_ratios': [prof_h_px, heat_h_px]},
        sharex=True, constrained_layout=True,
        figsize=(fig_w_in, fig_h_in), dpi=DPI
    )

    # ── TOP: profile (connect ONLY bins with signal) ─────────────────────────
    if prof_mean is None or prof_std is None:
        prof_mean = np.zeros(ncol, dtype=float)
        prof_std  = np.zeros(ncol, dtype=float)
    if prof_cnt is None:
        prof_cnt = np.zeros(ncol, dtype=int)

    signal_idx = np.flatnonzero(prof_cnt > 0)
    if signal_idx.size >= 2:
        axp.plot(signal_idx, prof_mean[signal_idx], linewidth=1.2)
    
    if show_profile_scatter and signal_idx.size > 0:
        step = max(1, int(profile_mark_every))
        xs = signal_idx[::step]
        ys = prof_mean[signal_idx][::step]
        es = prof_std[signal_idx][::step]
        axp.errorbar(xs, ys, yerr=es, fmt='o', markersize=profile_marker_size,
                     linewidth=0.8, capsize=profile_cap_size, elinewidth=0.7)

    axp.set_xlim(-0.5, ncol - 0.5)
    axp.set_ylim(0.0, 1.0)
    axp.set_ylabel('Fraction U', fontsize=11)
    axp.set_xticks([])

    # Motif center line
    if 0 in set(bins):
        x0 = int(np.where(bins == 0)[0][0])
    else:
        x0 = int(np.argmin(np.abs(bins)))
    axp.axvline(x0, color='red', linestyle='--', linewidth=1)

    # Optional CpG labels on very top
    if measured_bins is not None and len(measured_bins) > 0:
        bin_to_idx = {int(b): i for i, b in enumerate(bins)}
        pairs = [(bin_to_idx[int(b)], int(b)) for b in measured_bins if int(b) in bin_to_idx]
        if pairs:
            idxs_all, bins_all = zip(*pairs)
            idxs2 = list(idxs_all)[::max(1, label_every)]
            labels2 = [f"{int(b)} bp" for b in list(bins_all)[::max(1, label_every)]]
            ax_top = axp.secondary_xaxis('top')
            ax_top.set_xticks(idxs2)
            ax_top.set_xticklabels(labels2, rotation=label_rotate, fontsize=label_fontsize)
            ax_top.tick_params(axis='x', pad=2, length=2)
            ax_top.set_xlabel('Measured CpG positions', fontsize=10)

    # ── BOTTOM: heatmap ──────────────────────────────────────────────────────
     # ── BOTTOM: heatmap ──────────────────────────────────────────────────────
    if marker == 'square':
        # squares (original behavior): draw full 3-state matrix from VISUAL (lengthened) mat
        ax.imshow(np.clip(vis_mat, -1, 1).astype(np.int8) + 1,
                  cmap=cmap, vmin=0, vmax=2,
                  aspect='auto', interpolation='nearest', origin='upper')
    else:
        # circles: draw grey NA background, then exactly ONE circle per called bin
        # 1) grey for NA (-1), transparent elsewhere
        na_mask = (orig_mat < 0).astype(int)
        cmap_na = ListedColormap([(0, 0, 0, 0), (0.6, 0.6, 0.6, 1.0)])
        ax.imshow(
            na_mask, cmap=cmap_na, vmin=0, vmax=1,
            aspect='auto', interpolation='nearest', origin='upper'
        )

        # 2) one circle per called bin (use *original* matrix, not lengthened)
        #    1 == black state (status_black), 0 == white state (other)
        r_black, c_black = np.where(orig_mat == 1)
        r_white, c_white = np.where(orig_mat == 0)

        # circle size matched to cell size (area in points^2)
        D_pt = circle_scale * min(px_per_col, px_per_row) * 72.0 / DPI  # diameter in pt
        s = (np.pi / 4.0) * (D_pt ** 2)  # area that scatter() expects

        if r_black.size:
            ax.scatter(c_black, r_black, s=s, marker='o', c='k',
                       linewidths=0, zorder=3)
        if r_white.size:
            ax.scatter(c_white, r_white, s=s, marker='o',
                       facecolors='white', edgecolors='k', linewidths=0.5, zorder=3)



    # bounding box
    ax.add_patch(Rectangle((-0.5, -0.5), ncol, nrow, fill=False, linewidth=2.0, edgecolor='black'))


    # Row separators
    if row_sep and nrow > 1:
        y = np.arange(1, nrow) - 0.5
        ax.hlines(y, xmin=-0.5, xmax=ncol - 0.5,
                  colors=row_sep_color, linewidths=row_sep_width, zorder=3)

    # Center line & label
    ax.axvline(x0, color='red', linestyle='--', linewidth=1)
    ax.text(x0, -0.6, 'motif center', color='red', ha='center', va='bottom', fontsize=10)

    # X ticks
    def _zf_tick_positions(bins: np.ndarray):
        """Return (idx_list, label_list) for ZF 1..11."""
        idxs, labels = [], []
        # ZF k corresponds to rel_pos in [3*(k-6) .. 3*(k-6)+2]; tick near the start (or center).
        # We'll place the tick at bp ~ 3*(k-6) (nearest existing bin).
        for k in range(1, 12):
            target_bp = 3 * (k - 6)  # integer bp near the start of that 3bp window
            if target_bp < bins[0] or target_bp > bins[-1]:
                continue
            idx = int(np.argmin(np.abs(bins - target_bp)))
            idxs.append(idx)
            labels.append(str(k))  # or f"ZF{k}" if you prefer
        return idxs, labels

    if x_axis == 'bp':
        span = int(max(abs(bins[0]), abs(bins[-1])))
        anchors = np.array([-span, -span//2, 0, span//2, span], dtype=int)
        tick_idx = [int(np.argmin(np.abs(bins - a))) for a in anchors]
        ax.set_xticks(tick_idx)
        ax.set_xticklabels([str(int(bins[i])) for i in tick_idx], fontsize=10)
        ax.set_xlabel('Position relative to motif center (bp)', fontsize=11)

    elif x_axis == 'zf':
        zf_idx, zf_labels = _zf_tick_positions(bins)
        ax.set_xticks(zf_idx)
        ax.set_xticklabels(zf_labels, fontsize=10)
        ax.set_xlabel('CTCF zinc finger (ZF)', fontsize=11)

    elif x_axis == 'both':
        # Bottom: bp (as before)
        span = int(max(abs(bins[0]), abs(bins[-1])))
        anchors = np.array([-span, -span//2, 0, span//2, span], dtype=int)
        tick_idx = [int(np.argmin(np.abs(bins - a))) for a in anchors]
        ax.set_xticks(tick_idx)
        ax.set_xticklabels([str(int(bins[i])) for i in tick_idx], fontsize=10)
        ax.set_xlabel('Position relative to motif center (bp)', fontsize=11)

        # Top: ZF numbers
        zf_idx, zf_labels = _zf_tick_positions(bins)
        ax_top = ax.secondary_xaxis('top')
        ax_top.set_xticks(zf_idx)
        ax_top.set_xticklabels(zf_labels, fontsize=10)
        ax_top.set_xlabel('CTCF zinc finger (ZF)', fontsize=10)


    # Y axis
    ax.set_yticks([])
    ax.set_ylabel(f'{nrow} molecules', fontsize=11)

    # Group separators (between center-access blocks)
    if group_breaks is not None and len(group_breaks) > 0:
        for yb in group_breaks:
            if 0 < yb < nrow:
                ax.axhline(yb - 0.5, color='0.3', linewidth=1.2)

    # Legend (match marker shape to heatmap glyphs)
    shape = 'o' if marker == 'circle' else 's'
    legend_handles = [
        Line2D([0],[0], marker=shape, color='k', label=black_lbl,
               markerfacecolor='black', markersize=8, linewidth=0),
        Line2D([0],[0], marker=shape, color='k', label=white_lbl,
               markerfacecolor='white', markersize=8, linewidth=0.8),
        Line2D([0],[0], marker=shape, color='gray', label='no call / not assigned',
               markerfacecolor='0.6', markersize=8, linewidth=0)
    ]
    ax.legend(handles=legend_handles, frameon=False, loc='upper right')



    # Title
    if title is None:
        title = 'Per-read GpC states around motif (rows=reads, X=bp)'
    if group_counts is not None:
        fa, fi, pa, nc = group_counts
        title += f' | center: fully U={fa}, fully M={fi}, partial={pa}, none={nc}'
    fig.suptitle(title, y=0.995)
    
    # add the category bar / row stripe if requested
    # add the category bar / row stripe if requested
    # inside plot_profile_and_heat(...)
    if show_category_bar and (category_for_rows is not None) and category_colors:
        _add_category_bar(
            ax,
            category_for_rows,
            category_colors,
            show_stripe=show_row_category_stripe,
            bar_width_pct=category_bar_width_pct,
            stripe_width_pct=row_category_stripe_width_pct,
            label_min_frac=category_label_min_frac,
            pct_fontsize=category_label_fontsize,
            
        )











    # Save / show
    # Save / show
    if outpath:
        out = str(outpath)
        if not (out.lower().endswith('.pdf') or out.lower().endswith('.svg') or out.lower().endswith('.png')):
            out += '.pdf'  # default to PDF
        plt.savefig(out, dpi=DPI, bbox_inches='tight')
        print(f"Saved to {out} (reads={nrow}, bins={ncol}, dpi={DPI})")
        plt.close(fig)
    else:
        plt.show()



# ────────────────────────── main ──────────────────────────

def main():
    ap = argparse.ArgumentParser(description='Rows=reads, X=bp (symmetric bins). Heatmap + top fraction-U profile.')
    ap.add_argument('-i','--input', required=True, help='Calls TSV (wide 10-col or intersect 16+ col)')
    ap.add_argument('-m','--motif-bed', required=True, help='Motif BED: chr start end strand [specific_pos optional col5]')
    ap.add_argument('--calls-format', choices=['auto','wide','intersect'], default='auto')
    ap.add_argument('--assign-window', type=int, default=1500,
        help=('WIDE: max distance to assign call to nearest motif center; '
              '≤0 means unlimited (always assign to the nearest motif)'))

    ap.add_argument('--region', required=True, help='chr:start-end')
    ap.add_argument('--motif-index', type=int, default=None, help='If multiple motifs overlap, choose 0-based index')
    ap.add_argument('--span', type=int, default=500, help='Plot half-window (bp) around motif center (columns span -span..+span)')
    ap.add_argument('--core-span', type=int, default=60, help='Spanning criterion (both sides of 0 by >= core-span)')
    ap.add_argument('--bin', type=int, default=1, help='Bin size (bp) for X-axis')
    ap.add_argument('--black-is', choices=['M','U'], default='U', help='Which status is black (U=unmethylated, M=methylated)')

    ap.add_argument('--center-width', type=int, default=30, help='Half-width (bp) around 0 for center accessibility grouping')
    ap.add_argument('--min-cpg-per-read', type=int, default=0, help='Minimum CpG calls per read within ±span (after spanning)')
    ap.add_argument('--max-reads', type=int, default=0, help='If >0, cap number of reads (after ordering)')

    # CpG position labels
    ap.add_argument('--label-cpgs', action='store_true', help='Annotate measured CpG positions on the top axis')
    ap.add_argument('--label-every', type=int, default=1, help='Annotate every Nth measured CpG (1=all)')
    ap.add_argument('--label-fontsize', type=int, default=8, help='Font size for CpG position labels')
    ap.add_argument('--label-rotate', type=int, default=90, help='Rotation (deg) for CpG position labels')

    # Profile scatter density/size
    ap.add_argument('--profile-mark-every', type=int, default=1,
                    help='Plot markers+SD every Nth bin on the profile (1 = all bins)')
    ap.add_argument('--profile-marker-size', type=float, default=3.0,
                    help='Marker size for profile scatter')
    ap.add_argument('--profile-cap-size', type=float, default=2.0,
                    help='Cap size for error bars on profile scatter')

    # Cell geometry + resolution
    ap.add_argument('--row-px', type=float, default=6.0,
                    help='Pixel height per read row in the heatmap (increase to make rows thicker)')
    ap.add_argument('--col-px', type=float, default=6.0,
                    help='Pixel width per bp/bin column in the heatmap (increase to widen cells)')
    ap.add_argument('--dpi', type=int, default=96,
                    help='Figure DPI (export resolution)')

    # Visual lengthening of boxes (in bp; heatmap only)
    ap.add_argument('--lengthen-bp', type=int, default=0,
                    help='Visually extend each observed CpG box left/right by this many bp (heatmap only; profile unaffected)')

    # Row separators
    ap.add_argument('--row-sep', action='store_true',
                    help='Draw horizontal separator lines between reads')
    ap.add_argument('--row-sep-color', default='0.75',
                    help='Separator line color (e.g., 0.85, grey, #cccccc)')
    ap.add_argument('--row-sep-width', type=float, default=0.6,
                    help='Separator line width (points)')

    ap.add_argument('--report', action='store_true', help='Print selection counts at each stage')
    ap.add_argument('--export', required=True, help='Output PDF path')
    ap.add_argument('--profile', action='store_true')
    
    # add 'ctcf_categories' to choices
    ap.add_argument('--order-by',
    choices=['first_pos','read_id','center_access','ctcf_categories'],
    default='first_pos',
    help='Row order: earliest CpG, read_id, center accessibility groups, or CTCF categories')
    

# dynamic/static threshold (>=K measurable ZFs in 2..11)
    ap.add_argument('--ctcf-dyn-min-k', type=int, default=2,
        help='Min measurable ZFs in [1..11] to label dynamic/static (default 2)')


    # nucleosome-like flags (same definition as in the other script)
    ap.add_argument('--nuc-flag-up', action='store_true',
        help='Flag nucleosome-like reads if upstream window [-nuc-ndr-hi,-nuc-ndr-lo] has ≥1 CpG and all are U')
    ap.add_argument('--nuc-flag-down', action='store_true',
        help='Flag nucleosome-like reads if downstream window [nuc-ndr-lo,nuc-ndr-hi] has ≥1 CpG and all are U')
    ap.add_argument('--nuc-ndr-lo', type=int, default=40,
        help='Lower absolute bound (bp) for NDR windows')
    ap.add_argument('--nuc-ndr-hi', type=int, default=60,
        help='Upper absolute bound (bp) for NDR windows')
        
    ap.add_argument('--show-category-bar', action='store_true',
                    help='Draw a stacked fraction bar of CTCF/nucleosome/naked/dynamic/static categories on the right (only when ordering by ctcf_categories)')
    ap.add_argument('--row-category-stripe', action='store_true',
                    help='Draw a 1-column color stripe marking each row’s category (ctcf_categories only)')
    ap.add_argument('--category-colors', default='',
                    help='Comma-separated mapping name:color for category visuals')
    ap.add_argument('--category-bar-width-pct', type=float, default=10.0,
                help='Width of the stacked category bar as % of heatmap width (default 10)')
    ap.add_argument('--row-category-stripe-width-pct', type=float, default=2.0,
                    help='Width of the row-category stripe as % of heatmap width (default 2)')
    ap.add_argument('--category-label-fontsize', type=int, default=11,
                    help='Font size for % labels in the stacked bar (default 11)')
    ap.add_argument('--category-label-min-frac', type=float, default=0.03,
                    help='Minimum fraction to draw a % label (default 0.03 = 3%)')
    ap.add_argument('--no-profile-scatter', action='store_true',
                    help='Do not draw profile scatter markers or ±SD error bars')
    ap.add_argument(
        '--nuc-flanks-when-no-core',
        choices=['nucleosomal', 'naked'],
        default='naked',
        help=("When flanks look nucleosomal but no core ZFs (1..11) are bound, "
              "classify as 'naked' (default) or 'nucleosomal'."))

    ap.add_argument(
        '--only-ctcf',
        action='store_true',
        help='Show only CTCF-bound reads (ctcf_static + ctcf_dynamic); hide nucleosomal and naked')

    ap.add_argument('--x-axis', choices=['bp','zf','both'], default='bp',
    help="X-axis labeling: 'bp' (default), 'zf' (ZF 1..11), or 'both' (bp bottom, ZF top)")
    
    ap.add_argument(
        '--analysis-span', type=int, default=None,
        help=('Half-window (bp) used for *computations* (categories & nuc flags). '
              'Default: same as --span. If 0, auto = max(--span, --nuc-ndr-hi).')
    )

    ap.add_argument('--marker', choices=['square','circle'], default='square',
                    help='Heatmap cell glyph: square (default) or circle')
    ap.add_argument('--circle-scale', type=float, default=0.9,
                    help='Relative circle diameter vs cell size (0–1.2).')


    

                    
    




    args = ap.parse_args()
    
    raw_as = args.analysis_span
    if raw_as is None:
        analysis_span = args.span
    elif raw_as <= 0:
        analysis_span = max(args.span, args.nuc_ndr_hi)
    else:
        analysis_span = raw_as

    display_span = args.span

    calls_path = Path(args.input)
    motifs_path = Path(args.motif_bed)
    if not calls_path.exists(): print("ERROR: calls file not found", file=sys.stderr); sys.exit(2)
    if not motifs_path.exists(): print("ERROR: motif bed not found", file=sys.stderr); sys.exit(2)

    chrom, rstart, rend = parse_region(args.region)
    motif_df = read_motif_bed(str(motifs_path))
    target = find_target_motif(motif_df, chrom, rstart, rend, args.motif_index)
    center = int(target['specific_pos'])

    left  = center - analysis_span
    right = center + analysis_span



    fmt = detect_calls_format(str(calls_path)) if args.calls_format == 'auto' else args.calls_format

    if fmt == 'intersect':
        calls = stream_intersect_for_motif(str(calls_path), target, left, right)
    else:
        wide = stream_wide_for_region(str(calls_path), chrom, left, right)
        calls = map_wide_to_motif(wide, motif_df, chrom, args.assign_window)
        m = ((calls['chr_motif'].astype(str)==str(target['chr_motif'])) &
             (calls['start_motif']==int(target['start_motif'])) &
             (calls['end_motif']==int(target['end_motif'])) &
             (calls['strand_motif'].astype(str)==str(target['strand_motif'])))
        calls = calls[m]

    if calls.empty:
        print("No calls for the target motif/window. Nothing to plot.")
        sys.exit(0)

    # Compute once on the larger analysis window
    sub_all, stats = prepare_calls(
        calls, center, analysis_span, args.bin,
        core_span=args.core_span,
        min_cpg_per_read=args.min_cpg_per_read,
        report=args.report
    )
    if sub_all.empty:
        print("No reads pass spanning/min-CpG filters.")
        sys.exit(0)

    # Visible subset for the display window only
    sub_plot = sub_all[(sub_all['rel_pos'] >= -display_span) &
                       (sub_all['rel_pos'] <=  display_span)]
    if sub_plot.empty:
        print("No calls fall inside the display window.")
        sys.exit(0)



  

    # Base matrix ordering (first_pos or read_id)
    base_order = 'read_id' if args.order_by == 'read_id' else 'first_pos'
    mat, bins, read_ids = build_matrix_rows_symmetric(
        sub_plot, status_black=args.black_is,
        span=display_span, bin_size=args.bin, order_by=base_order
    )

   
    # ── NEW: order by CTCF categories (static → dynamic → nucleosomal → naked)
    # ── ORDERING & CATEGORY WIRING ───────────────────────────────────────────────
    group_breaks = None
    group_counts = None
    cats = None  # per-read categories (only when ctcf_categories)

    if args.order_by == 'ctcf_categories':
        order_idx, cat_breaks, cat_counts, cats = _order_by_ctcf_categories(
            sub_all,                 # <-- full analysis window for stable categories
            read_ids,                # <-- order the currently visible rows
            dyn_min_k=int(args.ctcf_dyn_min_k),
            nuc_lo=int(args.nuc_ndr_lo),
            nuc_hi=int(args.nuc_ndr_hi),
            flag_up=bool(args.nuc_flag_up),
            flag_dn=bool(args.nuc_flag_down),
            nuc_flanks_when_no_core=str(args.nuc_flanks_when_no_core),
        )
        mat      = mat[order_idx, :]
        read_ids = [read_ids[i] for i in order_idx]
        group_breaks = cat_breaks
        group_counts = tuple(cat_counts)
        if args.report:
            s, d, n, k = cat_counts
            print(f"[CTCF CATEGORIES] static={s}, dynamic={d} (dyn_min_k={args.ctcf_dyn_min_k}), "
                  f"nucleosomal={n}, naked={k}")

        # Optional: keep only static+dynamic
        if args.only_ctcf and cats is not None:
            cats_map = dict(zip(cats['read_id'].astype(str), cats['category']))
            keep_mask = np.array([cats_map.get(rid) in ('ctcf_static','ctcf_dynamic') for rid in read_ids], bool)
            if not keep_mask.any():
                print("No reads remain after --only-ctcf filter.")
                sys.exit(0)
            mat      = mat[keep_mask, :]
            read_ids = [rid for rid, keep in zip(read_ids, keep_mask) if keep]
            n_static  = sum(cats_map.get(rid) == 'ctcf_static'  for rid in read_ids)
            n_dynamic = sum(cats_map.get(rid) == 'ctcf_dynamic' for rid in read_ids)
            group_breaks = np.array([n_static], dtype=int)
            group_counts = (n_static, n_dynamic, 0, 0)


            if args.report:
                print(f"[ONLY CTCF] static={n_static}, dynamic={n_dynamic}, total={len(read_ids)}")

    elif args.order_by == 'center_access':
        order_idx, group_breaks, group_counts = reorder_by_center_access(
            mat, bins, black_is=args.black_is, center_width=args.center_width
        )
        mat = mat[order_idx, :]
        read_ids = [read_ids[i] for i in order_idx]
        if args.report:
            fa, fi, pa, nc = group_counts
            print(f"[CENTER ORDER] fully U={fa}, fully M={fi}, partial={pa}, no-center-calls={nc}")
    # else: keep base order


    # Truncate if requested (after final ordering)
    if args.max_reads and mat.shape[0] > args.max_reads:
        mat = mat[:args.max_reads, :]
        read_ids = read_ids[:args.max_reads]
        if group_breaks is not None:
            group_breaks = [b for b in group_breaks if b <= mat.shape[0]]

    # Category visuals (build after final read_ids are set)
    cat_colors = _parse_category_colors(args.category_colors)
    cat_for_rows = None
    if 'cats' in locals() and cats is not None:
        cats_map = dict(zip(cats['read_id'].astype(str), cats['category']))
        cat_for_rows = [cats_map.get(rid, 'naked') for rid in read_ids]







    # Measured CpG bins (positions with ≥1 call among plotted reads)
    measured_bins = np.sort(sub_plot['rel_bin'].unique()) if args.label_cpgs else None
    prof_mean, prof_std, prof_cnt = compute_fraction_U_profile(sub_plot, read_ids, bins)




    title = (f"{chrom}:{rstart}-{rend} | motif {target['chr_motif']}:"
             f"{target['start_motif']}-{target['end_motif']}({target['strand_motif']}) "
             f"center={center}; reads={len(read_ids)}")

    if args.profile:
        print(f"[PLOT] rows={mat.shape[0]} reads, cols={mat.shape[1]} bins "
              f"(bp from {bins[0]} to {bins[-1]} step {args.bin}); "
              f"measured_cpg_bins={0 if measured_bins is None else len(measured_bins)}")

    plot_profile_and_heat(
        mat, bins, read_ids, status_black=args.black_is,
        outpath=args.export, title=title,
        group_breaks=group_breaks, group_counts=group_counts,
        measured_bins=measured_bins, label_every=args.label_every,
        label_fontsize=args.label_fontsize, label_rotate=args.label_rotate,
        prof_mean=prof_mean, prof_std=prof_std, prof_cnt=prof_cnt,
        show_profile_scatter=(not args.no_profile_scatter),   # <-- NEW
        profile_mark_every=args.profile_mark_every,
        profile_marker_size=args.profile_marker_size,
        profile_cap_size=args.profile_cap_size,
        row_px=args.row_px, col_px=args.col_px, dpi=args.dpi,
        lengthen_bp=args.lengthen_bp,
        row_sep=args.row_sep, row_sep_color=args.row_sep_color, row_sep_width=args.row_sep_width,
        category_for_rows=cat_for_rows,
        category_colors=cat_colors,
        show_category_bar=args.show_category_bar,
        show_row_category_stripe=args.row_category_stripe,
        category_bar_width_pct=args.category_bar_width_pct,
        row_category_stripe_width_pct=args.row_category_stripe_width_pct,
        category_label_fontsize=args.category_label_fontsize,
        category_label_min_frac=args.category_label_min_frac,
        x_axis=args.x_axis,
        marker=args.marker,
        circle_scale=args.circle_scale,
    )




if __name__ == '__main__':
    main()


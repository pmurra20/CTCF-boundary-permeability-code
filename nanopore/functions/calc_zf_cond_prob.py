#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# Ensure PDF outputs embed fonts and use Arial
mpl.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.family'] = 'Arial'


def build_pivot(df_master, motif_df, args):
    """
    From calls + motif bed -> filtered per-(motif_id, read_id, ZF) pivot of bound (0/1),
    and the ZF axis (cols asc, rows desc).
    """
    # 1) Filter df_master by motif_df
    df = df_master.merge(
        motif_df[['chr_motif', 'start_motif', 'end_motif']],
        on=['chr_motif', 'start_motif', 'end_motif'],
        how='inner'
    )

    # Keep only M/U, define bound=1 if 'U'
    df = df[df['status'].isin(['M', 'U'])].copy()
    df['bound'] = (df['status'] == 'U').astype(int)

    # Compute rel_pos and assign ZF (3-bp bins; centered so ZF6 around rel_pos~0)
    sf = {'+': -1, '-': 1}
    df['rel_pos'] = (df['call_pos'] - df['specific_pos']) * df['strand_motif'].map(sf).fillna(1).astype(int)
    df['ZF'] = 6 + np.floor_divide(df['rel_pos'], 3).astype(int)

    # Build motif_id on the full table (needed for NDR + nuc-exclusion)
    def make_id(d):
        return (d['chr_motif'].astype(str) + '_' +
                d['start_motif'].astype(str) + '_' +
                d['end_motif'].astype(str) + '_' +
                d['strand_motif'].astype(str))
    df['motif_id'] = make_id(df)

    # ── Nucleosome exclusion (applies regardless of plotting NDR bins) ─────────
    drop_pairs = pd.DataFrame(columns=['motif_id', 'read_id'])
    if getattr(args, 'exclude_nucleosome_up', False) or getattr(args, 'exclude_nucleosome_down', False):
        lo = int(getattr(args, 'nuc_ndr_lo', 40)); hi = int(getattr(args, 'nuc_ndr_hi', 60))

        def _flag_all_U(df_win):
            if df_win.empty:
                return pd.DataFrame(columns=['motif_id','read_id','allU'])
            grp = df_win.groupby(['motif_id','read_id'])['status']
            n_calls = grp.count().rename('n_calls')
            fracU   = grp.apply(lambda s: (s == 'U').mean()).rename('fracU')
            tab = pd.concat([n_calls, fracU], axis=1).reset_index()
            tab['allU'] = (tab['n_calls'] >= 1) & (tab['fracU'] == 1.0)
            return tab.loc[tab['allU'], ['motif_id','read_id','allU']]

        n_up = 0; n_down = 0
        if getattr(args, 'exclude_nucleosome_up', False):
            up = df[df['rel_pos'].between(-hi, -lo)]
            up_flags = _flag_all_U(up)
            n_up = int(up_flags.shape[0])
            if n_up > 0:
                drop_pairs = pd.concat([drop_pairs, up_flags[['motif_id','read_id']]], ignore_index=True)
        if getattr(args, 'exclude_nucleosome_down', False):
            down = df[df['rel_pos'].between(lo, hi)]
            down_flags = _flag_all_U(down)
            n_down = int(down_flags.shape[0])
            if n_down > 0:
                drop_pairs = pd.concat([drop_pairs, down_flags[['motif_id','read_id']]], ignore_index=True)

        if not drop_pairs.empty:
            drop_pairs = drop_pairs.drop_duplicates()
            print(f"[nuc-excl] windows=[{lo},{hi}]  will remove (motif,read) pairs: {drop_pairs.shape[0]} (up={n_up}, down={n_down})")
        else:
            print(f"[nuc-excl] windows=[{lo},{hi}]  will remove: 0")

    # ── Core ZFs 1–11 with optional min_gpc ────────────────────────────────────
    df_core = df[df['ZF'].between(1, 11)].copy()

    # Optionally filter by min_gpc at (motif,read,ZF)
    if args.min_gpc > 0:
        counts = (
            df_core
            .groupby(['chr_motif', 'start_motif', 'end_motif', 'read_id', 'ZF'])['status']
            .count()
            .reset_index(name='n_calls')
        )
        keep = counts[counts['n_calls'] >= args.min_gpc][['chr_motif', 'start_motif', 'end_motif', 'read_id', 'ZF']]
        df_core = df_core.merge(
            keep.assign(_keep=1),
            on=['chr_motif', 'start_motif', 'end_motif', 'read_id', 'ZF'],
            how='inner'
        )

    # motif_id for core
    df_core['motif_id'] = make_id(df_core)

    # Collapse to one bound value per (motif_id, read_id, ZF)
    bound_df = (
        df_core
        .groupby(['motif_id', 'read_id', 'ZF'])['bound']
        .max()
        .reset_index()
    )

    # Apply nuc-exclusion to core (if any)
    if not drop_pairs.empty:
        before_pairs = bound_df[['motif_id', 'read_id']].drop_duplicates().shape[0]
        bound_df = bound_df.merge(drop_pairs.assign(_kill=1), on=['motif_id', 'read_id'], how='left')
        bound_df = bound_df[bound_df['_kill'].isna()].drop(columns=['_kill'])
        after_pairs = bound_df[['motif_id', 'read_id']].drop_duplicates().shape[0]
        print(f"[nuc-excl] removed core pairs: {before_pairs - after_pairs}")

    # Optionally enforce molecules with ≥ (or exactly) N distinct bound ZFs (CORE ONLY)
    if args.mol_threshold > 0:
        bound_counts = (
            bound_df[bound_df['bound'] == 1]
            .groupby(['motif_id', 'read_id'])['ZF']
            .nunique()
            .reset_index(name='n_bound')
        )
        if args.exact:
            keep_reads = bound_counts[bound_counts['n_bound'] == args.mol_threshold][['motif_id', 'read_id']]
        else:
            keep_reads = bound_counts[bound_counts['n_bound'] >= args.mol_threshold][['motif_id', 'read_id']]
        bound_df = bound_df.merge(keep_reads.assign(_keep=1), on=['motif_id', 'read_id'], how='inner')
        print(f"Applying --mol_threshold to CORE ZFs only (1-11): {'exact' if args.exact else '>='}{args.mol_threshold}  | kept pairs: {keep_reads.shape[0]}")

    
    # --- Optional flanking bins (outside the core ZFs) ------------------------
    if getattr(args, 'include_flanks', False):
        k = int(getattr(args, 'flank_bins', 8))
        w = int(getattr(args, 'flank_width', 3))
        start = int(getattr(args, 'flank_start', 18))

        df_all = df.copy()  # has motif_id, rel_pos, status, read_id

        def _win_to_bound(sub_df: pd.DataFrame) -> pd.DataFrame:
            """status→counts→(optional)min_gpc→bound per (motif_id,read_id)."""
            if sub_df.empty:
                return pd.DataFrame(columns=['motif_id','read_id','bound'])
            counts = (sub_df.groupby(['motif_id','read_id','status'])['status']
                      .count().reset_index(name='n_calls'))
            if args.min_gpc > 0:
                totals = (counts.groupby(['motif_id', 'read_id'])['n_calls']
                          .sum()
                          .reset_index(name='n_total'))
                keep = totals[totals['n_total'] >= args.min_gpc][['motif_id','read_id']]
                counts = counts.merge(keep.assign(_keep=1), on=['motif_id','read_id'], how='inner')
            b = (counts.groupby(['motif_id','read_id'])['status']
                 .apply(lambda s: 1 if 'U' in s.values else 0).reset_index(name='bound'))
            return b

        flank_rows = []

        # N side: ZF = -1..-k (nearest first). Windows are [lo_bp, hi_bp] inclusive.
        for i in range(1, k+1):
            hi_bp = -(start + (i-1)*w)   # e.g., -18, -21, ...
            lo_bp = hi_bp - (w-1)        # width = w
            subN = df_all[df_all['rel_pos'].between(lo_bp, hi_bp)]
            bN = _win_to_bound(subN); bN['ZF'] = -i
            flank_rows.append(bN)

        # C side: internally use 21..(20+k); we’ll label as +1..+k later
        for i in range(1, k+1):
            lo_bp =  start + (i-1)*w     # e.g., +18, +21, ...
            hi_bp =  lo_bp + (w-1)
            subC = df_all[df_all['rel_pos'].between(lo_bp, hi_bp)]
            bC = _win_to_bound(subC); bC['ZF'] = 20 + i
            flank_rows.append(bC)

        if flank_rows:
            flanks_df = pd.concat(flank_rows, ignore_index=True)
            # Apply nuc-exclusion to flank rows too
            if not drop_pairs.empty:
                flanks_df = flanks_df.merge(drop_pairs.assign(_kill=1),
                                            on=['motif_id','read_id'], how='left')
                flanks_df = flanks_df[flanks_df['_kill'].isna()].drop(columns=['_kill'])
            # Append to the core table
            bound_df = pd.concat([bound_df, flanks_df], ignore_index=True)

        # Heads-up if flanks may overlap your NDR window
        if args.include_ndr and start <= int(args.nuc_ndr_hi):
            print("[flanks] Note: flanks may overlap NDR windows; bins not mutually exclusive.")

    
    # Optionally add NDR bins as ZF=0 and ZF=12 (presence of any 'U' in window => bound=1)

    # Optionally add NDR bins as ZF=0 and ZF=12 (presence of any 'U' in window => bound=1)
    if args.include_ndr:
        df_all = df.copy()  # already has motif_id
        lo = int(args.nuc_ndr_lo); hi = int(args.nuc_ndr_hi)
        if lo > hi: lo, hi = hi, lo

        def ndr_block(d, lo_bp, hi_bp, zf_code):
            win = d[d['rel_pos'].between(lo_bp, hi_bp)]
            if win.empty:
                return pd.DataFrame(columns=['motif_id', 'read_id', 'bound', 'ZF'])
            counts = (win.groupby(['motif_id', 'read_id', 'status'])['status']
                          .count().reset_index(name='n_calls'))
            if args.min_gpc > 0:
                totals = counts.groupby(['motif_id', 'read_id'])['n_calls'].sum().reset_index(name='n_total')
                keep = totals[totals['n_total'] >= args.min_gpc][['motif_id', 'read_id']]
                counts = counts.merge(keep.assign(_keep=1), on=['motif_id', 'read_id'], how='inner')
            b = (counts.groupby(['motif_id', 'read_id'])['status']
                      .apply(lambda s: 1 if 'U' in s.values else 0)
                      .reset_index(name='bound'))
            b['ZF'] = zf_code
            return b

        up_bound = ndr_block(df_all, -hi, -lo, 0)
        dn_bound = ndr_block(df_all,  lo,  hi, 12)

        # Apply nuc-exclusion to NDR rows too
        if not drop_pairs.empty:
            if not up_bound.empty:
                up_bound = up_bound.merge(drop_pairs.assign(_kill=1), on=['motif_id','read_id'], how='left')
                up_bound = up_bound[up_bound['_kill'].isna()].drop(columns=['_kill'])
            if not dn_bound.empty:
                dn_bound = dn_bound.merge(drop_pairs.assign(_kill=1), on=['motif_id','read_id'], how='left')
                dn_bound = dn_bound[dn_bound['_kill'].isna()].drop(columns=['_kill'])

        bound_df = pd.concat([bound_df, up_bound, dn_bound], ignore_index=True)


    # ----- Choose bins to include -----
    # ----- Choose bins to include (and in what order) ------------------------
    if args.corr_all_bins:
        corr_bins = []

        # N flanks (furthest → nearest): -K .. -1
        if args.include_flanks:
            corr_bins.extend([-i for i in range(args.flank_bins, 0, -1)])

        if args.include_ndr:
            corr_bins.append(0)

        if args.include_cohesin:
            corr_bins.append(13)

        # Core bins
        if args.use_5clusters:
            core = [1, 2, 3, 4, 5]
        elif args.use_clusters:
            core = [1, 2, 3]
        else:
            core = list(range(1, 12))
        if args.hide_zf5:
            core = [b for b in core if b != (3 if args.use_5clusters else 5)]
        corr_bins.extend(core)

        if args.include_cohesin_control:
            corr_bins.append(14)

        if args.include_ndr:
            corr_bins.append(12)

        # C flanks (nearest → furthest): +1 .. +K (internally 21..20+K)
        if args.include_flanks:
            corr_bins.extend([20 + i for i in range(1, args.flank_bins + 1)])

        core_df = bound_df[bound_df['ZF'].isin(corr_bins)].copy()
        order_cols = corr_bins  # preserve intended order
    else:
        if args.use_5clusters:
            chosen = [1, 2, 3, 4, 5]
            if args.hide_zf5: chosen = [b for b in chosen if b != 3]
        elif args.use_clusters:
            chosen = [1, 2, 3]
        else:
            chosen = list(range(1, 12))
            if args.hide_zf5: chosen = [b for b in chosen if b != 5]
        core_df = bound_df[bound_df['ZF'].isin(chosen)].copy()
        order_cols = chosen


    # Pivot to matrix with possible NaNs where (motif_id, read_id, ZF) missing
    pivot = core_df.pivot_table(index=['motif_id', 'read_id'], columns='ZF', values='bound')
    present = [c for c in order_cols if c in pivot.columns]  # preserve intended order
    pivot = pivot.reindex(columns=present)


    # Optionally drop reads bound in NDRs (if NDRs exist in the pivot)
    if args.remove_ndr_bound:
        to_drop = pd.Series(False, index=pivot.index)
        if 0 in pivot.columns:
            to_drop |= (pivot[0] == 1)
        if 12 in pivot.columns:
            to_drop |= (pivot[12] == 1)
        pivot = pivot[~to_drop]

    # Optionally drop reads where all measured *core* ZFs are bound (ignore flanks/extra bins)
    if args.drop_all_bound:
        core_cols = [c for c in pivot.columns if isinstance(c, int) and 1 <= c <= 11]
        if core_cols:
            nonna_core = pivot[core_cols].notna()
            all_bound_core = ((pivot[core_cols] == 1) & nonna_core).sum(axis=1) == nonna_core.sum(axis=1)
            pivot = pivot.loc[~all_bound_core]


    zf_cols = present  # preserve intended order
    zf_rows = zf_cols[::-1]
    return pivot, zf_cols, zf_rows


def compute_cond_from_pivot(pivot, zf_cols, zf_rows, target_val):
    """
    Compute cond[i,j] = P(ZF_j=1 | ZF_i == target_val), oriented as rows desc, cols asc.
    """
    cond = pd.DataFrame(np.nan, index=zf_cols, columns=zf_cols, dtype=float)
    for i in zf_cols:
        rows_i = pivot[pivot[i] == target_val]  # automatically drops NaN in i
        for j in zf_cols:
            valid = rows_i[~rows_i[j].isna()]
            denom = len(valid)
            cond.at[i, j] = float((valid[j] == 1).sum()) / denom if denom > 0 else np.nan
    return cond.reindex(index=zf_rows, columns=zf_cols)


def compute_row_counts(pivot, zf_cols, target_val):
    """
    QC: per-i denominators = number of molecules where ZF_i is measured and equals target_val.
    Returns DataFrame with columns ['ZF', 'n_conditioning'] (ZF numeric).
    """
    data = [(i, int((pivot[i] == target_val).sum())) for i in zf_cols]
    return pd.DataFrame(data, columns=['ZF', 'n_conditioning'])


def compute_cell_counts_from_pivot(pivot, zf_cols, zf_rows, target_val):
    """
    Per-cell sample sizes for the conditional:
      counts[i,j] = number of molecules where ZF_i == target_val and ZF_j is measured (not NaN).
    Returned with rows descending, cols ascending (to match heatmaps).
    """
    counts = pd.DataFrame(0, index=zf_cols, columns=zf_cols, dtype=int)
    for i in zf_cols:
        rows_i = pivot[pivot[i] == target_val]  # drops NaN in i automatically
        for j in zf_cols:
            valid = rows_i[~rows_i[j].isna()]
            counts.at[i, j] = len(valid)
    return counts.reindex(index=zf_rows, columns=zf_cols)


def plot_heatmap(matrix, zf_cols, zf_rows, title, cmap, vmin, vmax, labels_map, outpath, cbar_label):
    """
    Heatmap with diagonal outline; show BLACK cells only where data is unavailable (NaN).
    """
    N = matrix.shape[0]
    fig, ax = plt.subplots(figsize=(8, 6))

    # Mask unavailable entries (NaN) and force them to render as black
    cm = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
    try:
        cm = cm.copy()
    except AttributeError:
        cm = mpl.colors.ListedColormap(cm(np.linspace(0, 1, cm.N)))
    cm.set_bad(color='black')
    data = np.ma.masked_invalid(matrix.values.astype(float))

    cax = ax.imshow(data, cmap=cm, vmin=vmin, vmax=vmax, interpolation='nearest')

    # Optional: very light cell borders (keep if you want)
    for r in range(N):
        for c in range(N):
            ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1,
                                       facecolor='none', edgecolor='black', linewidth=0.0))

    # Diagonal outline (visual guide only)
    for k in range(N):
        row_diag = N - (k + 1)
        col_diag = k
        ax.add_patch(plt.Rectangle((col_diag - 0.5, row_diag - 0.5), 1, 1,
                                   facecolor='none', edgecolor='black', linewidth=0.5))

    # ✅ REMOVE the unconditional “adjacent-pair black boxes” overlay
    # (that’s what was hiding real values like ZF4↔ZF6 when ZF5 is missing)

    # Axis ticks
    labels     = [labels_map.get(zf, f"{zf}") for zf in zf_cols]
    labels_rev = [labels_map.get(zf, f"{zf}") for zf in zf_rows]
    ax.set_xlabel("ZF (j)")
    ax.set_ylabel("ZF (i)")
    ax.set_xticks(np.arange(N))
    ax.set_yticks(np.arange(N))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels_rev)

    # Colorbar + title
    cbar = fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label)
    ax.set_title(title)
    plt.tight_layout()

    if outpath:
        if not outpath.lower().endswith('.pdf'):
            outpath = outpath + '.pdf'
        fig.savefig(outpath)
        plt.close(fig)
        print(f"Saved figure to {outpath}")
    else:
        plt.show()



def main():
    parser = argparse.ArgumentParser(
        description=(
            'Compute conditional probabilities P(ZF_j=1 | ZF_i=1) and P(ZF_j=1 | ZF_i=0) '
            'for a motif set, with optional Δ vs a second motif set. X ascending / Y descending.'
        )
    )
    parser.add_argument('-i', '--input', required=True,
                        help='Calls TSV (nanopolish/fimo intersect)')
    parser.add_argument('-m', '--motif-bed', required=True,
                        help='Motif BED/TSV: chr start end strand')
    parser.add_argument('--motif-bed2',
                        help='Second motif BED/TSV (optional) for Δ conditional probabilities')
    parser.add_argument('--min-gpc', type=int, default=0,
                        help='Minimum CpG calls per (motif,read,ZF); default=0')
    parser.add_argument('--mol-threshold', type=int, default=0,
                        help='Require ≥N distinct ZFs bound per read (0 disables)')
    parser.add_argument('--exact', action='store_true',
                        help='Use exactly N bound ZFs per read (with --mol-threshold)')
    parser.add_argument('--corr_all_bins', action='store_true',
                        help='Use full bin set (incl. NDR/cohesin bins via flags below)')
    parser.add_argument('--use_5clusters', action='store_true',
                        help='Use 5-cluster scheme (bins 1–5)')
    parser.add_argument('--use_clusters', action='store_true',
                        help='Use 3-cluster scheme (bins 1–3)')
    parser.add_argument('--include_ndr', action='store_true',
                        help='Include upstream (ZF=0) and downstream (ZF=12) NDR bins in plots')
    parser.add_argument('--include_cohesin', action='store_true',
                        help='With --corr_all_bins, include cohesin bin (13)')
    parser.add_argument('--include_cohesin_control', action='store_true',
                        help='With --corr_all_bins, include cohesin control bin (14)')
    parser.add_argument('--hide_zf5', action='store_true',
                        help='Exclude ZF5 (or its cluster)')
    parser.add_argument('--drop_all_bound', action='store_true',
                        help='Drop reads where ALL measured ZFs are bound')
    parser.add_argument('--remove_ndr_bound', action='store_true',
                        help='Drop reads bound in N_NDR (0) or C_NDR (12) if those bins are present')

    # Nucleosome exclusion flags (independent of plotting NDR bins)
    parser.add_argument('--exclude_nucleosome_up', action='store_true',
                        help='Drop (motif,read) with ALL U in upstream flank window [-nuc_ndr_hi,-nuc_ndr_lo]')
    parser.add_argument('--exclude_nucleosome_down', action='store_true',
                        help='Drop (motif,read) with ALL U in downstream flank window [nuc_ndr_lo,nuc_ndr_hi]')
    parser.add_argument('--nuc_ndr_lo', type=int, default=40,
                        help='Inner bound (bp) for nucleosome window (default 40)')
    parser.add_argument('--nuc_ndr_hi', type=int, default=60,
                        help='Outer bound (bp) for nucleosome window (default 60)')

    parser.add_argument('--cp_vmin', type=float, default=0.0,
                        help='Min for conditional probability color scale')
    parser.add_argument('--cp_vmax', type=float, default=1.0,
                        help='Max for conditional probability color scale')
    parser.add_argument('--diff_vmin', type=float, default=-1.0,
                        help='Min for Δ conditional probability color scale')
    parser.add_argument('--diff_vmax', type=float, default=1.0,
                        help='Max for Δ conditional probability color scale')

    # Outputs for Motif 1
    parser.add_argument('--export_cond_prob_bound',
                        help='Output PDF for P (j bound | i bound) heatmap (Motif 1)')
    parser.add_argument('--export_cond_prob_unbound',
                        help='Output PDF for P(j=1 | i=0) heatmap (Motif 1)')
    parser.add_argument('--export_cond_counts_bound',
                        help='Output TSV with per-i denominators for i=1 (Motif 1)')
    parser.add_argument('--export_cond_counts_unbound',
                        help='Output TSV with per-i denominators for i=0 (Motif 1)')
    parser.add_argument('--export_cell_counts_bound',
                        help='Output TSV for per-cell denominators used in P(j=1|i=1) (Motif 1)')
    parser.add_argument('--export_cell_counts_unbound',
                        help='Output TSV for per-cell denominators used in P(j=1|i=0) (Motif 1)')

    # Optional Δ & counts for Motif 2
    parser.add_argument('--export_diff_bound',
                        help='Output PDF for Δ P (j bound | i bound) (Motif1 − Motif2)')
    parser.add_argument('--export_diff_unbound',
                        help='Output PDF for Δ P(j=1 | i=0) (Motif1 − Motif2)')
    parser.add_argument('--export_cond_counts_bound2',
                        help='Output TSV with per-i denominators for i=1 (Motif 2)')
    parser.add_argument('--export_cond_counts_unbound2',
                        help='Output TSV with per-i denominators for i=0 (Motif 2)')
    parser.add_argument('--export_cell_counts_bound2',
                        help='Output TSV for per-cell denominators used in P(j=1|i=1) (Motif 2)')
    parser.add_argument('--export_cell_counts_unbound2',
                        help='Output TSV for per-cell denominators used in P(j=1|i=0) (Motif 2)')
    parser.add_argument('--include-flanks', dest='include_flanks', action='store_true',
                    help='Add flanking 3-bp bins outside the motif: N = -1..-K, C = +1..+K.')
    parser.add_argument('--flank-bins', dest='flank_bins', type=int, default=8,
                        help='Number of flanking bins per side (default 8).')
    parser.add_argument('--flank-width', dest='flank_width', type=int, default=3,
                        help='Width (bp) of each flanking bin (default 3).')
    parser.add_argument('--flank-start', dest='flank_start', type=int, default=18,
                        help='First bp offset just outside the core ZFs (default 18 for 3-bp bins).')

    args = parser.parse_args()

    # Read motif(s)
    motif_df1 = pd.read_csv(
        args.motif_bed, sep='\t', header=None, engine='python',
        usecols=[0, 1, 2, 3], names=['chr_motif', 'start_motif', 'end_motif', 'strand_motif']
    )
    motif_df2 = None
    if args.motif_bed2:
        motif_df2 = pd.read_csv(
            args.motif_bed2, sep='\t', header=None, engine='python',
            usecols=[0, 1, 2, 3], names=['chr_motif', 'start_motif', 'end_motif', 'strand_motif']
        )

    # Read calls
    df_master = pd.read_csv(args.input, sep='\t', header=None, engine='python')
    df_master.columns = [
        'chr_call','start_call','end_call','call_pos','strand_call',
        'read_id','llr_ratio','llr_met','llr_unmet','status',
        'chr_motif','start_motif','end_motif','strand_motif',
        'score','specific_pos'
    ]

    # Build Motif 1 pivot once
    pivot1, zf_cols, zf_rows = build_pivot(df_master, motif_df1, args)

    # Compute both conditionals for Motif 1
    cond1_bound   = compute_cond_from_pivot(pivot1, zf_cols, zf_rows, target_val=1)
    cond1_unbound = compute_cond_from_pivot(pivot1, zf_cols, zf_rows, target_val=0)

    # QC denominators (per row i)
    counts1_bound   = compute_row_counts(pivot1, zf_cols, target_val=1)
    counts1_unbound = compute_row_counts(pivot1, zf_cols, target_val=0)

    # Per-cell denominators
    cellcounts1_bound   = compute_cell_counts_from_pivot(pivot1, zf_cols, zf_rows, target_val=1) if args.export_cell_counts_bound else None
    cellcounts1_unbound = compute_cell_counts_from_pivot(pivot1, zf_cols, zf_rows, target_val=0) if args.export_cell_counts_unbound else None

    label_map = {0: 'N_NDR', 12: 'C_NDR', 13: 'Cohes_reg', 14: 'Ctrl_reg'}
    # Core labels (fallbacks)
    for z in range(1, 12):
        label_map.setdefault(z, f"{z}")
    # Flank labels (if present)
    if args.corr_all_bins and args.include_flanks:
        for i in range(1, args.flank_bins + 1):
            label_map[-i]     = f"-{i}"  # N side
            label_map[20 + i] = f"+{i}"  # C side


    # Plot Motif 1 (bound & unbound)
    if args.export_cond_prob_bound:
        plot_heatmap(
            cond1_bound, zf_cols, zf_rows,
            title="Conditional probability (Motif 1; given i=1)",
            cmap='Blues', vmin=args.cp_vmin, vmax=args.cp_vmax,
            labels_map=label_map, outpath=args.export_cond_prob_bound,
            cbar_label="P (j bound | i bound)"
        )
    if args.export_cond_prob_unbound:
        plot_heatmap(
            cond1_unbound, zf_cols, zf_rows,
            title="Conditional probability (Motif 1; given i=0)",
            cmap='Blues', vmin=args.cp_vmin, vmax=args.cp_vmax,
            labels_map=label_map, outpath=args.export_cond_prob_unbound,
            cbar_label="P(j=1 | i=0)"
        )

    # Save QC TSVs if requested (Motif 1)
    if args.export_cond_counts_bound:
        counts1_bound.to_csv(args.export_cond_counts_bound, sep='\t', index=False)
        print(f"Saved per-i denominators (i=1) to {args.export_cond_counts_bound}")
    if args.export_cond_counts_unbound:
        counts1_unbound.to_csv(args.export_cond_counts_unbound, sep='\t', index=False)
        print(f"Saved per-i denominators (i=0) to {args.export_cond_counts_unbound}")
    if args.export_cell_counts_bound and cellcounts1_bound is not None:
        cellcounts1_bound.to_csv(args.export_cell_counts_bound, sep='\t', index=True, index_label='ZF_row')
        print(f"Saved per-cell denominators (i=1, motif1) to {args.export_cell_counts_bound}")
    if args.export_cell_counts_unbound and cellcounts1_unbound is not None:
        cellcounts1_unbound.to_csv(args.export_cell_counts_unbound, sep='\t', index=True, index_label='ZF_row')
        print(f"Saved per-cell denominators (i=0, motif1) to {args.export_cell_counts_unbound}")

    # Optional Motif 2 + Δ (bound & unbound) + counts
    if motif_df2 is not None and (args.export_diff_bound or args.export_diff_unbound
                                  or args.export_cond_counts_bound2 or args.export_cond_counts_unbound2
                                  or args.export_cell_counts_bound2 or args.export_cell_counts_unbound2):
     
        
        pivot2, _, _ = build_pivot(df_master, motif_df2, args)
        pivot2 = pivot2.reindex(columns=zf_cols)
        cond2_bound   = compute_cond_from_pivot(pivot2, zf_cols, zf_rows, target_val=1)
        cond2_unbound = compute_cond_from_pivot(pivot2, zf_cols, zf_rows, target_val=0)

        counts2_bound   = compute_row_counts(pivot2, zf_cols, target_val=1) if args.export_cond_counts_bound2 else None
        counts2_unbound = compute_row_counts(pivot2, zf_cols, target_val=0) if args.export_cond_counts_unbound2 else None

        cellcounts2_bound   = compute_cell_counts_from_pivot(pivot2, zf_cols, zf_rows, target_val=1) if args.export_cell_counts_bound2 else None
        cellcounts2_unbound = compute_cell_counts_from_pivot(pivot2, zf_cols, zf_rows, target_val=0) if args.export_cell_counts_unbound2 else None

        if args.export_diff_bound:
            diff_bound = cond1_bound - cond2_bound.reindex(index=zf_rows, columns=zf_cols)
            plot_heatmap(
                diff_bound, zf_cols, zf_rows,
                title="Δ Conditional probability (Motif1 − Motif2; given i=1)",
                cmap='RdBu', vmin=args.diff_vmin, vmax=args.diff_vmax,
                labels_map=label_map, outpath=args.export_diff_bound,
                cbar_label="Δ P (j bound | i bound)"
            )
        if args.export_diff_unbound:
            diff_unbound = cond1_unbound - cond2_unbound.reindex(index=zf_rows, columns=zf_cols)
            plot_heatmap(
                diff_unbound, zf_cols, zf_rows,
                title="Δ Conditional probability (Motif1 − Motif2; given i=0)",
                cmap='RdBu', vmin=args.diff_vmin, vmax=args.diff_vmax,
                labels_map=label_map, outpath=args.export_diff_unbound,
                cbar_label="Δ P(j=1 | i=0)"
            )

        if args.export_cond_counts_bound2 and counts2_bound is not None:
            counts2_bound.to_csv(args.export_cond_counts_bound2, sep='\t', index=False)
            print(f"Saved per-i denominators (i=1, motif2) to {args.export_cond_counts_bound2}")
        if args.export_cond_counts_unbound2 and counts2_unbound is not None:
            counts2_unbound.to_csv(args.export_cond_counts_unbound2, sep='\t', index=False)
            print(f"Saved per-i denominators (i=0, motif2) to {args.export_cond_counts_unbound2}")
        if args.export_cell_counts_bound2 and cellcounts2_bound is not None:
            cellcounts2_bound.to_csv(args.export_cell_counts_bound2, sep='\t', index=True, index_label='ZF_row')
            print(f"Saved per-cell denominators (i=1, motif2) to {args.export_cell_counts_bound2}")
        if args.export_cell_counts_unbound2 and cellcounts2_unbound is not None:
            cellcounts2_unbound.to_csv(args.export_cell_counts_unbound2, sep='\t', index=True, index_label='ZF_row')
            print(f"Saved per-cell denominators (i=0, motif2) to {args.export_cell_counts_unbound2}")


if __name__ == "__main__":
    main()

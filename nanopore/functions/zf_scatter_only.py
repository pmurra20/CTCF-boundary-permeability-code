#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

#--------------------This function produces scatter plot and stats for Fig.3 -----------------------------
# python3 zf_scatter_only.py \
#  -i calls_vs_motif_specific.tsv \
#  --out-prefix fig1_bound_tot_th_1_min_gpc_1 \
#  --min_gpc 1 \
#  --include_ndr \
#  --mol_threshold 1 \
#  --exclude_nucleosome_up \
#  --exclude_nucleosome_down \
#  --hide_zf5 \
#  --nuc_ndr_lo 20 \
#  --nuc_ndr_hi 60 \
#  --occupancy_denominator kept \
#  --ymin -0.1 \
#  --ymax 1.1 \
#  --make_violin \
#  --show_n \
#  --show_molecules \
#  --scatter_size 10 \
#  --export_mappable_zf_summary mappable_per_site.tsv \
#  --export_mappability_by_gpc mappability_by_gpc.tsv \
#  --export_zf_balance ZF_summ.tsv \
#  --export_scatter_stats ZF_scatter.tsv \
#  --export_read_category_counts global.tsv \
#  --export_zf_stats per_bin_stats.tsv --run_qc --dynamic_curve_out dyn_curve \
#  --dynamic_curve_use kept \
#  --dynamic_curve_mode eq \
#  --dynamic_curve_kmax 5 --dynamic_curve_min_measured 2
#------------------- takes into account nucleosomal reads and removes them ---------------------------------


# Editable PDFs + Arial
mpl.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.family'] = 'Arial'

# ── Utils ────────────────────────────────────────────────────────────────────

# ── Regions BED filter ───────────────────────────────────────────────────────
def load_regions_bed_ids(path: str) -> set:
    """
    Read a 5-column BED-like file with columns:
      0: chr, 1: start, 2: end, 3: strand (+/-), 4: score (ignored)
    Returns a set of motif_id strings matching this script's motif_id scheme:
      f"{chr}_{start}_{end}_{strand}"
    """
    import pandas as _pd
    bed = _pd.read_csv(path, sep='\t', header=None, usecols=[0,1,2,3],
                       names=['chr','start','end','strand'])
    # enforce ints for coordinates; keep strand as-is ('+'/'-')
    bed['start'] = bed['start'].astype(int)
    bed['end']   = bed['end'].astype(int)
    motif_ids = (
        bed['chr'].astype(str) + '_' +
        bed['start'].astype(str) + '_' +
        bed['end'].astype(str) + '_' +
        bed['strand'].astype(str)
    )
    return set(motif_ids.values)

def compute_rel_pos(df: pd.DataFrame) -> pd.Series:
    """Signed offset from motif center in bp; oriented so ZF1→ZF11 reads left→right (N→C)."""
    sf = {'+': -1, '-': 1}  # flip + strand
    if not df['strand_motif'].isin(sf).all():
        bad = df.loc[~df['strand_motif'].isin(sf), 'strand_motif'].unique()
        raise ValueError(f"Unexpected strand_motif values: {bad}")
    return (df['call_pos'] - df['specific_pos']) * df['strand_motif'].map(sf).astype(int)


def map_zf(rel_pos: pd.Series) -> pd.Series:
    """Map relative positions (bp) to ZF bins (1..11) with 3 bp width centered on the motif."""
    return (6 + np.floor_divide(rel_pos, 3)).astype(int)

def _drop_pairs_safe(df_in: pd.DataFrame, excluded_pairs: pd.DataFrame) -> pd.DataFrame:
    """Drop any (motif_id, read_id) present in excluded_pairs from df_in (type-safe)."""
    if excluded_pairs is None or getattr(excluded_pairs, 'empty', True) or df_in.empty:
        return df_in
    df = df_in.copy()
    df['_key'] = df['motif_id'].astype(str) + '|' + df['read_id'].astype(str)
    exc = excluded_pairs.copy()
    exc['_key'] = exc['motif_id'].astype(str) + '|' + exc['read_id'].astype(str)
    drop_keys = set(exc['_key'].tolist())
    out = df[~df['_key'].isin(drop_keys)].drop(columns=['_key'])
    return out

# ── BH FDR adjust ─────────────────────────────────────────────────────────────
def bh_adjust(p_vals):
    """
    Benjamini–Hochberg FDR adjust a list of p-values.
    Returns adjusted p-values in the original order.
    """
    import numpy as _np
    p = _np.array(p_vals, dtype=float)
    m = len(p)
    order = _np.argsort(p)
    p_sorted = p[order]
    adj = _np.empty(m, dtype=float)
    for i, pi in enumerate(p_sorted):
        adj[i] = min(pi * m / (i + 1), 1.0)
    for i in range(m - 1, 0, -1):
        if adj[i-1] > adj[i]:
            adj[i-1] = adj[i]
    p_adj = _np.empty(m, dtype=float)
    p_adj[order] = adj
    return p_adj.tolist()

# ── Violin plot ───────────────────────────────────────────────────────────────
def export_zf_stats(bpm: pd.DataFrame, bins: list, labels: list, out_path: str):
    """
    Export per-bin descriptive stats of the *plotted* per-motif binding_prob values.
    Columns: bin, label, n, mean, sd, sem
    """
    rows = []
    for z, lab in zip(bins, labels):
        vals = bpm.loc[bpm['ZF'] == z, 'binding_prob'].dropna().values
        n = int(len(vals))
        mean = float(vals.mean()) if n > 0 else float('nan')
        sd = float(vals.std(ddof=1)) if n > 1 else (0.0 if n == 1 else float('nan'))
        sem = (sd / (n ** 0.5)) if n > 0 else float('nan')
        rows.append({'bin': int(z), 'label': lab, 'n': n, 'mean': mean, 'sd': sd, 'sem': sem})
    pd.DataFrame(rows, columns=['bin','label','n','mean','sd','sem']).to_csv(out_path, sep='\t', index=False)
    print(f"Wrote per-bin stats to {out_path}")


def plot_zf_violin(bpm: pd.DataFrame, bins: list, labels: list, args):
    """
    Violin plot of per-motif binding probabilities by bin, robust to empty bins.
    Skips bins with no data (after filters) and reports which were skipped.
    """
    # Build per-bin arrays
    raw = [(z, bpm.loc[bpm['ZF'] == z, 'binding_prob'].dropna().values) for z in bins]
    keep = [(z, vals) for z, vals in raw if len(vals) > 0]

    if not keep:
        print("No data for violin (all bins empty after filters); skipping.")
        return

    kept_bins = [z for z, _ in keep]
    data = [vals for _, vals in keep]

    # Map labels to kept bins
    label_map = {z: lab for z, lab in zip(bins, labels)}
    kept_labels = [label_map[z] for z in kept_bins]
    positions = np.arange(1, len(kept_bins) + 1, dtype=float)

    # Widths
    if getattr(args, 'violin_scale_widths', False):
        ns = np.array([len(v) for v in data], dtype=float)
        if ns.max() > 0:
            widths = 0.8 * (np.sqrt(ns / ns.max()) * 0.9 + 0.1)
        else:
            widths = 0.6
    else:
        widths = 0.6  # scalar ok

    fig, ax = plt.subplots(figsize=(8, 8))

    v = ax.violinplot(
        data, positions=positions, widths=widths,
        showmeans=False, showextrema=False, showmedians=False
    )

    # Style violin bodies
    for b in v['bodies']:
        b.set_alpha(0.35)
        b.set_edgecolor('black')
        b.set_linewidth(0.8)

    # Add medians, IQR, whiskers
    hw = float(getattr(args, 'violin_iqr_halfwidth', 0.35))
    rng = np.random.RandomState(0)
    for x, vals in zip(positions, data):
        q1 = np.percentile(vals, 25)
        q2 = np.percentile(vals, 50)
        q3 = np.percentile(vals, 75)
        iqr = q3 - q1
        lo = np.min(vals[vals >= (q1 - 1.5 * iqr)]) if len(vals) else q1
        hi = np.max(vals[vals <= (q3 + 1.5 * iqr)]) if len(vals) else q3
        ax.vlines(x, q1, q3, lw=6, alpha=0.9)       # IQR bar
        ax.hlines(q2, x - hw, x + hw, lw=2)         # median
        ax.vlines(x, lo, hi, lw=1)                  # whiskers
        ax.hlines([lo, hi], x - hw*0.55, x + hw*0.55, lw=1)

    # Optional jittered per-motif points (cap per bin)
    if getattr(args, 'violin_points', False):
        max_pts = int(getattr(args, 'violin_max_points', 300))
        for xi, vals in zip(positions, data):
            n = len(vals)
            if n == 0:
                continue
            if n > max_pts:
                idx = np.arange(n); rng.shuffle(idx)
                vals_plot = vals[np.sort(idx[:max_pts])]
            else:
                vals_plot = vals
            jitter = rng.normal(0, 0.04, size=len(vals_plot))
            ax.scatter(np.full(len(vals_plot), xi) + jitter, vals_plot, s=10, alpha=0.25, edgecolors='none')

    # Axes, labels, grid
    ax.set_xticks(positions)
    ax.set_xticklabels(kept_labels, rotation=0, ha='right')
    ax.set_ylabel("Fraction inaccessible")
    ax.set_xlim(0.5, len(positions) + 0.5)
    ax.set_ylim(-0.05, 1.05)
    ax.yaxis.grid(True, linestyle=':', alpha=0.5)

    # Title with nuc-excl annotation if active
    title = "Binding prob per bin – violin"
    # annotate when nucleosome exclusion is active
    if getattr(args, '_nuc_active', False):
        sides = []
        if getattr(args, '_nuc_up', False): sides.append('up')
        if getattr(args, '_nuc_down', False): sides.append('down')
        lo = int(getattr(args, '_nuc_lo', 40)); hi = int(getattr(args, '_nuc_hi', 60))
        title = title + f"  |  nuc-excl [{lo}-{hi}]bp ({'/'.join(sides) if sides else 'none'})"

    # always append denominator + set title
    title += "  |  denom=" + ("total" if args.occupancy_denominator == 'total' else "kept")
    ax.set_title(title)


    # n under x-axis (for kept bins only)
    if getattr(args, 'show_n', False):
        counts = [len(v) for v in data]
        for xi, n in zip(positions, counts):
            ax.text(xi, -0.08, f"n={n}", ha='center', va='top',
                    transform=ax.get_xaxis_transform(), fontsize=8, clip_on=False)
        plt.subplots_adjust(bottom=0.22)
    else:
        plt.subplots_adjust(bottom=0.12)

    fig.tight_layout()
    fig.savefig(args.out_prefix + "_zf_violin.pdf", dpi=600)
    plt.close(fig)
    print(f"Saved violin plot to {args.out_prefix}_zf_violin.pdf")

    # Log skipped bins (helps debugging)
    dropped = [label_map[z] for z in bins if z not in kept_bins]
    if dropped:
        print("Violin skipped empty bins:", ", ".join(dropped))


# ── Pairwise stats & heatmap ──────────────────────────────────────────────────

def export_pairwise_stats_and_heatmap(bpm: pd.DataFrame, bins: list, labels: list, args):
    """
    Pairwise tests between bins on per-motif binding probabilities.
    Default: Wilcoxon signed-rank (paired, two-sided) with BH/FDR.
    You can switch to unpaired Mann–Whitney with --pairwise_method mannwhitney.
    """
    try:
        from scipy.stats import mannwhitneyu as _mwu, wilcoxon as _wilcoxon
    except Exception as _e:
        print(f"Warning: scipy not available; skipping pairwise stats. ({_e})")
        return

    alpha = float(getattr(args, 'pairwise_alpha', 0.05))
    method = getattr(args, 'pairwise_method', 'wilcoxon').lower()

    rows = []
    pvals = []
    pairs = []

    if method == 'wilcoxon':
        # Pivot per motif so we can do paired tests on shared motifs
        pivot = bpm.pivot(index='motif_id', columns='ZF', values='binding_prob')
        for i, z1 in enumerate(bins):
            for j, z2 in enumerate(bins):
                if j <= i:
                    continue
                s1 = pivot[z1] if z1 in pivot.columns else None
                s2 = pivot[z2] if z2 in pivot.columns else None
                stat = float('nan'); p = float('nan'); n1 = 0; n2 = 0; n_paired = 0
                mean1 = float('nan'); mean2 = float('nan'); delta_mean = float('nan')
                if s1 is not None and s2 is not None:
                    pair = pd.concat([s1, s2], axis=1, join='inner').dropna()
                    n_paired = int(len(pair))
                    n1 = int(s1.notna().sum())
                    n2 = int(s2.notna().sum())
                    if n_paired > 0:
                        x = pair.iloc[:,0].values
                        y = pair.iloc[:,1].values
                        mean1 = float(np.mean(x)); mean2 = float(np.mean(y)); delta_mean = float(mean1 - mean2)
                        diffs = x - y
                        if np.allclose(diffs, 0):
                            stat, p = 0.0, 1.0
                        else:
                            try:
                                res = _wilcoxon(x, y, zero_method='wilcox', alternative='two-sided', correction=False, mode='auto')
                                stat, p = float(res.statistic), float(res.pvalue)
                            except Exception:
                                stat, p = float('nan'), float('nan')
                rows.append({
                    'bin1': int(z1), 'bin2': int(z2),
                    'test': 'wilcoxon',
                    'n1': n1, 'n2': n2, 'n_paired': n_paired,
                    'mean1': mean1, 'mean2': mean2, 'delta_mean': delta_mean,
                    'stat': stat, 'p_value': float(p if np.isfinite(p) else np.nan)
                })
                pvals.append(p if np.isfinite(p) else 1.0)
                pairs.append((int(z1), int(z2)))
    else:
        # Unpaired Mann–Whitney on per-motif values (as before)
        series = {int(z): bpm[bpm['ZF'] == z]['binding_prob'].values for z in bins}
        for i, z1 in enumerate(bins):
            x = series[int(z1)]
            for j, z2 in enumerate(bins):
                if j <= i:  # upper triangle
                    continue
                y = series[int(z2)]
                if len(x) == 0 or len(y) == 0:
                    stat, p = float('nan'), float('nan')
                else:
                    try:
                        res = _mwu(x, y, alternative='two-sided')
                        stat, p = float(res.statistic), float(res.pvalue)
                    except Exception:
                        stat, p = float('nan'), float('nan')
                rows.append({
                    'bin1': int(z1), 'bin2': int(z2),
                    'test': 'mannwhitney',
                    'n1': int(len(x)), 'n2': int(len(y)), 'n_paired': 0,
                    'mean1': float(np.mean(x) if len(x) else np.nan),
                    'mean2': float(np.mean(y) if len(y) else np.nan),
                    'delta_mean': float((np.mean(x) - np.mean(y)) if (len(x) and len(y)) else np.nan),
                    'stat': float(stat), 'p_value': float(p if np.isfinite(p) else np.nan)
                })
                pvals.append(p if np.isfinite(p) else 1.0)
                pairs.append((int(z1), int(z2)))

    # BH adjust
    p_adj = bh_adjust(pvals) if pvals else []
    for r, pa in zip(rows, p_adj):
        r['p_adj'] = float(pa)
        r['significant'] = int(pa < alpha)

    # Build symmetric p_adj matrix for heatmap
    z_to_idx = {int(z): i for i, z in enumerate(bins)}
    M = np.full((len(bins), len(bins)), np.nan, dtype=float)
    for (z1, z2), pa in zip(pairs, p_adj):
        i, j = z_to_idx[int(z1)], z_to_idx[int(z2)]
        M[i, j] = pa
        M[j, i] = pa

    # Export TSV
    out_tsv = getattr(args, 'export_pairwise_stats', None)
    if not out_tsv:
        out_tsv = f"{args.out_prefix}_zf_pairwise_{method}.tsv"
    pd.DataFrame(rows).to_csv(out_tsv, sep='\t', index=False)
    print(f"Wrote pairwise stats to {out_tsv} (method={method}, BH alpha={alpha})")

    # Heatmap PDF path
    out_pdf = getattr(args, 'export_pairwise_heatmap', None)
    if not out_pdf:
        out_pdf = f"{args.out_prefix}_zf_pairwise_padj_heatmap.pdf"

    # Plot -log10(p_adj) with stars
    fig, ax = plt.subplots(figsize=(1.2*len(bins), 1.2*len(bins)))
    with np.errstate(divide='ignore', invalid='ignore'):
        neglog = -np.log10(M)
    neglog[~np.isfinite(neglog)] = np.nan   # mask inf/nan
    im = ax.imshow(neglog, aspect='auto')
    ax.set_xticks(range(len(bins))); ax.set_yticks(range(len(bins)))
    ax.set_xticklabels(labels, rotation=45, ha='right'); ax.set_yticklabels(labels)
    fig.colorbar(im, ax=ax, label='-log10(p_adj)')

    def star_for(p):
        if not np.isfinite(p): return ""
        if p < 0.001: return "***"
        if p < 0.01:  return "**"
        if p < alpha: return "*"
        return ""
    for i in range(len(bins)):
        for j in range(len(bins)):
            s = star_for(M[i, j])
            if s:
                ax.text(j, i, s, ha='center', va='center', fontsize=7)

    ax.set_title(f"Pairwise {method.capitalize()}: BH-adjusted p-values")
    fig.tight_layout()
    fig.savefig(out_pdf, dpi=600)
    plt.close(fig)
    print(f"Wrote pairwise heatmap to {out_pdf}")

# ── Nucleosome exclusion (NDR-based windows) ─────────────────────────────────
def drop_nucleosome_reads(df_calls: pd.DataFrame, bound_df: pd.DataFrame, args):
    """
    Flag (motif_id, read_id) as nucleosome-like per side if there is ≥1 CpG in the window AND all are 'U'.
      Upstream:   rel_pos in [ -nuc_ndr_hi, -nuc_ndr_lo ]
      Downstream: rel_pos in [  nuc_ndr_lo,  nuc_ndr_hi ]
    Returns (filtered_bound_df, excluded_pairs_df[columns: motif_id, read_id, side]).
    """
    need_up = bool(getattr(args, 'exclude_nucleosome_up', False))
    need_dn = bool(getattr(args, 'exclude_nucleosome_down', False))
    if not (need_up or need_dn):
        return bound_df, pd.DataFrame(columns=['motif_id','read_id','side'])

    # ensure rel_pos & motif_id
    if 'rel_pos' not in df_calls.columns:
        df_calls['rel_pos'] = compute_rel_pos(df_calls)
    if 'motif_id' not in df_calls.columns:
        df_calls['motif_id'] = (df_calls['chr_motif'].astype(str) + '_' +
                                df_calls['start_motif'].astype(str) + '_' +
                                df_calls['end_motif'].astype(str) + '_' +
                                df_calls['strand_motif'].astype(str))
    lo = int(getattr(args, 'nuc_ndr_lo', 40))
    hi = int(getattr(args, 'nuc_ndr_hi', 60))
    if lo > hi:
        lo, hi = hi, lo

    core = df_calls[df_calls['status'].isin(['M','U'])].copy()
    drop_pairs, excl_rows = set(), []

    if need_up:
        up = core[core['rel_pos'].between(-hi, -lo)]
        if not up.empty:
            flagged = up.groupby(['motif_id','read_id'])['status'].apply(
                lambda s: (len(s)>0) and (np.all(s.values=='U'))
            ).reset_index(name='nuc_up')
            to_drop = flagged[flagged['nuc_up']==True][['motif_id','read_id']]
            drop_pairs.update(map(tuple, to_drop.values))
            excl_rows.extend([(a,b,'up') for a,b in to_drop.values])

    if need_dn:
        dn = core[core['rel_pos'].between(lo, hi)]
        if not dn.empty:
            flagged = dn.groupby(['motif_id','read_id'])['status'].apply(
                lambda s: (len(s)>0) and (np.all(s.values=='U'))
            ).reset_index(name='nuc_dn')
            to_drop = flagged[flagged['nuc_dn']==True][['motif_id','read_id']]
            drop_pairs.update(map(tuple, to_drop.values))
            excl_rows.extend([(a,b,'down') for a,b in to_drop.values])

    if not drop_pairs:
        return bound_df, pd.DataFrame(columns=['motif_id','read_id','side'])

    # filter bound_df
    bd = bound_df.set_index(['motif_id','read_id'])
    keep_idx = [idx for idx in bd.index if idx not in drop_pairs]
    removed = len(bd.index) - len(keep_idx)

    print(
        f"Excluded nucleosome-like (motif,read) pairs: {removed} "
        f"(unique={len(drop_pairs)}; up={'yes' if need_up else 'no'}, "
        f"down={'yes' if need_dn else 'no'}, window=[{lo},{hi}]bp)"
    )

    excl_df = pd.DataFrame(excl_rows, columns=['motif_id','read_id','side'])
    return bd.loc[keep_idx].reset_index(), excl_df

def export_read_category_counts(total_core: pd.DataFrame,
                               kept_core: pd.DataFrame,
                               nuc_excluded: pd.DataFrame,
                               out_path: str):
    """
    total_core: pre-exclusion core rows (ZF 1..11), one row per (motif_id,read_id,ZF) with 'bound' ∈ {0,1}
    kept_core:  post-exclusion core rows (ZF 1..11), same schema
    nuc_excluded: rows with columns ['motif_id','read_id','side'] (any side → nucleosome_bound)
    out_path: TSV to write

    Base categories (exclusive precedence):
      nucleosome_bound > CTCF_bound > nothing_bound

    Also reports CTCF-only dynamic/static among measurable ZFs in [1..11] for:
      - thresholds K ∈ {2,3,4,5} with "≥K" (at least K measurable ZFs)
      - exact K ∈ {2,3,4,5} with "==K" (exactly K measurable ZFs)
    """
    import numpy as np
    import pandas as pd

    # universes of (motif,read)
    U_total = total_core[['motif_id','read_id']].drop_duplicates()
    U_kept  = kept_core[['motif_id','read_id']].drop_duplicates()
    N_total = len(U_total); N_kept = len(U_kept)

    # nucleosome set
    if nuc_excluded is not None and not getattr(nuc_excluded, 'empty', True):
        nuc_pairs = set(map(tuple, nuc_excluded[['motif_id','read_id']].drop_duplicates().values))
    else:
        nuc_pairs = set()

    # classifier helper
    def classify_pairs(pair_df: pd.DataFrame, core_df: pd.DataFrame) -> pd.DataFrame:
        has_ctcf = (core_df.groupby(['motif_id','read_id'])['bound']
                    .max().rename('ctcf_any').reset_index())
        x = pair_df.merge(has_ctcf, on=['motif_id','read_id'], how='left').fillna({'ctcf_any':0})
        idx = list(map(tuple, x[['motif_id','read_id']].values))
        x['is_nuc'] = [k in nuc_pairs for k in idx]
        # precedence: nucleosome > CTCF > nothing
        x['category'] = np.select(
            [x['is_nuc'], ~x['is_nuc'] & (x['ctcf_any'] == 1)],
            ['nucleosome_bound', 'CTCF_bound'],
            default='nothing_bound'
        )
        return x[['motif_id','read_id','category']]

    total_pairs = classify_pairs(U_total, total_core)
    kept_pairs  = classify_pairs(U_kept,  kept_core)

    order = ['CTCF_bound','nucleosome_bound','nothing_bound']
    cnt_total = total_pairs['category'].value_counts().reindex(order, fill_value=0)
    cnt_kept  = kept_pairs['category'].value_counts().reindex(order, fill_value=0)

    # per-motif fraction helper
    def per_motif_stats(pairs_df: pd.DataFrame):
        if pairs_df.empty:
            return {c: (0, np.nan, np.nan, np.nan) for c in order}
        mat = (pairs_df.assign(n=1)
               .pivot_table(index='motif_id', columns='category', values='n',
                            aggfunc='sum', fill_value=0)
               .reindex(columns=order, fill_value=0))
        denom = mat.sum(axis=1).replace(0, np.nan)
        frac = mat.div(denom, axis=0)
        n_motifs = frac.shape[0]
        out = {}
        for c in order:
            col = frac[c]
            mean = float(col.mean())
            sd   = float(col.std(ddof=1)) if n_motifs > 1 else 0.0
            sem  = (sd / (n_motifs**0.5)) if n_motifs > 0 else np.nan
            out[c] = (n_motifs, mean, sd, sem)
        return out

    stats_total = per_motif_stats(total_pairs)
    stats_kept  = per_motif_stats(kept_pairs)

    rows = []
    for cat in order:
        n_motifs_total, mean_t, sd_t, sem_t = stats_total[cat]
        n_motifs_kept,  mean_k, sd_k, sem_k = stats_kept[cat]
        rows.append({
            'category': cat,
            'n_pairs_total': int(cnt_total[cat]),
            'frac_pairs_total': (cnt_total[cat] / N_total) if N_total > 0 else np.nan,
            'n_pairs_kept': int(cnt_kept[cat]),
            'frac_pairs_kept': (cnt_kept[cat] / N_kept) if N_kept > 0 else np.nan,
            'n_motifs_total': int(n_motifs_total),
            'mean_frac_per_motif_total': mean_t,
            'sd_frac_per_motif_total': sd_t,
            'sem_frac_per_motif_total': sem_t,
            'n_motifs_kept': int(n_motifs_kept),
            'mean_frac_per_motif_kept': mean_k,
            'sd_frac_per_motif_kept': sd_k,
            'sem_frac_per_motif_kept': sem_k,
        })

    # ---- dynamic/static among CTCF-bound for thresholds K in [2..5] on ZF 1..11 ----
    def motif_moments_flag(pairs_df: pd.DataFrame, flag_col: str):
        if pairs_df.empty:
            return (0, np.nan, np.nan, np.nan)
        mat = (pairs_df.groupby(['motif_id','read_id'])[flag_col].max()
               .reset_index()
               .groupby('motif_id')[flag_col].agg(['sum','count']))
        frac = mat['sum'] / mat['count'].replace(0, np.nan)
        n_m = frac.shape[0]
        mean = float(frac.mean())
        sd   = float(frac.std(ddof=1)) if n_m > 1 else 0.0
        sem  = (sd / (n_m**0.5)) if n_m > 0 else np.nan
        return (int(n_m), mean, sd, sem)

    def ctcf_dynamic_static_for_k(pairs_df: pd.DataFrame, core_df: pd.DataFrame, k: int, exact: bool):
        if pairs_df.empty or core_df.empty:
            return (pd.DataFrame(columns=['motif_id','read_id','dynamic','static_all_bound']), 0, 0, 0)
        ctcf_pairs = pairs_df[pairs_df['category'] == 'CTCF_bound'][['motif_id','read_id']].drop_duplicates()
        z = core_df[(core_df['ZF'] >= 1) & (core_df['ZF'] <= 11)][['motif_id','read_id','bound']].copy()
        if z.empty or ctcf_pairs.empty:
            return (pd.DataFrame(columns=['motif_id','read_id','dynamic','static_all_bound']), 0, 0, 0)
        by_pair = (z.groupby(['motif_id','read_id'])['bound']
                   .agg(['sum','count']).reset_index())
        by_pair = ctcf_pairs.merge(by_pair, on=['motif_id','read_id'], how='left').fillna({'sum':0,'count':0})
        by_pair = by_pair[(by_pair['count'] == k) if exact else (by_pair['count'] >= k)]
        if by_pair.empty:
            return (pd.DataFrame(columns=['motif_id','read_id','dynamic','static_all_bound']), 0, 0, 0)
        static_mask  = (by_pair['sum'] == by_pair['count'])
        dynamic_mask = ~static_mask
        pairs_out = by_pair[['motif_id','read_id']].copy()
        pairs_out['dynamic'] = dynamic_mask.astype(int)
        pairs_out['static_all_bound'] = static_mask.astype(int)
        return (pairs_out, int(dynamic_mask.sum()), int(static_mask.sum()), int(len(by_pair)))

    for k in [2,3,4,5]:
        dyn_t, n_dyn_t, n_sta_t, den_t = ctcf_dynamic_static_for_k(total_pairs, total_core, k, exact=False)
        dyn_k, n_dyn_k, n_sta_k, den_k = ctcf_dynamic_static_for_k(kept_pairs,  kept_core,  k, exact=False)
        n_m_dyn_t, mean_dyn_t, sd_dyn_t, sem_dyn_t = motif_moments_flag(dyn_t, 'dynamic')
        n_m_sta_t, mean_sta_t, sd_sta_t, sem_sta_t = motif_moments_flag(dyn_t, 'static_all_bound')
        n_m_dyn_k, mean_dyn_k, sd_dyn_k, sem_dyn_k = motif_moments_flag(dyn_k, 'dynamic')
        n_m_sta_k, mean_sta_k, sd_sta_k, sem_sta_k = motif_moments_flag(dyn_k, 'static_all_bound')
        rows.append({
            'category': f'ctcf_dynamic_mixture_zf>={k}',
            'n_pairs_total': n_dyn_t, 'frac_pairs_total': (n_dyn_t/den_t) if den_t>0 else np.nan,
            'n_pairs_kept': n_dyn_k,  'frac_pairs_kept':  (n_dyn_k/den_k) if den_k>0 else np.nan,
            'den_pairs_total': den_t, 'den_pairs_kept': den_k,
            'n_motifs_total': n_m_dyn_t, 'mean_frac_per_motif_total': mean_dyn_t,
            'sd_frac_per_motif_total': sd_dyn_t, 'sem_frac_per_motif_total': sem_dyn_t,
            'n_motifs_kept': n_m_dyn_k,  'mean_frac_per_motif_kept':  mean_dyn_k,
            'sd_frac_per_motif_kept':  sd_dyn_k,  'sem_frac_per_motif_kept':  sem_dyn_k,
        })
        rows.append({
            'category': f'ctcf_static_all_bound_zf>={k}',
            'n_pairs_total': n_sta_t, 'frac_pairs_total': (n_sta_t/den_t) if den_t>0 else np.nan,
            'n_pairs_kept': n_sta_k,  'frac_pairs_kept':  (n_sta_k/den_k) if den_k>0 else np.nan,
            'den_pairs_total': den_t, 'den_pairs_kept': den_k,
            'n_motifs_total': n_m_sta_t, 'mean_frac_per_motif_total': mean_sta_t,
            'sd_frac_per_motif_total': sd_sta_t, 'sem_frac_per_motif_total': sem_sta_t,
            'n_motifs_kept': n_m_sta_k,  'mean_frac_per_motif_kept':  mean_sta_k,
            'sd_frac_per_motif_kept':  sd_sta_k,  'sem_frac_per_motif_kept':  sem_sta_k,
        })

        # exact ==K
        dyn_t, n_dyn_t, n_sta_t, den_t = ctcf_dynamic_static_for_k(total_pairs, total_core, k, exact=True)
        dyn_k, n_dyn_k, n_sta_k, den_k = ctcf_dynamic_static_for_k(kept_pairs,  kept_core,  k, exact=True)
        n_m_dyn_t, mean_dyn_t, sd_dyn_t, sem_dyn_t = motif_moments_flag(dyn_t, 'dynamic')
        n_m_sta_t, mean_sta_t, sd_sta_t, sem_sta_t = motif_moments_flag(dyn_t, 'static_all_bound')
        n_m_dyn_k, mean_dyn_k, sd_dyn_k, sem_dyn_k = motif_moments_flag(dyn_k, 'dynamic')
        n_m_sta_k, mean_sta_k, sd_sta_k, sem_sta_k = motif_moments_flag(dyn_k, 'static_all_bound')
        rows.append({
            'category': f'ctcf_dynamic_mixture_zf=={k}',
            'n_pairs_total': n_dyn_t, 'frac_pairs_total': (n_dyn_t/den_t) if den_t>0 else np.nan,
            'n_pairs_kept': n_dyn_k,  'frac_pairs_kept':  (n_dyn_k/den_k) if den_k>0 else np.nan,
            'den_pairs_total': den_t, 'den_pairs_kept': den_k,
            'n_motifs_total': n_m_dyn_t, 'mean_frac_per_motif_total': mean_dyn_t,
            'sd_frac_per_motif_total': sd_dyn_t, 'sem_frac_per_motif_total': sem_dyn_t,
            'n_motifs_kept': n_m_dyn_k,  'mean_frac_per_motif_kept':  mean_dyn_k,
            'sd_frac_per_motif_kept':  sd_dyn_k,  'sem_frac_per_motif_kept':  sem_dyn_k,
        })
        rows.append({
            'category': f'ctcf_static_all_bound_zf=={k}',
            'n_pairs_total': n_sta_t, 'frac_pairs_total': (n_sta_t/den_t) if den_t>0 else np.nan,
            'n_pairs_kept': n_sta_k,  'frac_pairs_kept':  (n_sta_k/den_k) if den_k>0 else np.nan,
            'den_pairs_total': den_t, 'den_pairs_kept': den_k,
            'n_motifs_total': n_m_sta_t, 'mean_frac_per_motif_total': mean_sta_t,
            'sd_frac_per_motif_total': sd_sta_t, 'sem_frac_per_motif_total': sem_sta_t,
            'n_motifs_kept': n_m_sta_k,  'mean_frac_per_motif_kept':  mean_sta_k,
            'sd_frac_per_motif_kept':  sd_sta_k,  'sem_frac_per_motif_kept':  sem_sta_k,
        })

    pd.DataFrame(rows).to_csv(out_path, sep='\t', index=False)
    print(f"Wrote global read categories to {out_path} (TOTAL denom={N_total}, KEPT denom={N_kept})")



def export_global_read_category_summary(bound_df: pd.DataFrame,
                                       pre_nuc_bound: pd.DataFrame,
                                       nuc_excluded: pd.DataFrame,
                                       out_path: str):
    """
    Write a 3-row TSV with mean±SD (across motifs) of read-category fractions:
      - CTCF_bound: ≥1 core ZF bound
      - nucleosome_bound: nuc-like (up or down) AND not CTCF-bound
      - nothing_bound: neither of the above
    Fractions are reported for both kept and total denominators.
    """
    # --- Denominators (sets of (motif_id, read_id)) ---
    kept_pairs = bound_df[['motif_id','read_id']].drop_duplicates()
    total_pairs = pre_nuc_bound[['motif_id','read_id']].drop_duplicates()

    # --- CTCF-bound flags at read-level ---
    # kept: any non-NDR ZF has bound==1
    is_core_kept = (bound_df['ZF'] != 0) & (bound_df['ZF'] != 12)
    ctcf_kept_pairs = (bound_df[is_core_kept & (bound_df['bound'] == 1)]
                       [['motif_id','read_id']].drop_duplicates())
    ctcf_kept_set = set(map(tuple, ctcf_kept_pairs.values))

    # total: use pre-exclusion core bins (ZF 1..11)
    pre_core = pre_nuc_bound[pre_nuc_bound['ZF'].between(1,11)]
    ctcf_total_pairs = (pre_core[pre_core['bound'] == 1]
                        [['motif_id','read_id']].drop_duplicates())
    ctcf_total_set = set(map(tuple, ctcf_total_pairs.values))

    # --- Nucleosome-like flags (always from nuc_excluded) ---
    nuc_set = set(map(tuple, nuc_excluded[['motif_id','read_id']].drop_duplicates().values)) \
              if not nuc_excluded.empty else set()

    # --- Build per-motif counts under 'total' (exclusive categories) ---
    # classify each total pair with priority: CTCF > NUC > NONE
    total_pairs['_key'] = list(map(tuple, total_pairs[['motif_id','read_id']].values))
    total_pairs['cat'] = np.where(
        total_pairs['_key'].isin(ctcf_total_set), 'CTCF_bound',
        np.where(total_pairs['_key'].isin(nuc_set), 'nucleosome_bound', 'nothing_bound')
    )
    total_cts = (total_pairs.groupby(['motif_id','cat'])['_key']
                 .nunique().unstack('cat', fill_value=0))
    total_den = total_pairs.groupby('motif_id')['_key'].nunique()

    # --- Build per-motif counts under 'kept' (exclusive using kept set) ---
    kept_pairs['_key'] = list(map(tuple, kept_pairs[['motif_id','read_id']].values))
    kept_pairs['cat'] = np.where(
        kept_pairs['_key'].isin(ctcf_kept_set), 'CTCF_bound', 'nothing_bound'
    )
    # nucleosome reads were excluded, so no 'nucleosome_bound' in kept
    kept_cts = (kept_pairs.groupby(['motif_id','cat'])['_key']
                .nunique().unstack('cat', fill_value=0))
    kept_cts['nucleosome_bound'] = 0
    kept_den = kept_pairs.groupby('motif_id')['_key'].nunique()

    # --- Per-motif fractions ---
    cats = ['CTCF_bound','nucleosome_bound','nothing_bound']
    # align indexes
    total_cts = total_cts.reindex(columns=cats, fill_value=0)
    kept_cts  = kept_cts.reindex(columns=cats,  fill_value=0)

    total_frac = total_cts.div(total_den, axis=0).replace([np.inf, -np.inf], np.nan)
    kept_frac  = kept_cts.div(kept_den,  axis=0).replace([np.inf, -np.inf], np.nan)

    # --- Global mean±SD across motifs (skip motifs with 0 denom) ---
    def mm(df):
        return pd.Series({'mean': df.mean(skipna=True), 'sd': df.std(ddof=1, skipna=True)})

    rows = []
    for cat in cats:
        tf = total_frac[cat].dropna()
        kf = kept_frac[cat].dropna()
        rows.append({
            'category': cat,
            'mean_total': float(tf.mean()) if len(tf) else np.nan,
            'sd_total': float(tf.std(ddof=1)) if len(tf) > 1 else np.nan,
            'n_motifs_total': int(len(tf)),
            'mean_kept': float(kf.mean()) if len(kf) else np.nan,
            'sd_kept': float(kf.std(ddof=1)) if len(kf) > 1 else np.nan,
            'n_motifs_kept': int(len(kf)),
        })

    out = pd.DataFrame(rows, columns=[
        'category','mean_total','sd_total','n_motifs_total',
        'mean_kept','sd_kept','n_motifs_kept'
    ])
    out.to_csv(out_path, sep='\t', index=False)
    print(f"Wrote global bound summary to {out_path}")


# ── Plot ─────────────────────────────────────────────────────────────────────
def plot_zf_distribution(summary_df, data, positions, labels, total_motifs, args):
    fig, ax = plt.subplots(figsize=(8, 8))

    # jittered scatter of per-motif values
    for idx, vals in enumerate(data, start=1):
        if len(vals) > 0:
            x = np.full(len(vals), idx, dtype=float) + np.random.normal(0, 0.05, len(vals))
            ax.scatter(x, vals, s=args.scatter_size, color='grey', alpha=0.3, edgecolors='none')
    
    summary_df['std'] = summary_df['std'].fillna(0.0)

    # mean ± std
    ax.errorbar(positions, summary_df['mean'], yerr=summary_df['std'],
                fmt='o', color='red', capsize=0)

    # y limits
    max_with_err = (summary_df['mean'] + summary_df['std']).max()
    bottom = -0.05
    top = max(0.1, float(max_with_err) * 1.02 if pd.notna(max_with_err) else 1.0)
    # default auto (clamped to [0,1]), unless user overrides
    auto_bottom = max(0.0, bottom)
    auto_top = min(1.0, top)
    ax.set_ylim(
        auto_bottom if args.ymin is None else args.ymin,
        auto_top    if args.ymax is None else args.ymax
    )
    ax.autoscale(False)
    # ticks and labels
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=0, ha='right')

    # n labels
    if args.show_n:
        counts = [len(v) for v in data]
        for idx, n in enumerate(counts, start=1):
            ax.text(idx, -0.08, f"n={n}", ha='center', va='top',
                    transform=ax.get_xaxis_transform(), fontsize=8, clip_on=False)
        plt.subplots_adjust(bottom=0.22)
    else:
        plt.subplots_adjust(bottom=0.12)

    # title
    thresh = 'exact ' if args.exact else '≥'
    title = "Binding prob per "
    if args.use_5clusters:
        title += "5-cluster scheme "
    elif args.use_clusters:
        title += "3-cluster scheme "
    else:
        title += "bin "
    title += f"(core mol filter {thresh}{args.mol_threshold})"

    # annotate when nucleosome exclusion is active
    if getattr(args, '_nuc_active', False):
        sides = []
        if getattr(args, '_nuc_up', False): sides.append('up')
        if getattr(args, '_nuc_down', False): sides.append('down')
        lo = int(getattr(args, '_nuc_lo', 40)); hi = int(getattr(args, '_nuc_hi', 60))
        title = title + f"  |  nuc-excl [{lo}-{hi}]bp ({'/'.join(sides) if sides else 'none'})"
        
    title += "  |  denom=" + ("total" if args.occupancy_denominator == 'total' else "kept")
    ax.set_title(title)

    if args.show_molecules:
        ax.text(0.95, 0.95, f"N motif instances = {total_motifs}",
                transform=ax.transAxes, ha='right', va='top', fontsize=9)

    n_bins = len(positions)
    ax.set_xlim(0.5, n_bins + 0.5)
    ax.set_xlabel("Cluster" if (args.use_clusters or args.use_5clusters) else "Bin")
    ax.set_ylabel("Fraction inaccessible")

    fig.savefig(args.out_prefix + '_zf_scatter.pdf', dpi=1000)
    plt.close(fig)
    print(f"Saved zf_scatter to {args.out_prefix}_zf_scatter.pdf")

def plot_zf_scatter_density_heatmap(data, positions, labels, summary_df, args):
    """
    For each ZF/bin, compute a vertical density of Y and render as a column.
    Density is column-normalized so color reflects distribution shape per ZF.
    """
    import numpy as np
    try:
        from scipy.ndimage import gaussian_filter
    except Exception as _e:
        print(f"Warning: scipy not available; skipping scatter heatmap. ({_e})")
        return

    n_bins = len(positions)
    ny = int(getattr(args, 'heatmap_ny', 180))

    # Y limits consistent with other plots
    y_min = args.ymin if args.ymin is not None else -0.05
    y_max = args.ymax if args.ymax is not None else 1.05
    y_edges = np.linspace(y_min, y_max, ny + 1)

    # Build H[ny, n_bins]: each column is the Y-density for a ZF/bin
    H = np.zeros((ny, n_bins), dtype=float)
    for j, vals in enumerate(data):
        if len(vals) == 0:
            continue
        y = np.asarray(vals, dtype=float)
        y = y[(y >= y_min) & (y <= y_max)]
        if y.size == 0:
            continue
        counts, _ = np.histogram(y, bins=y_edges)
        # Convert to a PDF per column (shape emphasis)
        col = counts.astype(float)
        s = col.sum()
        if s > 0:
            col /= s
        H[:, j] = col

    # Smooth vertically (optional)
    sigma = float(getattr(args, 'heatmap_smooth', 1.2))
    if sigma > 0:
        H = gaussian_filter(H, sigma=(sigma, 0), mode='nearest')

    # Normalize each column to its own max for visual comparability
    colmax = H.max(axis=0)
    colmax[colmax == 0] = 1.0
    H = H / colmax

    # Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    cmap_name = getattr(args, 'heatmap_cmap', 'turbo')
    try:
        cmap = plt.get_cmap(cmap_name)
    except ValueError:
        print(f"Colormap '{cmap_name}' not found; falling back to 'viridis'.")
        cmap = plt.get_cmap('viridis')
    im = ax.imshow(
        H,
        origin='lower',
        aspect='auto',
        extent=[0.5, n_bins + 0.5, y_min, y_max],
        cmap=cmap,
        vmin=0.0, vmax=1.0,
        interpolation='nearest'
    )
    cbar = fig.colorbar(im, ax=ax, label='Density')

    # Axis / labels
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=0, ha='right')
    ax.set_ylabel("Fraction inaccessible")
    ax.set_xlim(0.5, n_bins + 0.5)
    ax.set_ylim(y_min, y_max)

    # Overlay mean ± SD (white) like in the scatter
    s = summary_df.copy()
    s['std'] = s['std'].fillna(0.0)
    ax.errorbar(
        positions, s['mean'], yerr=s['std'],
        fmt='o', color='white', ecolor='white', capsize=0, ms=4, lw=1
    )

    # Title mirrors your scatter title but notes density
    thresh = 'exact ' if args.exact else '≥'
    title = "Binding prob per "
    if args.use_5clusters:
        title += "5-cluster scheme "
    elif args.use_clusters:
        title += "3-cluster scheme "
    else:
        title += "bin "
    title += f"(core mol filter {thresh}{args.mol_threshold})"
    if getattr(args, '_nuc_active', False):
        sides = []
        if getattr(args, '_nuc_up', False): sides.append('up')
        if getattr(args, '_nuc_down', False): sides.append('down')
        lo = int(getattr(args, '_nuc_lo', 40)); hi = int(getattr(args, '_nuc_hi', 60))
        title += f"  |  nuc-excl [{lo}-{hi}]bp ({'/'.join(sides) if sides else 'none'})"
    title += "  |  denom=" + ("total" if args.occupancy_denominator == 'total' else "kept")
    ax.set_title(title + "  –  per-ZF vertical density")

    fig.tight_layout()
    out_pdf = args.out_prefix + "_zf_scatter_heatmap.pdf"
    fig.savefig(out_pdf, dpi=600)
    plt.close(fig)
    print(f"Saved per-ZF density heat map to {out_pdf}")


# ── Exports ──────────────────────────────────────────────────────────────────
def export_mappability_tables(df_all: pd.DataFrame, args):
    """Per-site mappability summary (respects --min_reads_per_zf) and aggregation by GpC."""
    # motif_id
    if 'motif_id' not in df_all.columns:
        df_all['motif_id'] = (df_all['chr_motif'].astype(str) + '_' +
                              df_all['start_motif'].astype(str) + '_' +
                              df_all['end_motif'].astype(str) + '_' +
                              df_all['strand_motif'].astype(str))
    # rel/ZF
    if 'rel_pos' not in df_all.columns:
        df_all['rel_pos'] = compute_rel_pos(df_all)
    df_core = df_all[df_all['status'].isin(['M','U'])].copy()
    df_core['ZF'] = map_zf(df_core['rel_pos'])
    df_core = df_core[df_core['ZF'].between(1, 11)]
    # distinct reads per motif×ZF
    zf_reads = df_core.groupby(['motif_id','ZF'])['read_id'].nunique().reset_index(name='n_reads_at_zf')
    zf_wide = zf_reads.pivot(index='motif_id', columns='ZF', values='n_reads_at_zf') \
                      .reindex(columns=list(range(1,12))).fillna(0).astype(int)
    zf_wide.columns = [f"zf{c}_reads" for c in zf_wide.columns]
    # mappable mask by threshold
    thr = int(getattr(args, 'min_reads_per_zf', 1))
    mappable_mask = (zf_wide.values >= thr).astype(int)
    n_zf_mappable = mappable_mask.sum(axis=1)
    fraction_zf_mappable = n_zf_mappable / 11.0
    summary = zf_wide.copy()
    summary['n_zf_mappable'] = n_zf_mappable
    summary['fraction_zf_mappable'] = fraction_zf_mappable
    if getattr(args, 'export_mappable_zf_summary', None):
        summary.reset_index().to_csv(args.export_mappable_zf_summary, sep='\t', index=False)
        print(f"Wrote per-site mappability summary to {args.export_mappable_zf_summary} (motifs={len(summary)})")
    # Aggregate by observed GpC count over core ZFs
    gpc_counts = df_core.groupby(['motif_id'])['call_pos'].nunique().reset_index(name='n_gpc_sites')
    merged = summary.merge(gpc_counts, on='motif_id', how='left')
    agg = merged.groupby('n_gpc_sites')['n_zf_mappable'].agg(['count','mean','median']).reset_index() \
               .rename(columns={'count':'n_sites','mean':'mean_zf_mappable','median':'median_zf_mappable'})
    if getattr(args, 'export_mappability_by_gpc', None):
        agg.to_csv(args.export_mappability_by_gpc, sep='\t', index=False)
        print(f"Wrote aggregated mappability-by-GpC table to {args.export_mappability_by_gpc}")

def export_zf_balance(df_all: pd.DataFrame, args):
    """Per-bin 'balance' summary under --min_reads_per_zf: sites_tested & reads_tested and fractions."""
    if not getattr(args, 'export_zf_balance', None):
        return
    if 'motif_id' not in df_all.columns:
        df_all['motif_id'] = (df_all['chr_motif'].astype(str) + '_' +
                              df_all['start_motif'].astype(str) + '_' +
                              df_all['end_motif'].astype(str) + '_' +
                              df_all['strand_motif'].astype(str))
    if 'rel_pos' not in df_all.columns:
        df_all['rel_pos'] = compute_rel_pos(df_all)
    df_core = df_all[df_all['status'].isin(['M','U'])].copy()
    df_core['ZF'] = map_zf(df_core['rel_pos'])
    df_core = df_core[df_core['ZF'].between(1, 11)]
    # reads per motif×ZF
    per = df_core.groupby(['motif_id','ZF'])['read_id'].nunique().reset_index(name='n_reads_at_zf')
    thr = int(getattr(args, 'min_reads_per_zf', 1))
    per['mappable'] = (per['n_reads_at_zf'] >= thr)
    # sites tested per ZF
    sites = per[per['mappable']].groupby('ZF')['motif_id'].nunique().reindex(list(range(1,12))).fillna(0).astype(int)
    # reads tested per ZF (sum over motifs that pass)
    reads = per[per['mappable']].groupby('ZF')['n_reads_at_zf'].sum().reindex(list(range(1,12))).fillna(0).astype(int)
    total_sites = int(df_core['motif_id'].nunique())
    total_reads = int(reads.sum()) if len(reads) else 0
    out = pd.DataFrame({
        'ZF': list(range(1,12)),
        'sites_tested': sites.values,
        'frac_sites_tested': sites.values / total_sites if total_sites>0 else np.nan,
        'reads_tested': reads.values,
        'frac_reads_tested': reads.values / total_reads if total_reads>0 else np.nan
    })
    out.to_csv(args.export_zf_balance, sep='\t', index=False)
    print(f"Wrote ZF-balance table to {args.export_zf_balance}")

def export_scatter_stats(bound_df: pd.DataFrame,
                         bpm: pd.DataFrame,
                         bins: list,
                         out_path: str,
                         pre_bound_df: pd.DataFrame,
                         excluded_pairs: pd.DataFrame,
                         args,
                         core_kept_precluster: pd.DataFrame = None):
    """
    Post-filter counts per plotted bin + counts of nucleosome-excluded pairs (up/down) that would
    have contributed. Also reports dynamic vs static read counts across core ZFs.

    Parameters
    ----------
    bound_df : pd.DataFrame
        Post-exclusion table used for plotting (after nucleosome exclusion and after any clustering).
        Columns: ['motif_id','read_id','ZF','bound'] (one row per motif×read×ZF/cluster).
    bpm : pd.DataFrame
        Per-motif binding probability table with ['motif_id','ZF','binding_prob'] used to derive
        motif counts per plotted bin.
    bins : list
        The list of bins actually plotted (e.g., [0,1..11,12] or clusters).
    out_path : str
        TSV output path.
    pre_bound_df : pd.DataFrame
        Pre-exclusion, pre-cluster table (typically `pre_nuc_bound`) with ['motif_id','read_id','ZF','bound'].
        Used to estimate how many excluded pairs would have contributed to each *original* ZF bin.
    excluded_pairs : pd.DataFrame
        Nucleosome-excluded pairs with columns ['motif_id','read_id','side'] (side ∈ {'up','down'}).
    args : argparse.Namespace
        CLI arguments (used only for consistency with caller; not required here).
    core_kept_precluster : pd.DataFrame, optional
        **Post-exclusion, pre-cluster** core table with columns ['motif_id','read_id','ZF','bound'].
        When provided, dynamic/static is computed strictly on core ZFs 1..11 from this table,
        independent of any cluster mapping used for plotting.
    """
    bd = bound_df.copy()
    bp = bpm.copy()
    pre = pre_bound_df.copy()
    excl = excluded_pairs.copy()

    # Global counts
    total_motifs = int(bp['motif_id'].nunique()) if not bp.empty else 0
    unique_reads_global = int(bd['read_id'].nunique()) if not bd.empty else 0

    # Motif/read participation per plotted bin
    motifs_per = {int(z): 0 for z in bins}
    reads_per  = {int(z): 0 for z in bins}
    if not bp.empty:
        tmp_m = (bp[(bp['ZF'].isin(bins)) & (bp['binding_prob'].notna())]
         .groupby('ZF')['motif_id'].nunique())
        for z, v in tmp_m.items():
            motifs_per[int(z)] = int(v)
    if not bd.empty:
        tmp_r = bd[bd['ZF'].isin(bins)].groupby('ZF')['read_id'].nunique()
        for z, v in tmp_r.items():
            reads_per[int(z)] = int(v)
    total_reads_bins = float(sum(reads_per.values()))

    # Excluded pairs → attribute to bins (pre-exclusion space)
    excl_up = set()
    excl_dn = set()
    if not excl.empty:
        excl_up = set(map(tuple, excl[excl['side'] == 'up'][['motif_id','read_id']].values))
        excl_dn = set(map(tuple, excl[excl['side'] == 'down'][['motif_id','read_id']].values))

    excl_up_per = {int(z): 0 for z in bins}
    excl_dn_per = {int(z): 0 for z in bins}
    if not pre.empty and (excl_up or excl_dn):
        pre_pairs = pre[['motif_id','read_id','ZF']].drop_duplicates()
        for z in bins:
            if z == 0:      # upstream NDR
                excl_up_per[int(z)] = len(excl_up)
                excl_dn_per[int(z)] = 0
                continue
            if z == 12:     # downstream NDR
                excl_up_per[int(z)] = 0
                excl_dn_per[int(z)] = len(excl_dn)
                continue
            mask_z = pre_pairs['ZF'] == z
            pairs_z = set(map(tuple, pre_pairs.loc[mask_z, ['motif_id','read_id']].values))
            excl_up_per[int(z)] = len(pairs_z & excl_up)
            excl_dn_per[int(z)] = len(pairs_z & excl_dn)

    # Assemble per-bin rows
    rows = []
    for z in bins:
        m = motifs_per[int(z)]
        r = reads_per[int(z)]
        frac_m = (m / total_motifs) if total_motifs > 0 else np.nan
        frac_r = (r / total_reads_bins) if total_reads_bins > 0 else np.nan
        rows.append({
            'bin': int(z),
            'motifs_in_bin': m,
            'frac_motifs_in_bin': frac_m,
            'reads_in_bin': r,
            'frac_reads_in_bin': frac_r,
            'excluded_up_pairs_in_bin': excl_up_per[int(z)],
            'excluded_down_pairs_in_bin': excl_dn_per[int(z)]
        })
    out_df = pd.DataFrame(rows)
    out_df['total_motifs_scatter'] = total_motifs
    out_df['unique_motifs_scatter_global'] = total_motifs
    out_df['total_reads_scatter_bins'] = int(total_reads_bins) if total_reads_bins == int(total_reads_bins) else total_reads_bins
    out_df['unique_reads_scatter_global'] = unique_reads_global

    # ---- Dynamic vs static (core ZFs only, POST-exclusion, PRE-cluster) ----
    # Always compute dynamic/static on POST-exclusion, PRE-cluster core ZFs (1..11)
    dynamic_pairs_total = static_all_bound_pairs_total = 0
    dynamic_unique_reads_total = static_all_bound_unique_reads_total = 0

    if core_kept_precluster is not None and not core_kept_precluster.empty:
        bd_core = (core_kept_precluster[core_kept_precluster['ZF'].between(1, 11)]
                   [['motif_id', 'read_id', 'ZF', 'bound']]
                   .drop_duplicates())

        if not bd_core.empty:
            by_pair = (bd_core
                       .groupby(['motif_id', 'read_id'])['bound']
                       .agg(['sum', 'count'])
                       .reset_index())

            dynamic_mask = (by_pair['sum'] > 0) & (by_pair['sum'] < by_pair['count'])
            static_all_bound_mask = (by_pair['sum'] == by_pair['count'])

            dynamic_pairs_total = int(dynamic_mask.sum())
            static_all_bound_pairs_total = int(static_all_bound_mask.sum())

            dynamic_unique_reads_total = int(by_pair.loc[dynamic_mask, 'read_id'].nunique())
            static_all_bound_unique_reads_total = int(by_pair.loc[static_all_bound_mask, 'read_id'].nunique())
    else:
        # Fallback to whatever core rows exist in bound_df if caller didn't pass precluster snapshot.
        # Note: after clustering, ZF labels may no longer be 1..11; this is a best-effort fallback.
        bins_core = [int(z) for z in bins if 1 <= int(z) <= 11]
        if len(bins_core) > 0 and not bd.empty:
            bd_core = (bd[bd['ZF'].isin(bins_core)]
                       [['motif_id', 'read_id', 'ZF', 'bound']]
                       .drop_duplicates())

            if not bd_core.empty:
                by_pair = (bd_core
                           .groupby(['motif_id', 'read_id'])['bound']
                           .agg(['sum', 'count'])
                           .reset_index())

                dynamic_mask = (by_pair['sum'] > 0) & (by_pair['sum'] < by_pair['count'])
                static_all_bound_mask = (by_pair['sum'] == by_pair['count'])

                dynamic_pairs_total = int(dynamic_mask.sum())
                static_all_bound_pairs_total = int(static_all_bound_mask.sum())

                dynamic_unique_reads_total = int(by_pair.loc[dynamic_mask, 'read_id'].nunique())
                static_all_bound_unique_reads_total = int(by_pair.loc[static_all_bound_mask, 'read_id'].nunique())

    # Append globals (same on every row for convenience)
    out_df['dynamic_pairs_total'] = dynamic_pairs_total
    out_df['static_all_bound_pairs_total'] = static_all_bound_pairs_total
    out_df['dynamic_unique_reads_total'] = dynamic_unique_reads_total
    out_df['static_all_bound_unique_reads_total'] = static_all_bound_unique_reads_total

    # Write
    out_df.to_csv(out_path, sep='\t', index=False)
    print(f"Wrote scatter-used counts to {out_path}")

def run_qc_checks(df_all, bound_df, pre_nuc_bound_adj, nuc_excluded, bpm, bins, labels, args):
    print("\n=== QC summary ===")
    # motifs contributing to plot stats (non-NaN binding_prob per bin)
    plot_ns = {int(z): int(bpm[(bpm['ZF']==z) & bpm['binding_prob'].notna()]['motif_id'].nunique()) for z in bins}
    print("QC[OK] motifs per plotted bin:", plot_ns)

    # reads per bin actually used post-exclusion (deduped per motif×read×ZF)
    reads_in_bin = (
        bound_df[bound_df['ZF'].isin(bins)]
        .groupby('ZF')['read_id'].nunique()
        .reindex(bins, fill_value=0).astype(int).to_dict()
    )
    print("QC[OK] reads_in_bin (post-exclusion):", reads_in_bin)

    # total motif instances in bpm
    print("QC[OK] total motifs in bpm:", int(bpm['motif_id'].nunique()))

    # nucleosome exclusions (if any)
    if nuc_excluded is not None and not nuc_excluded.empty:
        up = int((nuc_excluded['side'] == 'up').sum())
        dn = int((nuc_excluded['side'] == 'down').sum())
        print(f"QC[OK] nucleosome-excluded pairs: up={up}, down={dn}, total={up+dn}")
    else:
        print("QC[OK] nucleosome-excluded pairs: none")
    print("=== end QC ===\n")

def _dedup_core_111(core_df: pd.DataFrame) -> pd.DataFrame:
    """Ensure one row per (motif,read,ZF) in core 1..11 with bound∈{0,1}."""
    if core_df.empty:
        return core_df
    df = core_df[core_df['ZF'].between(1, 11)].copy()
    # in case there are duplicates, collapse to max bound
    df = (df.groupby(['motif_id','read_id','ZF'], as_index=False)['bound'].max())
    return df

def compute_dynamic_fraction_curve(core_df: pd.DataFrame,
                                   mode: str = 'eq',
                                   Kmax: int = 5,
                                   min_measured: int = 1) -> pd.DataFrame:
    """
    core_df: rows (motif_id, read_id, ZF, bound) in core ZFs 1..11 (pre- or post-exclusion per user choice).
    mode: 'eq'  => exactly K ZFs measured
          'ge'  => at least K ZFs measured
    min_measured: drop pairs with < min_measured ZFs measured before building K-slices
    Returns: DataFrame with columns K, den_pairs, num_dynamic, frac_dynamic
    """
    df = _dedup_core_111(core_df)
    if df.empty:
        return pd.DataFrame(columns=['K','den_pairs','num_dynamic','frac_dynamic'])

    by_pair = (df.groupby(['motif_id','read_id'])['bound']
               .agg(sum_bound='sum', count_measured='count')
               .reset_index())

    # ⬇️ NEW: enforce minimum # measured ZFs
    by_pair = by_pair[by_pair['count_measured'] >= int(min_measured)].copy()

    by_pair['is_ctcf'] = (by_pair['sum_bound'] > 0)
    by_pair['is_dyn']  = (by_pair['sum_bound'] > 0) & (by_pair['sum_bound'] < by_pair['count_measured'])

    rows = []
    startK = max(1, int(min_measured))  # ⬅️ don’t emit K below min_measured
    for K in range(startK, int(Kmax) + 1):
        if mode == 'eq':
            sel = (by_pair['count_measured'] == K)
        else:  # 'ge'
            sel = (by_pair['count_measured'] >= K)
        denom = int((by_pair['is_ctcf'] & sel).sum())
        dyn   = int((by_pair['is_dyn']  & sel).sum())
        frac  = (dyn / denom) if denom > 0 else float('nan')
        rows.append({'K': int(K), 'den_pairs': denom, 'num_dynamic': dyn, 'frac_dynamic': frac})

    # anchor point at K=0 for plotting
    rows.insert(0, {'K': 0, 'den_pairs': int(by_pair['is_ctcf'].sum()), 'num_dynamic': 0, 'frac_dynamic': 0.0})
    return pd.DataFrame(rows)

def _fit_exp_alpha_beta(K: np.ndarray,
                        y: np.ndarray,
                        weights: np.ndarray = None) -> dict:
    """
    Fit y ~ alpha * (1 - exp(-beta*K)) with alpha,beta >= 0.
    Tries SciPy if available; otherwise does a bounded coarse grid search.
    Returns dict with keys: alpha, beta, pred_11, r2, sse
    """
    K = np.asarray(K, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(K) & np.isfinite(y)
    K, y = K[m], y[m]

    if weights is None:
        w = np.ones_like(y)
    else:
        w = np.asarray(weights, dtype=float)[m]
        w = np.where(np.isfinite(w) & (w > 0), w, 1.0)

    def f(x, a, b):  # model
        return a * (1.0 - np.exp(-b * x))

    # Try SciPy
    try:
        from scipy.optimize import curve_fit
        # sensible initial guesses
        a0 = min(1.0, max(0.01, float(np.nanmax(y))))
        b0 = 0.5
        popt, _ = curve_fit(
            f, K, y, p0=(a0, b0),
            bounds=([0.0, 0.0], [1.0, 10.0]),
            sigma=1.0 / np.sqrt(w),
            absolute_sigma=False,
            maxfev=10000
        )
        a_hat, b_hat = float(popt[0]), float(popt[1])
    except Exception:
        # Coarse bounded grid search fallback
        y_max = float(np.nanmax(y)) if len(y) else 0.5
        a_grid = np.linspace(max(0.05, min(0.95, y_max)), 1.0, 61)
        b_grid = np.linspace(0.01, 5.0, 121)
        best = (1e99, 0.8, 0.8)
        for a in a_grid:
            # precompute (1 - exp(-b*K)) once per b
            for b in b_grid:
                yhat = f(K, a, b)
                sse = np.sum(w * (y - yhat) ** 2)
                if sse < best[0]:
                    best = (sse, a, b)
        a_hat, b_hat = best[1], best[2]

    yhat = f(K, a_hat, b_hat)
    # weighted R^2
    ybar = np.sum(w * y) / np.sum(w)
    ss_tot = np.sum(w * (y - ybar) ** 2)
    ss_res = np.sum(w * (y - yhat) ** 2)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else float('nan')
    pred_11 = f(11.0, a_hat, b_hat)
    return {'alpha': a_hat, 'beta': b_hat, 'pred_11': float(pred_11), 'r2': float(r2), 'sse': float(ss_res)}



def export_and_plot_dynamic_curve(core_df: pd.DataFrame,
                                  out_prefix: str,
                                  mode: str = 'eq',
                                  Kmax: int = 5,
                                  min_measured: int = 1):
    """
    Compute dynamic fraction *points* (no fit), write TSV and a scatter-only PDF.
    out_prefix: path prefix (writes <prefix>.tsv and <prefix>.pdf)
    """
    curve = compute_dynamic_fraction_curve(core_df, mode=mode, Kmax=Kmax, min_measured=min_measured)

    # Save TSV (optional, keep if you still want the numbers)
    curve.to_csv(out_prefix + '.tsv', sep='\t', index=False)

    # Scatter-only plot (finite points only)
    plot_df = curve[np.isfinite(curve['K']) & np.isfinite(curve['frac_dynamic'])].copy()

    fig, ax = plt.subplots(figsize=(6, 5))
    for sp in ax.spines.values():
        sp.set_zorder(0)
    
    ax.scatter(
        plot_df['K'], plot_df['frac_dynamic'],
        s=35,
        marker='o',
        facecolors='none',      # hollow circles
        edgecolors='blue',      # outline color (or keep as default)
        linewidths=1.8,
        zorder=10, # draw on top
        clip_on=False
    )


    ax.set_xlabel('# of ZFs sampled')
    ax.set_ylabel('dynamic/dynamic+static')
    ax.set_xlim(0.9, Kmax+0.1)
    ax.set_xticks(range(1, Kmax + 1))
    ax.set_ylim(0.0, 0.4)
    ax.set_title('Dynamic fraction vs #ZFs sampled')

    fig.tight_layout()
    fig.savefig(out_prefix + '.pdf', dpi=600)
    plt.close(fig)

    print(f"[dynamic-curve] scatter-only (no fit). wrote {out_prefix}.tsv and {out_prefix}.pdf")




# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description='Compute & plot binding probability per ZF (scatter-only)')
    p.add_argument('-i', '--input', required=True, help='Calls-vs-motif intersect TSV file')
    p.add_argument('--out-prefix', required=True, help='Prefix for output files')
    p.add_argument('--min_gpc', type=int, default=0, help='Minimum CpG calls per (motif,read,ZF) to keep')
    p.add_argument('--mol_threshold', type=int, default=1, help='Min distinct bound core ZFs per read (0=no filter)')
    p.add_argument('--exact', action='store_true', help='Use exact==threshold instead of >= threshold')
    p.add_argument('--focus_zf_range', type=str, help='Range "a-b" or list "a,b,c" of ZFs for focus filter')
    p.add_argument('--focus_zf_min', type=int, default=1, help='Min # bound ZFs within focus range to keep a read')
    p.add_argument('--drop_all_bound', action='store_true', help='Drop (motif,read) pairs with all measured ZFs bound')
    p.add_argument('--use_clusters', action='store_true', help='Collapse to 3 clusters: 1-3, 4-8, 9-11')
    p.add_argument('--use_5clusters', action='store_true', help='Collapse to 5 clusters: 1-2,3-4,5-7,8-9,10-11')
    p.add_argument('--hide_zf5', action='store_true', help='Hide ZF5 (or cluster 3 in 5-cluster mode)')
    p.add_argument('--include_ndr', action='store_true', help='Include upstream (ZF=0) and downstream (ZF=12) NDR bins')
    p.add_argument('--show_n', action='store_true', help='Show n per bin')
    p.add_argument('--show_molecules', action='store_true', help='Annotate with total motif count')
    p.add_argument('--make_violin', action='store_true', help='Also save a violin plot of per-motif binding probabilities')
    p.add_argument('--violin_points', action='store_true', help='Overlay jittered per-motif points on violin')
    p.add_argument('--violin_max_points', type=int, default=300, help='Max points per bin when overlaying jitter (default 300)')
    p.add_argument('--violin_scale_widths', action='store_true', help='Scale violin widths by sqrt(n) of motifs per bin')
    p.add_argument('--violin_iqr_halfwidth', type=float, default=0.35, help='Half-width of the IQR bar (default 0.35 x-axis units)')
    p.add_argument('--export_pairwise_stats', type=str, default=None, help='TSV path for pairwise tests (BH-corrected)')
    p.add_argument('--export_pairwise_heatmap', type=str, default=None, help='PDF path for pairwise adjusted-p heatmap')
    p.add_argument('--pairwise_alpha', type=float, default=0.05, help='Alpha threshold for BH significance (default 0.05)')
    p.add_argument('--pairwise_method', type=str, default='wilcoxon', help='Pairwise test: wilcoxon|mannwhitney (default wilcoxon)')

    # Nucleosome exclusion controls (honored only when --include_ndr is set)
    p.add_argument('--exclude_nucleosome_up', action='store_true',
                   help='Exclude reads with ALL CpGs bound (U) in upstream NDR window [ -nuc_ndr_hi, -nuc_ndr_lo ]')
    p.add_argument('--exclude_nucleosome_down', action='store_true',
                   help='Exclude reads with ALL CpGs bound (U) in downstream NDR window [  nuc_ndr_lo,  nuc_ndr_hi ]')
    p.add_argument('--nuc_ndr_lo', type=int, default=40, help='Lower absolute bound (bp) for NDR windows')
    p.add_argument('--nuc_ndr_hi', type=int, default=60, help='Upper absolute bound (bp) for NDR windows')

    # Mappability exports
    p.add_argument('--export_mappable_zf_summary', type=str, default=None, help='Per-site mappability summary TSV')
    p.add_argument('--export_mappability_by_gpc', type=str, default=None, help='Aggregated by GpC count TSV')
    p.add_argument('--min_reads_per_zf', type=int, default=1, help='Threshold for mappability exports')

    # ZF balance export
    p.add_argument('--export_zf_balance', type=str, default=None, help='Per-ZF sites/reads tested under mappability')

    # Scatter stats export
    p.add_argument('--export_scatter_stats', type=str, default=None,
                   help='TSV of post-filter counts per plotted bin + nucleosome-excluded pairs by side & bin')
    p.add_argument(
        '--occupancy-denominator', '--occupancy_denominator',
        dest='occupancy_denominator',
        choices=['kept','total'],
        default='kept',
        help="Denominator for per-motif binding fraction: "
             "'kept' (post-exclusion reads) or 'total' (pre-exclusion reads)."
    )
    p.add_argument('--ymin', type=float, default=None)
    p.add_argument('--ymax', type=float, default=None)
    p.add_argument('--scatter_size', type=float, default=35,
               help='Dot size for per-motif scatter points in the scatter plot (default 35)')
    p.add_argument(
        '--export_read_category_counts', type=str, default=None,
        help='TSV of global counts of read categories (CTCF_bound, nucleosome_bound, nothing_bound)'
    )
    p.add_argument(
        '--export_global_bound_summary', type=str, default=None,
        help='TSV with global fractions (mean±SD across motifs) for CTCF bound, nucleosome bound, nothing bound')

    p.add_argument('--make_scatter_heatmap', action='store_true',
               help='Save a per-ZF vertical-density heat map of the scatter')
    p.add_argument('--heatmap_ny', type=int, default=180,
                   help='Vertical (Y) bins for the density heat map')
    p.add_argument('--heatmap_smooth', type=float, default=1.2,
                   help='Gaussian sigma (in Y bins) for vertical smoothing')
    p.add_argument('--heatmap_cmap', default='turbo',
                   help='Matplotlib colormap (default: turbo)')
    
    # Per-ZF/bin descriptive stats export
    p.add_argument('--export_zf_stats', type=str, default=None,
               help='TSV: per-bin (ZF/cluster/NDR) mean, SD, n, SEM of binding_prob (post-filters; uses chosen denominator)')
    p.add_argument('--run_qc', action='store_true', help='Print a sanity-check summary of inputs, filters, and exports')
    
    p.add_argument('--dynamic_curve_out', type=str, default=None,
               help='Prefix for dynamic-fraction vs K export and plot (writes <prefix>.tsv and <prefix>.pdf)')
    p.add_argument('--dynamic_curve_use', choices=['kept','total'], default='kept',
                   help="Use post-exclusion ('kept') or pre-exclusion ('total') core reads")
    p.add_argument('--dynamic_curve_mode', choices=['eq','ge'], default='eq',
                   help="How to define K: 'eq' = exactly K ZFs measured; 'ge' = at least K measured")
    p.add_argument('--dynamic_curve_kmax', type=int, default=5,
                   help='Largest K to compute from data (we will still extrapolate to 11)')
    p.add_argument('--exclude_zfs_est_dynamics', type=str, default=None,
        help='Comma-separated ZF IDs to ignore ONLY for dynamic-curve estimation (e.g. "1" or "1,2")')
    p.add_argument(
    '--dynamic_curve_min_measured', type=int, default=1,
    help='Minimum # of core ZFs measured per (motif,read) to include in the dynamic-curve estimation (default 1; set 2 to drop single-ZF pairs)')
    
    p.add_argument('--regions-bed', type=str, default=None,
               help='Optional BED file (chrom start end strand score) to restrict analysis to listed motif instances.')
    p.add_argument('--export_scatter_points', type=str, default=None,
    help='TSV of per-motif points used in the scatter (motif_id, ZF/cluster/NDR label, binding_prob, counts)')
    
    p.add_argument('--keep_dynamic_only', action='store_true',
               help='Keep only dynamic (motif,read) pairs: among core ZFs 1..11 measured, at least one bound and at least one unbound; drops all-bound and all-unbound')
    p.add_argument('--export_scatter_matrix', type=str, default=None,
    help='Wide matrix: rows=motif_id, cols=plotted bins (ZF 1–11; N_NDR/C_NDR if included; or clusters), values=binding_prob')








    args = p.parse_args()

    # Load
    df = pd.read_csv(args.input, sep='\t', header=None, engine='python')
    df.columns = [
        'chr_call', 'start_call', 'end_call', 'call_pos', 'strand_call',
        'read_id', 'llr_ratio', 'llr_met', 'llr_unmet', 'status',
        'chr_motif', 'start_motif', 'end_motif', 'strand_motif',
        'score', 'specific_pos'
    ]

    # Prepare
    df = df[df['status'].isin(['M','U'])].copy()
    df['motif_id'] = (df['chr_motif'].astype(str) + '_' +
                      df['start_motif'].astype(str) + '_' +
                      df['end_motif'].astype(str) + '_' +
                      df['strand_motif'].astype(str))
    if 'rel_pos' not in df.columns:
        df['rel_pos'] = compute_rel_pos(df)
    df['ZF'] = map_zf(df['rel_pos'])
    
    # Optional BED-based restriction of motif instances
    if args.regions_bed:
        keep_ids = load_regions_bed_ids(args.regions_bed)
        before = df['motif_id'].nunique()
        df = df[df['motif_id'].isin(keep_ids)].copy()
        after = df['motif_id'].nunique()
        kept_pct = (after / before * 100.0) if before else 0.0
        print(f"[regions-bed] retained {after}/{before} motif instances ({kept_pct:.1f}%) from {args.regions_bed}")
        if after == 0:
            print("[regions-bed] No motifs from the BED matched the input. Exiting.")
            return


    # Core (1..11)
    base = df[df['ZF'].between(1, 11)].groupby(['motif_id','read_id','ZF','status'])['status'] \
                                       .count().reset_index(name='n_calls')
    # min_gpc per (motif,read,ZF)
    if args.min_gpc > 0:
        total_calls = base.groupby(['motif_id','read_id','ZF'])['n_calls'].sum().reset_index(name='n_total')
        keep = total_calls[total_calls['n_total'] >= args.min_gpc][['motif_id','read_id','ZF']]
        base = base.merge(keep.assign(_keep=1), on=['motif_id','read_id','ZF'], how='left')
        base = base[base['_keep'] == 1].drop(columns=['_keep'])

    # Reduce to bound per (motif,read,ZF)
    bound_df = base.groupby(['motif_id','read_id','ZF'])['status'].apply(
        lambda s: 1 if ('U' in s.values) else 0
    ).reset_index(name='bound')


    # Molecule-level filters
    if args.mol_threshold > 0:
        core_bound = bound_df[(bound_df['ZF'].between(1, 11)) & (bound_df['bound'] == 1)]
        print(f"Applying --mol_threshold to CORE ZFs only (1–11): {('exact' if args.exact else '≥')}{args.mol_threshold}")
        cnt = core_bound.groupby(['motif_id','read_id'])['ZF'].nunique().reset_index(name='nbound')
        keep_pairs = cnt[cnt['nbound'] == args.mol_threshold] if args.exact else cnt[cnt['nbound'] >= args.mol_threshold]
        bound_df = bound_df.merge(keep_pairs[['motif_id','read_id']], on=['motif_id','read_id'], how='inner')

    if args.focus_zf_range:
        if '-' in args.focus_zf_range:
            a, b = map(int, args.focus_zf_range.split('-'))
            focus = list(range(a, b+1))
        else:
            focus = [int(x) for x in args.focus_zf_range.split(',')]
        sub = bound_df[(bound_df['ZF'].isin(focus)) & (bound_df['bound'] == 1)]
        cnt = sub.groupby(['motif_id','read_id'])['ZF'].nunique().reset_index(name='focus_nbound')
        keep = cnt[cnt['focus_nbound'] >= args.focus_zf_min][['motif_id','read_id']]
        bound_df = bound_df.merge(keep, on=['motif_id','read_id'], how='inner')

    if args.drop_all_bound:
        pivot_tmp = bound_df.pivot_table(index=['motif_id','read_id'], columns='ZF', values='bound')
        nonna = pivot_tmp.notna()
        ones = (pivot_tmp == 1) & nonna
        all_bound_mask = (ones.sum(axis=1) == nonna.sum(axis=1))
        keep_pairs = all_bound_mask[~all_bound_mask].index
        bound_df = bound_df.set_index(['motif_id','read_id']).loc[keep_pairs].reset_index()

    # Nucleosome-like exclusion (only if include_ndr and flag(s) set)
    nuc_active = (args.include_ndr and (args.exclude_nucleosome_up or args.exclude_nucleosome_down))
    setattr(args, '_nuc_active', nuc_active)
    setattr(args, '_nuc_up', bool(args.exclude_nucleosome_up))
    setattr(args, '_nuc_down', bool(args.exclude_nucleosome_down))
    setattr(args, '_nuc_lo', int(args.nuc_ndr_lo))
    setattr(args, '_nuc_hi', int(args.nuc_ndr_hi))

    pre_nuc_bound = bound_df.copy()
    pre_core_for_totals = pre_nuc_bound.copy()

    if nuc_active:
        bound_df, nuc_excluded = drop_nucleosome_reads(df.copy(), bound_df, args)
        print(f"Nucleosome exclusion ACTIVE (include_ndr={args.include_ndr}, up={args.exclude_nucleosome_up}, down={args.exclude_nucleosome_down}).")
    else:
        nuc_excluded = pd.DataFrame(columns=['motif_id','read_id','side'])
        print(f"Nucleosome exclusion SKIPPED (include_ndr={args.include_ndr}, up={args.exclude_nucleosome_up}, down={args.exclude_nucleosome_down}).")
   
    # -- Optional: keep only dynamic pairs (post-exclusion, core ZFs 1..11) --
    if getattr(args, 'keep_dynamic_only', False):
        bd_core = (bound_df[bound_df['ZF'].between(1, 11)]
                   .groupby(['motif_id','read_id','ZF'], as_index=False)['bound'].max())
        if bd_core.empty:
            print("[keep_dynamic_only] No core (1..11) rows available after filters; exiting.")
            return
        by_pair = (bd_core.groupby(['motif_id','read_id'])['bound']
                   .agg(sum_bound='sum', count_measured='count').reset_index())
        dyn = by_pair[(by_pair['sum_bound'] > 0) & (by_pair['sum_bound'] < by_pair['count_measured'])]
        before_pairs = int(bound_df[['motif_id','read_id']].drop_duplicates().shape[0])
        kept_pairs = dyn[['motif_id','read_id']]
        bound_df = bound_df.merge(kept_pairs, on=['motif_id','read_id'], how='inner')
        after_pairs = int(bound_df[['motif_id','read_id']].drop_duplicates().shape[0])
        print(f"[keep_dynamic_only] kept {after_pairs}/{before_pairs} (motif,read) pairs that are dynamic (0<bound<count) across ZFs 1..11.")
        if after_pairs == 0:
            print("[keep_dynamic_only] No dynamic pairs left; exiting.")
            return
    # Optionally add NDR pseudo-bins using the SAME window edges as nucleosome exclusion
    if args.include_ndr:
        lo = int(args.nuc_ndr_lo); hi = int(args.nuc_ndr_hi)
        if lo > hi:
            lo, hi = hi, lo

        # Upstream NDR: rel_pos in [-hi, -lo]
        up = df[df['rel_pos'].between(-hi, -lo)]
        up_counts = up.groupby(['motif_id','read_id','status'])['status'].count().reset_index(name='n_calls')
        if args.min_gpc > 0:
            up_total = up_counts.groupby(['motif_id','read_id'])['n_calls'].sum().reset_index(name='n_total')
            upkeep = up_total[up_total['n_total'] >= args.min_gpc][['motif_id','read_id']]
            up_counts = up_counts.merge(upkeep.assign(_keep=1), on=['motif_id','read_id'], how='left')
            up_counts = up_counts[up_counts['_keep'] == 1].drop(columns=['_keep'])
        up_bound = (up_counts.groupby(['motif_id','read_id'])['status']
                    .apply(lambda s: 1 if 'U' in s.values else 0).reset_index(name='bound'))
        up_bound['ZF'] = 0

        # Downstream NDR: rel_pos in [lo, hi]
        dn = df[df['rel_pos'].between(lo, hi)]
        dn_counts = dn.groupby(['motif_id','read_id','status'])['status'].count().reset_index(name='n_calls')
        if args.min_gpc > 0:
            dn_total = dn_counts.groupby(['motif_id','read_id'])['n_calls'].sum().reset_index(name='n_total')
            dnkeep = dn_total[dn_total['n_total'] >= args.min_gpc][['motif_id','read_id']]
            dn_counts = dn_counts.merge(dnkeep.assign(_keep=1), on=['motif_id','read_id'], how='left')
            dn_counts = dn_counts[dn_counts['_keep'] == 1].drop(columns=['_keep'])
        dn_bound = (dn_counts.groupby(['motif_id','read_id'])['status']
                    .apply(lambda s: 1 if 'U' in s.values else 0).reset_index(name='bound'))
        dn_bound['ZF'] = 12

        # Append NDR rows and keep nucleosome-excluded pairs excluded
        bound_df = pd.concat([bound_df, up_bound, dn_bound], ignore_index=True)
        bound_df = _drop_pairs_safe(bound_df, nuc_excluded)
    # >>> add the snapshot here <<<
    kept_core_for_categories = (
    bound_df.groupby(['motif_id','read_id','ZF'], as_index=False)['bound'].max())

    # ── Cluster remapping (after NDR handling)
    if args.use_5clusters:
        def map_cluster(z):
            if 1 <= z <= 2: return 1
            if 3 <= z <= 4: return 2
            if 5 <= z <= 7: return 3
            if 8 <= z <= 9: return 4
            if 10 <= z <= 11: return 5
            return z
        bound_df['ZF'] = bound_df['ZF'].map(map_cluster)

    elif args.use_clusters:
        def map_cluster(z):
            if 1 <= z <= 3: return 1
            if 4 <= z <= 8: return 2
            if 9 <= z <= 11: return 3
            return z
        bound_df['ZF'] = bound_df['ZF'].map(map_cluster)

    # ✅ Always ensure one row per (motif, read, ZF/cluster) after any NDR/cluster operations
    bound_df = (bound_df
                .groupby(['motif_id', 'read_id', 'ZF'], as_index=False)['bound']
                .max())

    total = bound_df['motif_id'].nunique()
    
    # ---- Per-motif binding probability (single source of truth) ----
    # Build a pre-exclusion table aligned to the SAME ZF coding we applied to bound_df
    pre_nuc_bound_adj = pre_nuc_bound.copy()

    if args.use_5clusters:
        def _map5(z):
            if 1 <= z <= 2: return 1
            if 3 <= z <= 4: return 2
            if 5 <= z <= 7: return 3
            if 8 <= z <= 9: return 4
            if 10 <= z <= 11: return 5
            return z
        pre_nuc_bound_adj['ZF'] = pre_nuc_bound_adj['ZF'].map(_map5)
    elif args.use_clusters:
        def _map3(z):
            if 1 <= z <= 3: return 1
            if 4 <= z <= 8: return 2
            if 9 <= z <= 11: return 3
            return z
        pre_nuc_bound_adj['ZF'] = pre_nuc_bound_adj['ZF'].map(_map3)

    # One row per (motif, read, ZF/cluster) PRE-exclusion as well
    pre_nuc_bound_adj = (pre_nuc_bound_adj
                         .groupby(['motif_id','read_id','ZF'], as_index=False)['bound']
                         .max())
                         
    # Denominator (total) = #reads per motif×ZF BEFORE nucleosome exclusion
    # Numerator/denominators
    kept_grp = bound_df.groupby(['motif_id','ZF'])['bound']
    num_bound_kept = kept_grp.sum().rename('num_bound_kept')
    den_kept       = kept_grp.size().rename('den_kept')

    # Pre-exclusion denominator aligned to same ZF coding
    den_total = (pre_nuc_bound_adj.groupby(['motif_id','ZF'])['read_id']
                 .nunique().rename('den_total'))

    bpm_tbl = (num_bound_kept.to_frame()
               .join(den_kept, how='outer')
               .join(den_total, how='outer'))

    # NDR bins may not exist pre-exclusion; fall back to kept denom there
    bpm_tbl[['den_total','den_kept','num_bound_kept']] = bpm_tbl[['den_total','den_kept','num_bound_kept']].fillna(0)

    # If using total denominator, guarantee total ≥ kept (logical: total is pre-filter superset)
    if args.occupancy_denominator == 'total':
        bpm_tbl['den_total'] = np.maximum(bpm_tbl['den_total'], bpm_tbl['den_kept'])
        den_col = 'den_total'
    else:
        den_col = 'den_kept'

    # Final safety checks
    assert (bpm_tbl['num_bound_kept'] <= bpm_tbl[den_col]).all(), \
           "Numerator exceeds denominator even after dedup—please report."
    assert (bpm_tbl['num_bound_kept'].between(0, bpm_tbl[den_col])).all(), \
           "Counts out of bounds."

    bpm_tbl['binding_prob'] = np.where(
        bpm_tbl[den_col] > 0,
        bpm_tbl['num_bound_kept'] / bpm_tbl[den_col],
        np.nan
    )
    # Keep full per-motif table (counts + binding_prob); use a slim view for plotting
    bpm_points = bpm_tbl.reset_index()  # columns: motif_id, ZF, num_bound_kept, den_kept, den_total, binding_prob
    bpm = bpm_points[['motif_id','ZF','binding_prob']]


    if total == 0:
        print("No molecules passed the combined filters. Try relaxing filters.")
        return

    # Build bins & labels
    if args.use_5clusters:
        core_bins = [1,2,3,4,5]
    elif args.use_clusters:
        core_bins = [1,2,3]
    else:
        core_bins = list(range(1,12))
    if args.hide_zf5:
        core_bins = [c for c in core_bins if c != (3 if args.use_5clusters else 5)]
    # Drop arbitrary ZFs/clusters from plotting/exports



    bins = []
    if args.include_ndr: bins.append(0)
    bins.extend(core_bins)
    if args.include_ndr: bins.append(12)
    



    labels = []
    for z in bins:
        if args.include_ndr and z == 0: labels.append('N_NDR')
        elif args.include_ndr and z == 12: labels.append('C_NDR')
        else:
            if args.use_5clusters:
                labels.append({1:"ZF1-2",2:"ZF3-4",3:"ZF5-7",4:"ZF8-9",5:"ZF10-11"}.get(z, f"ZF{z}"))
            elif args.use_clusters:
                labels.append({1:"Cluster_I",2:"Cluster_II",3:"Cluster_III"}.get(z, f"{z}"))
            else:
                labels.append(f"{z}")
    
    

    # Optional: export the exact points used in the scatter
    if getattr(args, 'export_scatter_points', None):
        label_map = {int(z): lab for z, lab in zip(bins, labels)}
        points = bpm_points[bpm_points['ZF'].isin(bins)].copy()
        points['label'] = points['ZF'].map(label_map)
        points['denominator_used'] = 'total' if args.occupancy_denominator == 'total' else 'kept'
        # nice column order
        cols = ['motif_id','ZF','label','binding_prob','num_bound_kept','den_kept','den_total','denominator_used']
        cols = [c for c in cols if c in points.columns]  # keep only available cols
        points = points[cols]
        points.to_csv(args.export_scatter_points, sep='\t', index=False)
        print(f"Wrote scatter points to {args.export_scatter_points}")
    
    # Optional: export a wide matrix (rows=motifs, cols=plotted bins) of binding_prob
    if getattr(args, 'export_scatter_matrix', None):
        # map plotted bin IDs → axis labels actually used in this run
        label_map = {int(z): lab for z, lab in zip(bins, labels)}

        # use the per-motif long table you already computed for plotting
        # (works whether you kept bpm_points or not)
        mat_src = bpm[bpm['ZF'].isin(bins)].copy()
        mat_src['label'] = mat_src['ZF'].map(label_map)

        wide = (mat_src
                .pivot(index='motif_id', columns='label', values='binding_prob')
                .reindex(columns=labels))  # keep column order consistent with the x-axis

        # write TSV with motif_id as the first column
        out = wide.reset_index()
        out.to_csv(args.export_scatter_matrix, sep='\t', index=False)
        print(f"Wrote wide scatter matrix to {args.export_scatter_matrix} "
              f"(columns: {', '.join(out.columns[1:])})")

    

    # Summary stats & data for scatter
    summ = (bpm[bpm['binding_prob'].notna()]
        .groupby('ZF')['binding_prob']
        .agg(mean='mean', std='std'))
    
    summary_df = summ.reindex(bins).reset_index()
    data = [
    bpm.loc[(bpm['ZF'] == z) & bpm['binding_prob'].notna(), 'binding_prob'].values
    for z in bins
    ]
    positions = list(range(1, len(bins) + 1))
    plot_ns = [len(v) for v in data]
    print("n per bin (plot/export-consistent):", dict(zip(bins, plot_ns)))
    

    if args.run_qc:
        run_qc_checks(
            df.copy(),
            bound_df.copy(),
            pre_nuc_bound_adj.copy(),
            nuc_excluded.copy(),
            bpm.copy(),
            bins,
            labels,
            args
        )







    # Plot
    plot_zf_distribution(summary_df, data, positions, labels, total_motifs=total, args=args)
    if getattr(args, 'export_zf_stats', None):
        export_zf_stats(bpm, bins, labels, args.export_zf_stats)



    if args.make_scatter_heatmap:
        plot_zf_scatter_density_heatmap(data, positions, labels, summary_df, args)


    # Optional violin
    if args.make_violin:
        plot_zf_violin(bpm.copy(), bins, labels, args)

    # Optional pairwise stats + heatmap (always computed if any export flag is given)
    if args.export_pairwise_stats or args.export_pairwise_heatmap:
        export_pairwise_stats_and_heatmap(bpm.copy(), bins, labels, args)

    # Exports
    export_mappability_tables(df.copy(), args)
    export_zf_balance(df.copy(), args)
    
    if args.export_global_bound_summary:
        export_global_read_category_summary(
            bound_df.copy(), pre_nuc_bound.copy(), nuc_excluded.copy(),
            args.export_global_bound_summary
        )


    if args.export_scatter_stats:
        export_scatter_stats(
            bound_df.copy(), bpm.copy(), bins, args.export_scatter_stats,
            pre_nuc_bound.copy(), nuc_excluded.copy(), args,core_kept_precluster=kept_core_for_categories.copy()
        )
    
    # --- Dynamic fraction curve + exponential extrapolation (optional) ---
    if args.dynamic_curve_out:
        core_for_curve = (
            kept_core_for_categories if args.dynamic_curve_use == 'kept'
            else pre_core_for_totals
        )
        core_for_curve = core_for_curve[core_for_curve['ZF'].between(1, 11)].copy()

        # (optional) dynamic-specific ZF drops you already have
        val = getattr(args, 'exclude_zfs_est_dynamics', None)
        if val:
            excl = {int(x.strip()) for x in str(val).split(',') if x.strip()}
            if excl:
                core_for_curve = core_for_curve[~core_for_curve['ZF'].isin(excl)].copy()
                print(f"[dynamic-curve] excluding ZFs {sorted(excl)} from estimation only")

        print(f"[dynamic-curve] min_measured={args.dynamic_curve_min_measured}, mode={args.dynamic_curve_mode}, use={args.dynamic_curve_use}")

        export_and_plot_dynamic_curve(
            core_df=core_for_curve.copy(),
            out_prefix=args.dynamic_curve_out,
            mode=args.dynamic_curve_mode,
            Kmax=int(args.dynamic_curve_kmax),
            min_measured=int(args.dynamic_curve_min_measured)
        )



    # Global read-category counts (CTCF/nucleosome/nothing)
    if args.export_read_category_counts:
        total_core = pre_core_for_totals[pre_core_for_totals['ZF'].between(1, 11)][
            ['motif_id','read_id','ZF','bound']
        ].copy()

        kept_core = kept_core_for_categories[kept_core_for_categories['ZF'].between(1, 11)][
            ['motif_id','read_id','ZF','bound']
        ].copy()

        export_read_category_counts(total_core, kept_core, nuc_excluded, args.export_read_category_counts)


if __name__ == '__main__':
    main()

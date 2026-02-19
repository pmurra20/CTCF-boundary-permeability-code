#!/usr/bin/env python3
import argparse
import sys
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from typing import List, Iterable


# Publication-friendly PDF text
mpl.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.family'] = 'Arial'


# ── Utils ────────────────────────────────────────────────────────────────────
def _parse_col_spec(spec: str, valid_cols: Iterable[int]) -> List[int]:
    """Parse '1-3' and/or '1,2,12' into a de-duplicated list of ints that exist in valid_cols."""
    if not spec:
        return []
    valid_set = set(int(v) for v in valid_cols)  # handles numpy.int64, etc.
    wanted = []
    for tok in spec.replace(' ', '').split(','):
        if not tok:
            continue
        if '-' in tok:
            a, b = tok.split('-')
            a, b = int(a), int(b)
            wanted.extend(range(min(a, b), max(a, b) + 1))
        else:
            wanted.append(int(tok))
    out, seen = [], set()
    for c in wanted:
        if (c in valid_set) and (c not in seen):
            seen.add(c)
            out.append(c)
    return out


def compute_rel_pos(df: pd.DataFrame) -> pd.Series:
    """Signed offset from motif center in bp; strand-aware."""
    sf = {'+': -1, '-': 1}
    return (df['call_pos'] - df['specific_pos']) * df['strand_motif'].map(sf).fillna(1).astype(int)

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

# ── BH FDR adjust ────────────────────────────────────────────────────────────
def bh_adjust(p_vals):
    """
    Benjamini–Hochberg FDR adjust a list-like of p-values (global across all tests).
    Returns a NumPy array of adjusted p-values in the original order.
    """
    p = np.asarray(p_vals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    p_sorted = p[order]
    adj = np.empty(m, dtype=float)
    # forward
    for i, pi in enumerate(p_sorted):
        adj[i] = min(pi * m / (i + 1), 1.0)
    # enforce monotonicity
    for i in range(m - 2, -1, -1):
        adj[i] = min(adj[i], adj[i + 1])
    p_adj = np.empty(m, dtype=float)
    p_adj[order] = adj
    return p_adj


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
    excl_df = pd.DataFrame(excl_rows, columns=['motif_id','read_id','side'])
    return bd.loc[keep_idx].reset_index(), excl_df


# ── Signals I/O ──────────────────────────────────────────────────────────────
def load_signals(beds, labels):
    """
    Each BED: chr, start, end, strand, signal
    Returns a single DataFrame with columns [motif_id, subunit, signal].
    """
    dfs = []
    for bed, lab in zip(beds, labels):
        df = pd.read_csv(bed, sep='\t', header=None,
                         names=['chr','start','end','strand','signal'])
        df['motif_id'] = (
            df['chr'].astype(str) + '_' +
            df['start'].astype(str) + '_' +
            df['end'].astype(str) + '_' +
            df['strand'].astype(str)
        )
        df['subunit'] = lab
        dfs.append(df[['motif_id','subunit','signal']])
    return pd.concat(dfs, ignore_index=True)


# ── Core pipeline to build per-motif binding probabilities with filters ──────
def build_binding_prob_table(calls_tsv, args):
    """
    Implements the same inclusion/exclusion logic as your scatter script,
    but returns only a per-motif per-bin (ZF/cluster/NDR) binding_prob table.

    Returns:
      bpm: DataFrame [motif_id, ZF, binding_prob]
      bins: list of bins used (order for outputs/plots)
      labels: list of human-readable labels for bins
    """
    # Load
    df = pd.read_csv(calls_tsv, sep='\t', header=None, engine='python')
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
    nuc_active = (args.exclude_nucleosome_up or args.exclude_nucleosome_down)
    print(f"Nucleosome exclusion {'ACTIVE' if nuc_active else 'SKIPPED'} "
      f"(include_ndr={args.include_ndr}, up={args.exclude_nucleosome_up}, "
      f"down={args.exclude_nucleosome_down}, window=[{args.nuc_ndr_lo},{args.nuc_ndr_hi}])")

    pre_nuc_bound = bound_df.copy()  # snapshot for 'total' denominator later

    if nuc_active:
        bound_df, nuc_excluded = drop_nucleosome_reads(df.copy(), bound_df, args)
    else:
        nuc_excluded = pd.DataFrame(columns=['motif_id','read_id','side'])
    
  # --- NEW: keep only static/dynamic pairs on core ZFs (post-exclusion, pre-NDR/cluster) ---
    if args.pair_filter != "all":
        core = (bound_df[bound_df['ZF'].between(1, 11)]
                [['motif_id','read_id','ZF','bound']].drop_duplicates())
        if not core.empty:
            by_pair = (core.groupby(['motif_id','read_id'])['bound']
                            .agg(['sum','count']).reset_index())

            # optional measurability gate
            k = int(getattr(args, 'pair_filter_meas_k', 0))
            if k > 0:
                if getattr(args, 'pair_filter_meas_exact', False):
                    by_pair = by_pair[by_pair['count'] == k]
                else:
                    by_pair = by_pair[by_pair['count'] >= k]

            if args.pair_filter == "static":
                # static_all_bound: all measured core ZFs are bound (and count>0)
                mask = (by_pair['count'] > 0) & (by_pair['sum'] == by_pair['count'])
            else:  # "dynamic"
                # dynamic_mixture: at least one measured unbound; and at least one bound
                mask = (by_pair['sum'] > 0) & (by_pair['sum'] < by_pair['count'])

            keep_pairs = by_pair.loc[mask, ['motif_id','read_id']]
            if keep_pairs.empty:
                return pd.DataFrame(columns=['motif_id','ZF','binding_prob']), [], []

            # Always restrict the kept (numerator)
            bound_df = bound_df.merge(keep_pairs, on=['motif_id','read_id'], how='inner')

            # Only restrict the "total" denominator when the user chose kept;
            # leave it unfiltered for --occupancy_denominator total.
            if args.occupancy_denominator == 'kept':
                pre_nuc_bound = pre_nuc_bound.merge(keep_pairs, on=['motif_id','read_id'], how='inner')
            else:
                print("pair_filter active; TOTAL denominator left unfiltered (pre-exclusion universe).")

            if args.pair_filter == "static" and args.occupancy_denominator == "kept":
                print("Note: static + kept → binding_prob = 1 wherever measured; correlations may be degenerate.")

        # --- OPTIONAL cohesin windows on both sides (pseudo-bins 13,14) ---
    
    
    if getattr(args, 'include_cohesin', False):
        lo = int(getattr(args, 'cohesin_lo', 16))
        hi = int(getattr(args, 'cohesin_hi', 30))
        if lo > hi:
            lo, hi = hi, lo

        def _window_to_bound(sub_df: pd.DataFrame) -> pd.DataFrame:
            """status→counts→min_gpc→bound (per motif_id, read_id)"""
            if sub_df.empty:
                return pd.DataFrame(columns=['motif_id','read_id','bound'])
            counts = (sub_df.groupby(['motif_id','read_id','status'])['status']
                      .count().reset_index(name='n_calls'))
            if args.min_gpc > 0:
                totals = (counts.groupby(['motif_id','read_id'])['n_calls']
                          .sum().reset_index(name='n_total'))
                keep = totals[totals['n_total'] >= args.min_gpc][['motif_id','read_id']]
                counts = counts.merge(keep.assign(_keep=1), on=['motif_id','read_id'], how='left')
                counts = counts[counts['_keep'] == 1].drop(columns=['_keep'])
            if counts.empty:
                return pd.DataFrame(columns=['motif_id','read_id','bound'])
            bound = (counts.groupby(['motif_id','read_id'])['status']
                     .apply(lambda s: 1 if ('U' in s.values) else 0)
                     .reset_index(name='bound'))
            return bound

        # Upstream (N side): rel_pos ∈ [−hi, −lo]
        nside = df[df['rel_pos'].between(-hi, -lo)]
        n_bound_pre = _window_to_bound(nside); n_bound_pre['ZF'] = 13  # for TOTAL denominator

        # Downstream (C side): rel_pos ∈ [+lo, +hi]
        cside = df[df['rel_pos'].between(lo, hi)]
        c_bound_pre = _window_to_bound(cside); c_bound_pre['ZF'] = 14  # for TOTAL denominator

        # Add to pre-exclusion universe (for den_total)
        pre_nuc_bound = pd.concat([pre_nuc_bound, n_bound_pre, c_bound_pre], ignore_index=True)

        # Add to kept numerator and honor nucleosome-excluded pairs
        tmp_kept = pd.concat([n_bound_pre, c_bound_pre], ignore_index=True)
        bound_df = pd.concat([bound_df, tmp_kept], ignore_index=True)
        bound_df = _drop_pairs_safe(bound_df, nuc_excluded)
        # --- OPTIONAL flanking windows (immediate 3-bp tiles outside the motif) ---
    if getattr(args, 'include_flanks', False):
        k = int(getattr(args, 'flank_bins', 8))
        w = int(getattr(args, 'flank_width', 3))
        start = int(getattr(args, 'flank_start', 18))  # first base outside ZF1..ZF11

        def _win_to_bound(sub_df: pd.DataFrame) -> pd.DataFrame:
            if sub_df.empty:
                return pd.DataFrame(columns=['motif_id','read_id','bound'])
            counts = (sub_df.groupby(['motif_id','read_id','status'])['status']
                      .count().reset_index(name='n_calls'))
            if args.min_gpc > 0:
                totals = (counts.groupby(['motif_id','read_id'])['n_calls']
                          .sum().reset_index(name='n_total'))
                keep = totals[totals['n_total'] >= args.min_gpc][['motif_id','read_id']]
                counts = counts.merge(keep.assign(_keep=1),
                                      on=['motif_id','read_id'], how='left')
                counts = counts[counts['_keep'] == 1].drop(columns=['_keep'])
            if counts.empty:
                return pd.DataFrame(columns=['motif_id','read_id','bound'])
            bound = (counts.groupby(['motif_id','read_id'])['status']
                     .apply(lambda s: 1 if ('U' in s.values) else 0)
                     .reset_index(name='bound'))
            return bound

        n_list, c_list = [], []

        # N side bins: -1..-k (each w bp wide), nearest first (just outside motif)
        for i in range(1, k+1):
            hi_bp = -(start + (i-1)*w)            # e.g., -18, -21, ...
            lo_bp = hi_bp - (w-1)                 # inclusive lower bound
            sub = df[df['rel_pos'].between(lo_bp, hi_bp)]
            b = _win_to_bound(sub)
            b['ZF'] = -i                          # -1..-k
            n_list.append(b)

        # C side bins: +1..+k → internal codes 21..(20+k) to avoid ZF collisions
        for i in range(1, k+1):
            lo_bp =  start + (i-1)*w              # e.g., +18, +21, ...
            hi_bp =  lo_bp + (w-1)
            sub = df[df['rel_pos'].between(lo_bp, hi_bp)]
            b = _win_to_bound(sub)
            b['ZF'] = 20 + i                      # 21..(20+k) == +1..+k for labels
            c_list.append(b)

        # Add to TOTAL denominator universe (pre-exclusion)
        pre_nuc_bound = pd.concat([pre_nuc_bound, *n_list, *c_list], ignore_index=True)

        # Add to kept numerator and then honor nucleosome-excluded pairs
        tmp = pd.concat([*n_list, *c_list], ignore_index=True)
        bound_df = pd.concat([bound_df, tmp], ignore_index=True)
        bound_df = _drop_pairs_safe(bound_df, nuc_excluded)

        # Optional heads-up about potential overlap with cohesin/NDR windows
        if getattr(args, 'include_cohesin', False) and start <= int(getattr(args, 'cohesin_hi', 30)):
            print("Warning: flanks may overlap cohesin windows; bins are not mutually exclusive.")
        if getattr(args, 'include_ndr', False) and (start <= int(args.nuc_ndr_hi)):
            print("Warning: flanks may overlap NDR windows; bins are not mutually exclusive.")

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

    # ── Cluster remapping (after NDR handling) ──
    if args.use_5clusters:
        def map5(z):
            if 1 <= z <= 2: return 1
            if 3 <= z <= 4: return 2
            if 5 <= z <= 7: return 3
            if 8 <= z <= 9: return 4
            if 10 <= z <= 11: return 5
            return z
        bound_df['ZF'] = bound_df['ZF'].map(map5)
        pre_nuc_bound['ZF'] = pre_nuc_bound['ZF'].map(map5)
    elif args.use_clusters:
        def map3(z):
            if 1 <= z <= 3: return 1
            if 4 <= z <= 8: return 2
            if 9 <= z <= 11: return 3
            return z
        bound_df['ZF'] = bound_df['ZF'].map(map3)
        pre_nuc_bound['ZF'] = pre_nuc_bound['ZF'].map(map3)

    # Ensure one row per (motif, read, ZF/cluster)
    bound_df = (bound_df.groupby(['motif_id','read_id','ZF'], as_index=False)['bound'].max())
    pre_nuc_bound = (pre_nuc_bound.groupby(['motif_id','read_id','ZF'], as_index=False)['bound'].max())

    # Build bins & labels

    if args.use_5clusters:
        core_bins = [1,2,3,4,5]
    elif args.use_clusters:
        core_bins = [1,2,3]
    else:
        core_bins = list(range(1,12))
    if args.hide_zf5:
        core_bins = [c for c in core_bins if c != (3 if args.use_5clusters else 5)]

    # Flank codes (negative for N; 21..(20+k) for C so they don't collide with 1..11)
    n_flanks = ([-i for i in range(args.flank_bins, 0, -1)]  # -k..-1
                if getattr(args, 'include_flanks', False) else [])
    c_flanks = ([20 + i for i in range(1, args.flank_bins + 1)]  # 21..(20+k)
                if getattr(args, 'include_flanks', False) else [])

    # Final column order:
    # [N-flanks] [N_NDR] [N_COH] [core ZF/cluster] [C_COH] [C_NDR] [C-flanks]
    bins = []
    bins.extend(n_flanks)
    if args.include_ndr:     bins.append(0)    # N_NDR
    if args.include_cohesin: bins.append(13)   # N_COH
    bins.extend(core_bins)
    if args.include_cohesin: bins.append(14)   # C_COH
    if args.include_ndr:     bins.append(12)   # C_NDR
    bins.extend(c_flanks)

    labels = []
    for z in bins:
        if z == 0:         labels.append('N_NDR')
        elif z == 12:      labels.append('C_NDR')
        elif z == 13:      labels.append('N_COH')
        elif z == 14:      labels.append('C_COH')
        elif z < 0:        labels.append(str(z))                    # -1..-k
        elif z > 20:       labels.append(f"+{z-20}")               # 21.. → +1..
        else:
            if args.use_5clusters:
                labels.append({1:"ZF1-2",2:"ZF3-4",3:"ZF5-7",4:"ZF8-9",5:"ZF10-11"}.get(z, f"ZF{z}"))
            elif args.use_clusters:
                labels.append({1:"Cluster_I",2:"Cluster_II",3:"Cluster_III"}.get(z, f"{z}"))
            else:
                labels.append(f"ZF{z}")



    # Numerator (kept) and denominators (kept vs total) for occupancy
    kept_grp = bound_df.groupby(['motif_id','ZF'])['bound']
    num_bound_kept = kept_grp.sum().rename('num_bound_kept')
    den_kept = kept_grp.size().rename('den_kept')
    den_total = (pre_nuc_bound.groupby(['motif_id','ZF'])['read_id']
                 .nunique().rename('den_total'))

    bpm_tbl = (num_bound_kept.to_frame().join(den_kept, how='outer').join(den_total, how='outer')).fillna(0)
    if args.occupancy_denominator == 'total':
        bpm_tbl['den_total'] = np.maximum(bpm_tbl['den_total'], bpm_tbl['den_kept'])
        den_col = 'den_total'
    else:
        den_col = 'den_kept'

    bpm_tbl['binding_prob'] = np.where(
        bpm_tbl[den_col] > 0, bpm_tbl['num_bound_kept'] / bpm_tbl[den_col], np.nan
    )
    bpm = bpm_tbl.reset_index()[['motif_id','ZF','binding_prob']]

    # Restrict bpm to the plotted/used bins, preserving order
    bpm = bpm[bpm['ZF'].isin(bins)].copy()

    return bpm, bins, labels




# ── Correlate per-bin binding with signals across motifs ─────────────────────
def correlate_bpm_vs_signals(bpm, signals, zf_list):
    """
    bpm: [motif_id, ZF, binding_prob]
    signals: [motif_id, subunit, signal]
    zf_list: iterable of ZFs (bins) to test, in the order to report

    Returns:
      r_df:    DataFrame (rows=subunits, cols=ZFs) with Pearson r
      p_df:    same shape with raw p-values
      padj_df: same shape with BH-adjusted p-values (global correction over all cells)
    """
    subunits = list(pd.unique(signals['subunit']))   # preserves order of appearance
    r_mat = np.full((len(subunits), len(zf_list)), np.nan)
    p_mat = np.full_like(r_mat, np.nan, dtype=float)

    for i, sub in enumerate(subunits):
        sig_sub = signals[signals['subunit'] == sub][['motif_id', 'signal']]
        for j, zf in enumerate(zf_list):
            bm = bpm[bpm['ZF'] == zf][['motif_id', 'binding_prob']]
            merged = bm.merge(sig_sub, on='motif_id', how='inner').dropna()
            if len(merged) >= 2:
                r, p = pearsonr(merged['binding_prob'], merged['signal'])
            else:
                r, p = np.nan, np.nan
            r_mat[i, j] = r
            p_mat[i, j] = p

    # Global BH across all tests
    flat_p = p_mat.ravel()
    mask = np.isfinite(flat_p)
    flat_q = np.full_like(flat_p, np.nan, dtype=float)
    if mask.any():
        flat_q[mask] = bh_adjust(flat_p[mask])
    q_mat = flat_q.reshape(p_mat.shape)

    r_df = pd.DataFrame(r_mat, index=subunits, columns=zf_list)
    p_df = pd.DataFrame(p_mat, index=subunits, columns=zf_list)
    padj_df = pd.DataFrame(q_mat, index=subunits, columns=zf_list)
    return r_df, p_df, padj_df


# ── Plot heatmaps (optional) ─────────────────────────────────────────────────
def plot_heatmaps(r_df, padj_df, labels_by_zf, out_prefix, r_vmin=-1.0, r_vmax=1.0,
                  pmin=0.0, pmax=None):
    # Order/labels
    zf_list = list(r_df.columns)
    col_labels = [labels_by_zf.get(z, f"ZF{z}") for z in zf_list]

    # Pearson r
    fig1, ax1 = plt.subplots(figsize=(1.2*len(zf_list), 1 + 0.7*len(r_df.index)))
    im1 = ax1.imshow(r_df.values, cmap='coolwarm_r', vmin=r_vmin, vmax=r_vmax, aspect='auto')
    ax1.set_xticks(np.arange(len(zf_list)))
    ax1.set_yticks(np.arange(len(r_df.index)))
    ax1.set_xticklabels(col_labels, rotation=45, ha='right')
    ax1.set_yticklabels(r_df.index)
    ax1.set_xlabel("Bin (ZF/cluster/NDR)")
    ax1.set_ylabel("Subunit")
    ax1.set_title("Pearson r: binding vs signal")
    fig1.colorbar(im1, ax=ax1, label='Pearson r')
    fig1.tight_layout()
    fig1.savefig(f"{out_prefix}_pearsonr_heatmap.pdf", dpi=600)
    plt.close(fig1)

    # -log10(q)
    neglogq = -np.log10(np.clip(padj_df.values, 1e-300, 1))
    vmax = pmax if pmax is not None else (np.nanmax(neglogq) if np.isfinite(neglogq).any() else 1.0)
    fig2, ax2 = plt.subplots(figsize=(1.2*len(zf_list), 1 + 0.7*len(r_df.index)))
    im2 = ax2.imshow(neglogq, cmap='gray_r', vmin=pmin, vmax=vmax, aspect='auto')
    ax2.set_xticks(np.arange(len(zf_list)))
    ax2.set_yticks(np.arange(len(r_df.index)))
    ax2.set_xticklabels(col_labels, rotation=45, ha='right')
    ax2.set_yticklabels(r_df.index)
    ax2.set_xlabel("Bin (ZF/cluster/NDR)")
    ax2.set_ylabel("Subunit")
    ax2.set_title("-log10(q): BH-adjusted p-values")
    fig2.colorbar(im2, ax=ax2, label='-log10(q)')
    fig2.tight_layout()
    fig2.savefig(f"{out_prefix}_neglogq_heatmap.pdf", dpi=600)
    plt.close(fig2)


# ── CLI ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Correlate per-bin binding probabilities (after full inclusion/exclusion "
                    "logic) against one or more motif-centered signal tracks; output r/p/q tables "
                    "and optional heatmaps."
    )
    ap.add_argument("-i", "--input", required=True,
                    help="Calls-vs-motif intersect TSV (see expected columns in code).")
    ap.add_argument("--signals", nargs="+", required=True, metavar="BED",
                    help="One or more BED/TSV files: chr start end strand signal")
    ap.add_argument("--labels", nargs="+", required=True, metavar="LABEL",
                    help="Labels for the --signals, same length/order.")
    ap.add_argument("--out-prefix", required=True, help="Prefix for outputs (TSVs, PDFs).")

    # ── Inclusion/exclusion flags borrowed from scatter pipeline ──
    ap.add_argument("--min_gpc", type=int, default=0, help="Min calls per (motif,read,ZF) to keep")
    ap.add_argument("--mol_threshold", type=int, default=1, help="Min distinct bound core ZFs per read (0=no filter)")
    ap.add_argument("--exact", action="store_true", help="Use exact==threshold instead of >= threshold")
    ap.add_argument("--focus_zf_range", type=str, help='Range "a-b" or list "a,b,c" of ZFs for focus filter')
    ap.add_argument("--focus_zf_min", type=int, default=1, help="Min # bound ZFs within focus range to keep a read")
    ap.add_argument("--drop_all_bound", action="store_true", help="Drop (motif,read) with all measured ZFs bound")
    ap.add_argument("--use_clusters", action="store_true", help="Collapse to 3 clusters: 1-3, 4-8, 9-11")
    ap.add_argument("--use_5clusters", action="store_true", help="Collapse to 5 clusters: 1-2,3-4,5-7,8-9,10-11")
    ap.add_argument("--hide_zf5", action="store_true", help="Hide ZF5 (or cluster 3 in 5-cluster mode)")
    ap.add_argument("--include_ndr", action="store_true", help="Include upstream (ZF=0) and downstream (ZF=12) NDR bins")
    ap.add_argument("--exclude_nucleosome_up", action="store_true",
                    help="Exclude reads with ALL CpGs bound (U) in upstream NDR window [ -nuc_ndr_hi, -nuc_ndr_lo ]")
    ap.add_argument("--exclude_nucleosome_down", action="store_true",
                    help="Exclude reads with ALL CpGs bound (U) in downstream NDR window [  nuc_ndr_lo,  nuc_ndr_hi ]")
    ap.add_argument("--nuc_ndr_lo", type=int, default=40, help="Lower absolute bound (bp) for NDR windows")
    ap.add_argument("--nuc_ndr_hi", type=int, default=60, help="Upper absolute bound (bp) for NDR windows")
    ap.add_argument(
        '--occupancy-denominator', '--occupancy_denominator',
        dest='occupancy_denominator',
        choices=['kept','total'],
        default='kept',
        help="Denominator for binding fraction: 'kept' (post-exclusion) or 'total' (pre-exclusion)."
    )

    # Plot options (same defaults as before)
    ap.add_argument("--plot", action="store_true", help="Write two heatmaps (r and -log10 q).")
    ap.add_argument("--r_vmin", type=float, default=-1.0)
    ap.add_argument("--r_vmax", type=float, default=1.0)
    ap.add_argument("--pneglog_vmin", type=float, default=0.0)
    ap.add_argument("--pneglog_vmax", type=float, default=None)
    ap.add_argument(
    "--cohesin_sort_cols",
    type=str,
    default=None,
    help='Sort rows (subunits) by the mean Pearson r across these ZF columns; '
         'accepts "a-b" ranges and/or comma lists, e.g. "1-3" or "0,1-3,12".')
    
    ap.add_argument(
        "--pair_filter", choices=["all", "static", "dynamic"], default="all",
        help="Restrict correlation to only static pairs (all measured core ZFs bound), only dynamic pairs (mix of bound/unbound), or all pairs.")
    ap.add_argument(
        "--pair_filter_meas_k", type=int, default=0,
        help="When applying --pair_filter, require at least K measurable core ZFs (1..11) on the read. 0 disables this requirement.")
    ap.add_argument(
        "--pair_filter_meas_exact", action="store_true",
        help="With --pair_filter_meas_k, use exactly K measurable core ZFs instead of >=K.")

    ap.add_argument("--include_cohesin", action="store_true",
        help="Add cohesin-footprint windows as extra bins: 13=N_COH (-hi..-lo), 14=C_COH (+lo..+hi).")
    ap.add_argument("--cohesin_lo", type=int, default=16,
        help="Lower absolute bound (bp) for cohesin windows (default 16).")
    ap.add_argument("--cohesin_hi", type=int, default=30,
        help="Upper absolute bound (bp) for cohesin windows (default 30).")
    ap.add_argument("--include_flanks", action="store_true",
    help="Add 3-bp flanking bins outside the motif: N side = -1..-K, C side = +1..+K.")
    ap.add_argument("--flank_bins", type=int, default=8,
        help="Number of flanking bins per side (default 8).")
    ap.add_argument("--flank_width", type=int, default=3,
        help="Width (bp) of each flanking bin (default 3).")
    ap.add_argument("--flank_start", type=int, default=18,
        help="First bp offset outside motif (default 18 for 3-bp ZF binning).")

    
    args = ap.parse_args()
    if len(args.signals) != len(args.labels):
        ap.error("--signals and --labels must have equal length and order.")
    if args.use_clusters and args.use_5clusters:
        ap.error("Choose either --use_clusters or --use_5clusters, not both.")

    # Build per-motif binding probabilities with full filtering logic
    bpm, bins, labels = build_binding_prob_table(args.input, args)
    if bpm.empty:
        sys.exit("No data after filtering — nothing to correlate.")

    # Load and combine signals
    sig = load_signals(args.signals, args.labels)

    # Correlate (use bins as the column order)
    r_df, p_df, q_df = correlate_bpm_vs_signals(bpm, sig, bins)
    
    sort_cols = _parse_col_spec(args.cohesin_sort_cols, r_df.columns.tolist())
    if sort_cols:
        key = r_df[sort_cols].mean(axis=1, skipna=True)
        order = key.sort_values(ascending=False).index  # descending by mean r
        r_df = r_df.loc[order]
        p_df = p_df.loc[order]
        q_df = q_df.loc[order]
        print(f"Row order: sorted by mean r across columns {sort_cols} (descending).")

    # Write tidy outputs
    r_df.to_csv(f"{args.out_prefix}_pearsonr.tsv", sep="\t")
    p_df.to_csv(f"{args.out_prefix}_pvals.tsv", sep="\t")
    q_df.to_csv(f"{args.out_prefix}_padj.tsv",  sep="\t")
    print(f"Wrote:\n  {args.out_prefix}_pearsonr.tsv\n  {args.out_prefix}_pvals.tsv\n  {args.out_prefix}_padj.tsv")

    # Optional heatmaps (labels mapped per bin)
    if args.plot:
        labels_by_zf = {z: lab for z, lab in zip(bins, labels)}
        plot_heatmaps(
            r_df, q_df, labels_by_zf, args.out_prefix,
            r_vmin=args.r_vmin, r_vmax=args.r_vmax,
            pmin=args.pneglog_vmin, pmax=args.pneglog_vmax
        )


if __name__ == "__main__":
    main()


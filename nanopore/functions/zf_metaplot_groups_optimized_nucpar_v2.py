#!/usr/bin/env python3
import argparse
import os
import re
import sys
import time
import tempfile
from pathlib import Path
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from pandas.api.types import is_categorical_dtype

# ────────────────── Matplotlib defaults ──────────────────
mpl.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.family'] = 'Arial'

# ────────────────── utils ──────────────────
def now():
    return time.perf_counter()

def log(msg, enabled=True):
    if enabled:
        print(msg, flush=True)

def safe_save_pdf(fig, outpath, label="figure"):
    """Create parent dir, write to temp .pdf, atomic replace; fallback to /tmp on error."""
    import os, re, tempfile
    from pathlib import Path
    import matplotlib.pyplot as plt

    if not outpath:
        plt.show()
        return

    p = Path(outpath)
    if p.suffix.lower() != '.pdf':
        p = p.with_suffix('.pdf')

    try:
        if p.parent and not p.parent.exists():
            p.parent.mkdir(parents=True, exist_ok=True)

        # write to a real .pdf temp file in the target directory
        with tempfile.NamedTemporaryFile(prefix=p.stem + '_', suffix='.pdf', dir=str(p.parent), delete=False) as tmp:
            tmp_path = Path(tmp.name)
        fig.savefig(str(tmp_path), format='pdf')
        os.replace(str(tmp_path), str(p))
        print(f"Saved {label} to {p}")
    except Exception as e:
        # fallback to system temp dir (also .pdf extension)
        safe_name = re.sub(r'[^A-Za-z0-9_.-]+', '_', p.name)
        fallback = Path(tempfile.gettempdir()) / f"{int(time.time())}_{safe_name}"
        fig.savefig(str(fallback), format='pdf')
        print(f"Warning: {e}. Wrote fallback {label} to {fallback}")
    finally:
        plt.close(fig)

def make_motif_id(df):
    return (df['chr_motif'].astype(str) + '_' +
            df['start_motif'].astype(str) + '_' +
            df['end_motif'].astype(str) + '_' +
            df['strand_motif'].astype(str))

def strand_factor(s):
    """Robust strand → ±1 map that works with categoricals on old pandas."""
    return s.astype(str).map({'+': -1, '-': 1}).fillna(1).astype('int8')
    
def coalesce_specific_pos_inplace(df):
    """Ensure a plain 'specific_pos' column exists; drop any _x/_y leftovers."""
    if 'specific_pos' not in df.columns:
        if 'specific_pos_y' in df.columns:
            df['specific_pos'] = df['specific_pos_y'].astype('int64')
        elif 'specific_pos_x' in df.columns:
            df['specific_pos'] = df['specific_pos_x'].astype('int64')
        else:
            # fallback: midpoint
            df['specific_pos'] = ((df['start_motif'].astype('int64') +
                                   df['end_motif'].astype('int64')) // 2).astype('int64')
    # clean up merge suffixes if present
    for c in ('specific_pos_x','specific_pos_y'):
        if c in df.columns:
            df.drop(columns=c, inplace=True)

def flag_nucleosome_pairs(df_calls_with_motif, lo=40, hi=60, up=True, down=True):
    """
    Return a set of (motif_id, read_id) pairs flagged as nucleosomal based on flank windows:
      upstream = [-hi, -lo], downstream = [lo, hi].
    A pair is flagged if the window has ≥1 CpG call and ALL of those calls are 'U'.

    Robust to:
      • specific_pos_x/_y merge suffixes
      • missing/NaN specific_pos (uses midpoint fallback)
      • categorical groupers on old pandas (avoids cartesian explosion)
    """
    import numpy as np
    import pandas as pd
    from pandas.api.types import is_categorical_dtype

    if df_calls_with_motif is None or df_calls_with_motif.empty or not (up or down):
        return set()

    # Keep only needed rows/cols early
    need_cols = [
        'chr_motif', 'start_motif', 'end_motif', 'strand_motif',
        'read_id', 'call_pos', 'status'
    ]
    # We may or may not have plain 'specific_pos'; cope with _x/_y too.
    have = [c for c in need_cols + ['specific_pos', 'specific_pos_x', 'specific_pos_y'] if c in df_calls_with_motif.columns]
    df = df_calls_with_motif.loc[df_calls_with_motif['status'].isin(['M','U']), have].copy()
    if df.empty:
        return set()

    # Coalesce specific_pos → plain column
    if 'specific_pos' not in df.columns:
        if 'specific_pos_y' in df.columns:
            df['specific_pos'] = df['specific_pos_y']
        elif 'specific_pos_x' in df.columns:
            df['specific_pos'] = df['specific_pos_x']
    # If still missing or NaN, use midpoint fallback (when starts/ends are present)
    if 'specific_pos' not in df.columns or df['specific_pos'].isna().any():
        if all(c in df.columns for c in ['start_motif','end_motif']):
            mid = ((df['start_motif'].astype('int64') + df['end_motif'].astype('int64')) // 2)
            if 'specific_pos' in df.columns:
                df['specific_pos'] = pd.to_numeric(df['specific_pos'], errors='coerce').fillna(mid)
            else:
                df['specific_pos'] = mid
        else:
            # Cannot compute; drop rows with missing specific_pos
            df['specific_pos'] = pd.to_numeric(df.get('specific_pos'), errors='coerce')

    # Clean up any merge suffix remnants
    for c in ('specific_pos_x','specific_pos_y'):
        if c in df.columns:
            df.drop(columns=c, inplace=True)

    # Make sure we have everything needed
    req = ['chr_motif','start_motif','end_motif','strand_motif','read_id','call_pos','specific_pos','status']
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError("flag_nucleosome_pairs: missing required columns: " + ", ".join(missing))

    # Drop rows with NA call_pos / specific_pos to avoid int casts failing
    df['call_pos'] = pd.to_numeric(df['call_pos'], errors='coerce')
    df['specific_pos'] = pd.to_numeric(df['specific_pos'], errors='coerce')
    df = df.dropna(subset=['call_pos','specific_pos'])
    if df.empty:
        return set()

    df['call_pos'] = df['call_pos'].astype('int64')
    df['specific_pos'] = df['specific_pos'].astype('int64')

    # Strand-aware rel position; robust mapping even if categorical
    sgn = df['strand_motif'].astype(str).map({'+': -1, '-': 1}).fillna(1).astype('int8')
    rel = ((df['call_pos'] - df['specific_pos']) * sgn).astype('int32')

    # Absolute-distance prefilter: only rows within ±hi matter
    close = (df['call_pos'] - df['specific_pos']).abs() <= int(hi)
    if not close.any():
        return set()
    df = df.loc[close, ['chr_motif','start_motif','end_motif','strand_motif','read_id','status']]

    # Build masks for requested flanks on the *filtered* index
    mask = pd.Series(False, index=df.index)
    if up:
        mask |= rel.loc[df.index].between(-int(hi), -int(lo))
    if down:
        mask |= rel.loc[df.index].between(int(lo), int(hi))
    if not mask.any():
        return set()

    sub = df.loc[mask].copy()
    sub['isU'] = (sub['status'] == 'U').astype('uint8')

    # ---- SAFE GROUPBY (avoid categorical cartesian explosion) ----
    keys = ['chr_motif','start_motif','end_motif','strand_motif','read_id']
    tmp = sub.copy()
    for k in keys:
        if is_categorical_dtype(tmp[k]):
            tmp[k] = tmp[k].astype(str)

    g = tmp.groupby(keys, sort=False)['isU']
    cnt = g.size().reset_index(name='n')
    su  = g.sum().reset_index(name='u')

    agg = cnt.merge(su, on=keys, how='inner')
    ok = agg.loc[(agg['n'] >= 1) & (agg['u'] == agg['n']), keys]

    if ok.empty:
        return set()

    # Convert to motif_id + read_id pairs
    ok['motif_id'] = (ok['chr_motif'].astype(str) + '_' +
                      ok['start_motif'].astype(str) + '_' +
                      ok['end_motif'].astype(str) + '_' +
                      ok['strand_motif'].astype(str))
    return set(zip(ok['motif_id'].tolist(), ok['read_id'].tolist()))



# ────────────────── memory-safe groupby helpers ──────────────────
def _groupby_size_safe(df, keys, sort=False):
    """
    Fast groupby size that avoids categorical cartesian explosions.
    Tries observed=True; if pandas too old, casts categoricals to string.
    """
    try:
        return (df.groupby(keys, observed=True, sort=sort)
                  .size()
                  .reset_index(name='n'))
    except TypeError:
        tmp = df.copy()
        for k in keys:
            if k in tmp.columns and is_categorical_dtype(tmp[k]):
                tmp[k] = tmp[k].astype(str)
        return (tmp.groupby(keys, sort=sort)
                   .size()
                   .reset_index(name='n'))

def _groupby_agg_safe(df, keys, value, agg='max', sort=False, nunique_col=None):
    """
    Memory-safe groupby aggregation with observed=True, falling back
    to casting categoricals to string when needed.
    """
    try:
        if agg == 'max':
            gb = df.groupby(keys, observed=True, sort=sort)[value].max()
            return gb.reset_index()
        elif agg == 'nunique':
            gb = df.groupby(keys, observed=True, sort=sort)[nunique_col].nunique()
            return gb.reset_index(name='n')
        else:
            raise ValueError("Unsupported agg")
    except TypeError:
        tmp = df.copy()
        for k in keys:
            if k in tmp.columns and is_categorical_dtype(tmp[k]):
                tmp[k] = tmp[k].astype(str)
        if agg == 'max':
            gb = tmp.groupby(keys, sort=sort)[value].max()
            return gb.reset_index()
        elif agg == 'nunique':
            gb = tmp.groupby(keys, sort=sort)[nunique_col].nunique()
            return gb.reset_index(name='n')

def parse_xwin(s: str):
    """Parse 'a:b' (ints, bp) and return (a,b) ordered."""
    if not s:
        return None
    m = re.match(r'^\s*(-?\d+)\s*:\s*(-?\d+)\s*$', s)
    if not m:
        raise argparse.ArgumentTypeError("Expected format 'a:b', e.g. -200:400")
    a, b = int(m.group(1)), int(m.group(2))
    if a > b:
        a, b = b, a
    return (a, b)



# ────────────────── motif I/O ──────────────────
def read_motif_bed(path):
    """Read motif bed; if specific_pos not present, use midpoint."""
    m = pd.read_csv(path, sep='\t', header=None, engine='c',
                    usecols=[0,1,2,3], dtype={0:'category',1:'int64',2:'int64',3:'category'})
    m.columns = ['chr_motif','start_motif','end_motif','strand_motif']
    try:
        sp = pd.read_csv(path, sep='\t', header=None, engine='c', usecols=[4], dtype={4:'float64'})
        sp = pd.to_numeric(sp.iloc[:,0], errors='coerce')
    except Exception:
        sp = pd.Series([], dtype='float64')
    if len(sp) == len(m):
        m['specific_pos'] = sp
    else:
        m['specific_pos'] = np.nan
    center = (m['start_motif'] + m['end_motif']) // 2
    m['specific_pos'] = m['specific_pos'].fillna(center).astype('int64')
    return m[['chr_motif','start_motif','end_motif','strand_motif','specific_pos']].copy()

# ────────────────── calls I/O ──────────────────
def detect_calls_format(path):
    """Return 'intersect' (16+ cols) or 'wide' (10 cols) based on column count."""
    first = pd.read_csv(path, sep='\t', header=None, engine='c', nrows=5)
    ncol = first.shape[1]
    if ncol >= 16:
        return 'intersect'
    elif ncol == 10:
        return 'wide'
    else:
        return 'wide' if ncol <= 12 else 'intersect'

def load_calls_intersect(path, chroms=None, profile=False):
    t0 = now()
    usecols = list(range(16))
    dtypes = {
        0:'category', 1:'int64', 2:'int64', 3:'int64', 4:'category',
        5:'category', 6:'float32', 7:'float32', 8:'float32', 9:'category',
        10:'category', 11:'int64', 12:'int64', 13:'category',
        14:'category', 15:'int64',
    }
    df = pd.read_csv(path, sep='\t', header=None, engine='c', usecols=usecols, dtype=dtypes, memory_map=True)
    df.columns = [
        'chr_call','start_call','end_call','call_pos','strand_call',
        'read_id','llr_ratio','llr_met','llr_unmet','status',
        'chr_motif','start_motif','end_motif','strand_motif',
        'motif_seq','specific_pos'
    ]
    if chroms is not None:
        keep = set(chroms)
        df = df[df['chr_motif'].astype(str).isin(keep)]
    log(f"[IO] intersect read {len(df):,} rows in {now()-t0:.2f}s", profile)
    return df

def load_calls_wide(path, max_rows=0, chroms=None, profile=False):
    t0 = now()
    dtypes = {
        0:'category', 1:'int64', 2:'int64', 3:'int64', 4:'category',
        5:'category', 6:'float32', 7:'float32', 8:'float32', 9:'category'
    }
    kwargs = dict(sep='\t', header=None, engine='c', dtype=dtypes, memory_map=True)
    if max_rows and max_rows > 0:
        kwargs['nrows'] = max_rows
    df = pd.read_csv(path, **kwargs)
    if chroms is not None:
        keep = set(chroms)
        df = df[df.iloc[:,0].astype(str).isin(keep)]
    log(f"[IO] wide read {len(df):,} rows in {now()-t0:.2f}s", profile)
    return df

def align_calls_to_motifs_wide(df_calls, motif_df, assign_window, chroms=None, profile=False):
    """
    Map 'wide' calls (no motif columns) to nearest motif center within assign_window.
    Per-chrom merge_asof on sorted arrays.
    """
    t0 = now()
    motif_df = motif_df.copy()
    if chroms is not None:
        keep = set(chroms)
        motif_df = motif_df[motif_df['chr_motif'].astype(str).isin(keep)]
    motif_df['specific_pos'] = motif_df['specific_pos'].astype('int64')

    calls = df_calls.copy()
    calls.columns = [
        'chr_call','start_call','end_call','call_pos','strand_call',
        'read_id','llr_ratio','llr_met','llr_unmet','status'
    ]
    if chroms is not None:
        keep = set(chroms)
        calls = calls[calls['chr_call'].astype(str).isin(keep)]
    calls['call_pos'] = calls['call_pos'].astype('int64')

    out = []
    for chrom, left in calls.groupby('chr_call', sort=False):
        right = motif_df[motif_df['chr_motif'] == chrom]
        if right.empty:
            continue
        left = left.sort_values('call_pos', kind='mergesort')
        right = right.sort_values('specific_pos', kind='mergesort')
        merged = pd.merge_asof(
            left, right,
            left_on='call_pos', right_on='specific_pos',
            direction='nearest', tolerance=assign_window
        )
        merged = merged.dropna(subset=['specific_pos'])
        out.append(merged)
    if not out:
        cols = [
            'chr_call','start_call','end_call','call_pos','strand_call',
            'read_id','llr_ratio','llr_met','llr_unmet','status',
            'chr_motif','start_motif','end_motif','strand_motif','specific_pos'
        ]
        df = pd.DataFrame(columns=cols)
    else:
        df = pd.concat(out, ignore_index=True)
    log(f"[ASSIGN] wide→motif assigned {len(df):,} rows in {now()-t0:.2f}s", profile)
    return df

# ────────────────── core logic ──────────────────
def _prep_core(df_calls):
    """Filter to M/U, compute bound, rel_pos, ZF; keep core ZFs only."""
    df = df_calls[df_calls['status'].isin(['M', 'U'])].copy()
    df['bound'] = (df['status'] == 'U').astype('int8')
    sf = strand_factor(df['strand_motif'])
    df['rel_pos'] = ((df['call_pos'].astype('int64') - df['specific_pos'].astype('int64')) * sf).astype('int64')
    df['ZF'] = (6 + np.floor_divide(df['rel_pos'], 3)).astype('int8')
    df_core = df[(df['ZF'] >= 1) & (df['ZF'] <= 11)].copy()
    return df_core

def _per_chrom_bound_df(args_tuple):
    """
    Worker for per-chrom processing:
      - optional min_gpc filter
      - build motif_id
      - collapse to (motif_id, read_id, ZF) -> bound (max)
      - optional mol_threshold filter
    Returns bound_df for that chromosome.
    """
    chrom, df_core_sub, min_gpc, mol_threshold, exact = args_tuple

    # min_gpc (per (chr,start,end,read,ZF))
    if min_gpc > 0 and not df_core_sub.empty:
        gcols = ['chr_motif','start_motif','end_motif','read_id','ZF']
        counts = _groupby_size_safe(df_core_sub[gcols], gcols, sort=False).rename(columns={'n':'n_calls'})
        keep = counts[counts['n_calls'] >= min_gpc][gcols]
        df_core_sub = df_core_sub.merge(keep.assign(_keep=1), on=gcols, how='inner')

    if df_core_sub.empty:
        return pd.DataFrame(columns=['motif_id','read_id','ZF','bound'])

    df_core_sub['motif_id'] = make_motif_id(df_core_sub)

    # Collapse to one bound per (motif_id, read_id, ZF)
    bound_df = _groupby_agg_safe(df_core_sub,
                                 keys=['motif_id','read_id','ZF'],
                                 value='bound', agg='max',
                                 sort=False)

    # mol_threshold filter on distinct bound ZFs per read
    if mol_threshold > 0 and not bound_df.empty:
        bound_only = bound_df[bound_df['bound'] == 1]
        bc = _groupby_agg_safe(bound_only,
                               keys=['motif_id','read_id'],
                               value=None, agg='nunique',
                               sort=False, nunique_col='ZF').rename(columns={'n':'n_bound'})
        if exact:
            keep_reads = bc[bc['n_bound'] == mol_threshold][['motif_id','read_id']]
        else:
            keep_reads = bc[bc['n_bound'] >= mol_threshold][['motif_id','read_id']]
        bound_df = bound_df.merge(keep_reads.assign(_keep=1), on=['motif_id','read_id'], how='inner')

    return bound_df[['motif_id','read_id','ZF','bound']]

def build_pivot_parallel(df_calls, min_gpc=0, mol_threshold=0, exact=False, drop_all_bound=False,
                         jobs=1, profile=False):
    """
    Build per-(motif_id, read_id, ZF) pivot (ZF1..ZF11) with parallel per-chrom aggregation.
    """
    if df_calls is None or len(df_calls) == 0:
        return pd.DataFrame()

    t0 = now()
    # Precompute core fields once
    df_core = _prep_core(df_calls)
    log(f"[PIVOT] prepped core ZFs rows={len(df_core):,} in {now()-t0:.2f}s", profile)

    if df_core.empty:
        return pd.DataFrame()

    # Partition by chromosome of motif (safe; motif_id includes chr)
    parts = []
    for chrom, sub in df_core.groupby('chr_motif', sort=False):
        parts.append((chrom, sub, min_gpc, mol_threshold, exact))

    # Parallel map
    t1 = now()
    if jobs is None or jobs < 1:
        jobs = 1
    jobs = min(jobs, len(parts)) if parts else 1
    if jobs == 1:
        chunk_results = [_per_chrom_bound_df(tup) for tup in parts]
    else:
        # On macOS, set start method to 'fork' via CLI env if needed
        with Pool(processes=jobs) as pool:
            chunk_results = pool.map(_per_chrom_bound_df, parts)
    log(f"[PIVOT] per-chrom aggregation in {now()-t1:.2f}s using {jobs} job(s)", profile)

    # Concatenate all bound_df chunks
    if not chunk_results:
        return pd.DataFrame()
    bound_df = pd.concat(chunk_results, ignore_index=True)

    # Pivot (no aggregation) — robust across old pandas
    t2 = now()
    # Ensure unique (motif_id, read_id, ZF) triples
    bound_df = bound_df.drop_duplicates(['motif_id','read_id','ZF'])
    pivot = (bound_df
             .set_index(['motif_id','read_id','ZF'])['bound']
             .unstack('ZF'))
    pivot = pivot.reindex(columns=sorted(pivot.columns))

    if drop_all_bound and not pivot.empty:
        nonna = pivot.notna()
        all_bound = ((pivot == 1) & nonna).sum(axis=1) == nonna.sum(axis=1)
        pivot = pivot.loc[~all_bound]
    log(f"[PIVOT] built pivot {pivot.shape[0]:,}×{pivot.shape[1]:,} in {now()-t2:.2f}s (total {now()-t0:.2f}s)", profile)
    return pivot

def select_group_ids(pivot, group):
    """Return index tuples (motif_id, read_id) matching logical ZF patterns."""
    if pivot is None or pivot.empty:
        return []
    cols = {zf: (zf in pivot.columns) for zf in range(1, 12)}
    def col(z): return pivot[z] if cols.get(z, False) else pd.Series(np.nan, index=pivot.index)

    if group == 'N_open':  # (ZF1 or ZF2 unbound=0) AND (ZF6 or ZF7 bound=1)
        mask = ((col(1) == 0) | (col(2) == 0)) & ((col(6) == 1) | (col(7) == 1))
    elif group == 'C_open':  # (ZF10 or ZF11 unbound=0) AND (ZF6 or ZF7 bound=1)
        mask = ((col(10) == 0) | (col(11) == 0)) & ((col(6) == 1) | (col(7) == 1))
    elif group == 'ZF2U_ZF6B':  # (ZF2 unbound) AND (ZF6 bound)
        mask = (col(2) == 0) & (col(6) == 1)
    else:
        raise ValueError("group must be 'N_open', 'C_open', or 'ZF2U_ZF6B'")
    mask = mask.fillna(False)
    return mask[mask].index  # MultiIndex of (motif_id, read_id)

def compute_metaplot(df_calls_with_motif, selected_ids, window=1000, bin_size=5):
    """Fraction unmethylated per bin across +/- window (strand-aware)."""
    if selected_ids is None or len(selected_ids) == 0:
        return pd.DataFrame({'rel_bin': [], 'frac_unmeth': []})

    df = df_calls_with_motif.copy()
    df = df[df['status'].isin(['M', 'U'])].copy()

    sf = strand_factor(df['strand_motif'])
    df['rel_pos'] = ((df['call_pos'].astype('int64') - df['specific_pos'].astype('int64')) * sf).astype('int64')

    df['motif_id'] = make_motif_id(df)
    key = pd.MultiIndex.from_tuples(selected_ids, names=['motif_id', 'read_id'])
    df = df.set_index(['motif_id', 'read_id']).loc[key].reset_index()

    df = df[(df['rel_pos'] >= -window) & (df['rel_pos'] <= window)].copy()
    if df.empty:
        return pd.DataFrame({'rel_bin': [], 'frac_unmeth': []})
    df['rel_bin'] = (np.floor_divide(df['rel_pos'], bin_size) * bin_size).astype('int32')

    df['unmeth'] = (df['status'] == 'U').astype('int8')
    agg = df.groupby('rel_bin', sort=True)['unmeth'].agg(['sum', 'count']).reset_index()
    agg['frac_unmeth'] = agg['sum'] / agg['count']

    all_bins = pd.Index(np.arange(-window, window + 1, bin_size, dtype=int), name='rel_bin')
    out = pd.DataFrame({'rel_bin': all_bins})
    out = out.merge(agg[['rel_bin','frac_unmeth']], on='rel_bin', how='left')
    return out

def compute_metaplot_stratified(df_calls, selected_pairs, nuc_pairs, window=1000, bin_size=5):
    """
    One-pass stratified metaplot for condition 1:
    Returns a DataFrame with columns ['rel_bin','label','frac_unmeth'] where label is
    'retained' or 'nucleosome-flagged'.
    """
    import numpy as np
    import pandas as pd

    if not selected_pairs:
        return pd.DataFrame(columns=['rel_bin','label','frac_unmeth'])

    # Keep only what we need, ensure motif_id exists if needed
    need_cols = ['motif_id','read_id','strand_motif','call_pos','specific_pos','status']
    extra_cols = ['chr_motif','start_motif','end_motif']
    have = [c for c in need_cols + extra_cols if c in df_calls.columns]
    df = df_calls[df_calls['status'].isin(['M','U'])][have].copy()

    if 'motif_id' not in df.columns or df['motif_id'].isna().any():
        if all(c in df.columns for c in ['chr_motif','start_motif','end_motif','strand_motif']):
            df['motif_id'] = (df['chr_motif'].astype(str) + '_' +
                              df['start_motif'].astype(str) + '_' +
                              df['end_motif'].astype(str) + '_' +
                              df['strand_motif'].astype(str))
        else:
            raise ValueError("compute_metaplot_stratified: need 'motif_id' or chr/start/end/strand to build it.")

    sel = set(selected_pairs)
    df['pair'] = list(zip(df['motif_id'], df['read_id']))
    df = df[df['pair'].isin(sel)]
    if df.empty:
        return pd.DataFrame(columns=['rel_bin','label','frac_unmeth'])

    # Label retained vs nucleosome-flagged
    nuc_pairs = set(nuc_pairs or [])
    df['label'] = np.where(df['pair'].isin(nuc_pairs), 'nucleosome-flagged', 'retained')

    # Strand-aware relative position + window
    sf = strand_factor(df['strand_motif'])
    df['rel_pos'] = ((df['call_pos'].astype('int64') - df['specific_pos'].astype('int64')) * sf).astype('int32')
    df = df[(df['rel_pos'] >= -window) & (df['rel_pos'] <= window)]
    if df.empty:
        return pd.DataFrame(columns=['rel_bin','label','frac_unmeth'])

    # Bin & frac(U)
    df['rel_bin'] = (df['rel_pos'].astype('int32') // bin_size * bin_size).astype('int32')
    isU = (df['status'] == 'U').astype('uint8')
    out = (df.assign(isU=isU)
             .groupby(['rel_bin','label'], sort=False)['isU']
             .agg(['sum','count'])
             .reset_index())
    out['frac_unmeth'] = out['sum'] / out['count']
    return out[['rel_bin','label','frac_unmeth']].sort_values(['rel_bin','label'])
    
def compute_metaplot_labeled(df_calls_with_motif, pair_to_label, window=1000, bin_size=5):
    """
    One-pass metaplot for multiple classes.
    pair_to_label: dict {(motif_id, read_id): "Label"}
    Returns DF: rel_bin, label, frac_unmeth
    """
    if not pair_to_label:
        return pd.DataFrame(columns=['rel_bin','label','frac_unmeth'])

    df = df_calls_with_motif[df_calls_with_motif['status'].isin(['M','U'])].copy()
    if df.empty:
        return pd.DataFrame(columns=['rel_bin','label','frac_unmeth'])

    if 'motif_id' not in df.columns:
        df['motif_id'] = make_motif_id(df)

    sf = strand_factor(df['strand_motif'])
    df['rel_pos'] = ((df['call_pos'].astype('int64') - df['specific_pos'].astype('int64')) * sf).astype('int32')
    df = df[(df['rel_pos'] >= -window) & (df['rel_pos'] <= window)]
    if df.empty:
        return pd.DataFrame(columns=['rel_bin','label','frac_unmeth'])

    df['rel_bin'] = (np.floor_divide(df['rel_pos'], bin_size) * bin_size).astype('int32')
    df['unmeth'] = (df['status'] == 'U').astype('uint8')

    lab = pd.Series(pair_to_label)
    lab.index = pd.MultiIndex.from_tuples(lab.index, names=['motif_id','read_id'])

    df = (df.set_index(['motif_id','read_id'])
            .join(lab.rename('label'), how='inner')
            .reset_index())
    if df.empty:
        return pd.DataFrame(columns=['rel_bin','label','frac_unmeth'])

    out = (df.groupby(['rel_bin','label'], sort=False)['unmeth']
             .agg(['sum','count'])
             .reset_index())
    out['frac_unmeth'] = out['sum'] / out['count']
    return out[['rel_bin','label','frac_unmeth']].sort_values(['rel_bin','label'])

    
def moving_average(series, window):
    """Centered rolling mean over 'window' bins; window=1 leaves series unchanged."""
    if window is None or window <= 1:
        return series
    return series.rolling(window=int(window), center=True, min_periods=1).mean()
    
def plot_metaplot(lines, labels, title, outpath, smooth=1, xwin=None, colors=None, lw=2):
    """Plot multiple lines on one metaplot figure (optionally smoothed in bins)."""
    fig, ax = plt.subplots(figsize=(8, 4))
    clipped_lines = []

    for i, (df_line, label) in enumerate(zip(lines, labels)):
        if df_line is None or df_line.empty:
            continue
        df_use = df_line
        if xwin is not None:
            a, b = xwin
            df_use = df_use[(df_use['rel_bin'] >= a) & (df_use['rel_bin'] <= b)].copy()
        if df_use.empty:
            continue

        x = df_use['rel_bin']
        y = df_use['frac_unmeth']
        y_sm = moving_average(y, smooth)

        c = None
        if colors is not None and i < len(colors):
            c = colors[i]

        ax.plot(x, y_sm, label=label, color=c, lw=lw, zorder=3)
        clipped_lines.append(df_use)

    ax.set_xlabel('Position relative to motif center (bp)')
    ax.set_ylabel('Fraction unmethylated')
    ax.set_title(title)

    leg = ax.legend(frameon=False)
    # color legend text to match lines (nice for “as in fig”)
    if colors is not None and leg is not None:
        for txt, c in zip(leg.get_texts(), colors):
            txt.set_color(c)

    # x-limits
    if xwin is not None:
        ax.set_xlim(xwin[0], xwin[1])
    else:
        xs = [df['rel_bin'] for df in clipped_lines]
        if xs:
            xmin = min(s.min() for s in xs)
            xmax = max(s.max() for s in xs)
        else:
            xmin, xmax = -1000, 1000
        ax.set_xlim(xmin, xmax)

    ax.set_ylim(0, 1)
    plt.tight_layout()
    safe_save_pdf(fig, outpath, label="metaplot")



from pathlib import Path
import numpy as np

def variant_pdf(path, suffix):
    """Return <stem><suffix>.pdf next to the given path."""
    p = Path(path)
    if p.suffix.lower() != '.pdf':
        p = p.with_suffix('.pdf')
    return str(p.with_name(p.stem + suffix + p.suffix))

def split_static_dynamic(pivot, pairs):
    """
    Classify (motif_id, read_id) pairs using ZF1..ZF11 in 'pivot' (post-filter).
      static  = all measured core ZFs are 1
      dynamic = at least one 1 and at least one 0
      naked   = all measured core ZFs are 0
    Returns: (static_pairs, dynamic_pairs, naked_pairs, missing_pairs)
    """
    if pivot is None or pivot.empty or not pairs:
        return [], [], [], []
    cols = [c for c in pivot.columns if isinstance(c, (int, np.integer)) and 1 <= int(c) <= 11]
    static, dynamic, naked, missing = [], [], [], []
    try:
        pairs = pairs.tolist() if hasattr(pairs, 'tolist') else list(pairs)
    except Exception:
        pairs = list(pairs)

    idx = pivot.index
    for t in pairs:
        if t not in idx:
            missing.append(t); continue
        vals = pivot.loc[t, cols].dropna()
        if vals.empty:
            missing.append(t); continue
        any1 = (vals == 1).any()
        any0 = (vals == 0).any()
        if any1 and any0:
            dynamic.append(t)
        elif any1 and not any0:
            static.append(t)
        elif not any1 and any0:
            naked.append(t)
        else:
            naked.append(t)
    return static, dynamic, naked, missing


    


# ────────────────── main ──────────────────
def main():
    p = argparse.ArgumentParser(description="Strand-aware metaplots for ZF configuration groups from intersect-style or wide calls.")
    p.add_argument('-i', '--input', required=True, help='Calls TSV (condition 1)')
    p.add_argument('--input2', help='Calls TSV (condition 2, optional)')
    p.add_argument('-m', '--motif-bed', required=True,
                   help='Motif BED/TSV: chr start end strand [specific_pos optional col5]')
    p.add_argument('--calls-format', choices=['auto','intersect','wide'], default='auto',
                   help='Input calls format: 16+ col intersect or 10-col wide (auto-detect default)')
    p.add_argument('--assign-window', type=int, default=1500, help='WIDE mode: max distance from motif center to assign a call (bp)')
    p.add_argument('--min-gpc', type=int, default=0, help='Min CpG calls per (motif,read,ZF)')
    p.add_argument('--mol-threshold', type=int, default=0, help='≥N bound ZFs per read (0 disables)')
    p.add_argument('--exact', action='store_true', help='Use exactly N bound ZFs per read (with --mol-threshold)')
    p.add_argument('--drop_all_bound', action='store_true', help='Drop reads where ALL measured ZFs are bound')
    p.add_argument('--window', type=int, default=1000, help='Half-window around center for metaplot (bp)')
    p.add_argument('--bin', type=int, default=5, help='Bin size (bp)')
    p.add_argument('--chroms', type=str, help='Comma-separated chromosomes to keep (e.g., chr1,chr2,chrX)')
    p.add_argument('--max-rows', type=int, default=0, help='WIDE mode: read only first N rows for testing')
    p.add_argument('--jobs', type=int, default=max(1, min(cpu_count(), 16)),
                   help='Parallel jobs for per-chrom aggregation (default: CPU cores, capped at 16)')
    p.add_argument('--profile', action='store_true', help='Print stage timings')

    # exports
    p.add_argument('--export_metaplot_n_open', default=None, help='PDF for Group 1 (N_open) metaplot')
    p.add_argument('--export_metaplot_c_open', default=None, help='PDF for Group 2 (C_open) metaplot')

    p.add_argument('--export_metaplot_2u6b', help='PDF for Group 3 (ZF2 unbound & ZF6 bound)')
    p.add_argument('--export_curves_tsv', help='Optional TSV with underlying curves (wide format)')

    # nucleosome exclusion
    p.add_argument('--exclude_nucleosome_up', action='store_true',
                   help='Exclude pairs with ALL U in upstream flank window [-nuc_ndr_hi, -nuc_ndr_lo].')
    p.add_argument('--exclude_nucleosome_down', action='store_true',
                   help='Exclude pairs with ALL U in downstream flank window [nuc_ndr_lo, nuc_ndr_hi].')
    p.add_argument('--nuc_ndr_lo', type=int, default=40,
                   help='Inner bound (bp) of nucleosome flank window (default 40).')
    p.add_argument('--nuc_ndr_hi', type=int, default=60,
                   help='Outer bound (bp) of nucleosome flank window (default 60).')

    # stratified overlays
    p.add_argument('--metaplot_nuc_stratify', action='store_true',
                   help='Also plot retained vs nucleosome-flagged overlays for condition 1.')
    p.add_argument('--export_metaplot_n_open_nuc', help='PDF for Group 1 (N_open) stratified overlay.')
    p.add_argument('--export_metaplot_c_open_nuc', help='PDF for Group 2 (C_open) stratified overlay.')
    p.add_argument('--export_metaplot_2u6b_nuc', help='PDF for Group 3 (ZF2U & ZF6B) stratified overlay.')
    p.add_argument('--metaplot_nuc_mix', action='store_true',
               help='Plot a single combined metaplot using ALL selected pairs (N_open, C_open, ZF2U&ZF6B), stratified as included vs excluded.')
    p.add_argument('--export_metaplot_nuc_mix', help='PDF for combined included vs excluded stratified metaplot.')
    
    p.add_argument('--smooth', type=int, default=1,
               help='Rolling window (in bins) to smooth plotted curves; 1 disables.')
    
    p.add_argument('--plot_static_dynamic', action='store_true',
               help='Make static-only and dynamic-only metaplots per group (C1/C2 lines).')
    p.add_argument('--export_metaplot_all', type=str, default=None,
               help='PDF for mixture metaplot using ALL selected pairs (C1/C2 lines).')
               
    p.add_argument('--global_plots', action='store_true',
               help='Build global (non-grouped) plots for dynamic, static, mix, nucleosome, and all.')
    p.add_argument('--export_global_dynamic', help='PDF: dynamic-only molecules (no nuc/naked/static).')
    p.add_argument('--export_global_static', help='PDF: static-only molecules (no nuc/naked/dynamic).')
    p.add_argument('--export_global_mix_sd', help='PDF: mixture of static+dynamic (no nuc/naked).')
    p.add_argument('--export_global_nucleosome', help='PDF: nucleosome-only molecules (no naked/static/dynamic).')
    p.add_argument('--export_global_all', help='PDF: all molecules (nucleosome + naked + static + dynamic).')
    p.add_argument('--export_global_naked', help='PDF: naked-only molecules (exclude nucleosome & static & dynamic).')
    p.add_argument('--plot-xwin', type=parse_xwin, default=None,
               help="Custom x-axis window in bp as 'a:b' relative to motif center (e.g., -200:400). "
                    "Only affects plotting; increase --window if you crop beyond computed range.")
    p.add_argument('--export_global_overlay', help='PDF: overlay of nucleosomes/unbound/static/dynamic (condition 1).')







    args = p.parse_args()
    
    if args.plot_xwin:
        log(f"[PLOT] applying custom x window: {args.plot_xwin[0]}:{args.plot_xwin[1]}", True)


    chroms = args.chroms.split(',') if args.chroms else None

    # Fast existence checks
    for path,label in [(args.input,'--input'), (args.motif_bed,'--motif-bed')]:
        if not Path(path).exists():
            print(f"ERROR: {label} not found: {path}", file=sys.stderr); sys.exit(2)
    if args.input2 and not Path(args.input2).exists():
        print(f"ERROR: --input2 not found: {args.input2}", file=sys.stderr); sys.exit(2)

    motif_df = read_motif_bed(args.motif_bed)

    # Condition 1
    fmt1 = detect_calls_format(args.input) if args.calls_format == 'auto' else args.calls_format
    t0 = now()
    if fmt1 == 'intersect':
        df1 = load_calls_intersect(args.input, chroms=chroms, profile=args.profile)
        df1 = df1.merge(motif_df[['chr_motif','start_motif','end_motif','strand_motif','specific_pos']],
                        on=['chr_motif','start_motif','end_motif','strand_motif'],
                        how='inner')
        coalesce_specific_pos_inplace(df1)
    else:
        df1_raw = load_calls_wide(args.input, max_rows=args.max_rows, chroms=chroms, profile=args.profile)
        df1 = align_calls_to_motifs_wide(df1_raw, motif_df, assign_window=args.assign_window,
                                         chroms=chroms, profile=args.profile)
    log(f"[PREP] condition1 ready in {now()-t0:.2f}s", args.profile)

    # Nucleosome flags (Condition 1)
    nuc_up = args.exclude_nucleosome_up
    nuc_down = args.exclude_nucleosome_down
    nuc_lo, nuc_hi = int(args.nuc_ndr_lo), int(args.nuc_ndr_hi)
    nuc_pairs1 = flag_nucleosome_pairs(df1, lo=nuc_lo, hi=nuc_hi, up=nuc_up, down=nuc_down) if (nuc_up or nuc_down) else set()

    # Condition 2 (optional)
    df2 = None
    if args.input2:
        fmt2 = detect_calls_format(args.input2) if args.calls_format == 'auto' else args.calls_format
        t0 = now()
        if fmt2 == 'intersect':
            df2 = load_calls_intersect(args.input2, chroms=chroms, profile=args.profile)
            df2 = df2.merge(motif_df[['chr_motif','start_motif','end_motif','strand_motif','specific_pos']],
                            on=['chr_motif','start_motif','end_motif','strand_motif'],
                            how='inner')
            coalesce_specific_pos_inplace(df2)
        else:
            df2_raw = load_calls_wide(args.input2, max_rows=args.max_rows, chroms=chroms, profile=args.profile)
            df2 = align_calls_to_motifs_wide(df2_raw, motif_df, assign_window=args.assign_window,
                                             chroms=chroms, profile=args.profile)
        log(f"[PREP] condition2 ready in {now()-t0:.2f}s", args.profile)

    # Build pivots (parallel per-chrom)
    pivot1 = build_pivot_parallel(df1, min_gpc=args.min_gpc, mol_threshold=args.mol_threshold,
                                  exact=args.exact, drop_all_bound=args.drop_all_bound,
                                  jobs=args.jobs, profile=args.profile)
    pivot2 = build_pivot_parallel(df2, min_gpc=args.min_gpc, mol_threshold=args.mol_threshold,
                                  exact=args.exact, drop_all_bound=args.drop_all_bound,
                                  jobs=args.jobs, profile=args.profile) if df2 is not None else None

    # Select groups per condition
    ids1_N = select_group_ids(pivot1, 'N_open')
    ids1_C = select_group_ids(pivot1, 'C_open')
    ids1_26 = select_group_ids(pivot1, 'ZF2U_ZF6B')
    pre_N, pre_C, pre_26 = len(ids1_N), len(ids1_C), len(ids1_26)

    # Keep pre-exclusion copies for stratified overlays
    ids1_N_all  = list(ids1_N)
    ids1_C_all  = list(ids1_C)
    ids1_26_all = list(ids1_26)
    

    def _exclude_flagged(ids, flagged):
        """Return a list of (motif_id, read_id) not in the flagged set."""
        if ids is None:
            return []
        # Normalize to a plain list of tuples to avoid MultiIndex truthiness issues
        try:
            import pandas as pd  # already imported above; safe if repeated
            if isinstance(ids, pd.MultiIndex):
                iterable = ids.tolist()
            else:
                iterable = list(ids)
        except Exception:
            iterable = list(ids)

        if len(iterable) == 0:
            return []
        return [t for t in iterable if t not in flagged]


    if nuc_pairs1:
        ids1_N  = _exclude_flagged(ids1_N,  nuc_pairs1)
        ids1_C  = _exclude_flagged(ids1_C,  nuc_pairs1)
        ids1_26 = _exclude_flagged(ids1_26, nuc_pairs1)
        log(f"[NUC] excluded {pre_N - len(ids1_N)} N_open, {pre_C - len(ids1_C)} C_open, {pre_26 - len(ids1_26)} ZF2U/ZF6B", True)

    ids2_N = select_group_ids(pivot2, 'N_open') if pivot2 is not None else []
    ids2_C = select_group_ids(pivot2, 'C_open') if pivot2 is not None else []
    ids2_26 = select_group_ids(pivot2, 'ZF2U_ZF6B') if pivot2 is not None else []
    
    # Compute metaplots (only if needed)
    need_group_metaplots = any([
        args.export_metaplot_n_open,
        args.export_metaplot_c_open,
        args.export_metaplot_2u6b,
        args.metaplot_nuc_stratify,
        args.plot_static_dynamic,
        args.export_metaplot_all,
        args.export_curves_tsv,
    ])

    if need_group_metaplots:
        t0 = now()
        meta1_N  = compute_metaplot(df1, ids1_N,  window=args.window, bin_size=args.bin)
        meta1_C  = compute_metaplot(df1, ids1_C,  window=args.window, bin_size=args.bin)
        meta1_26 = compute_metaplot(df1, ids1_26, window=args.window, bin_size=args.bin)
        meta2_N  = compute_metaplot(df2, ids2_N,  window=args.window, bin_size=args.bin) if df2 is not None else None
        meta2_C  = compute_metaplot(df2, ids2_C,  window=args.window, bin_size=args.bin) if df2 is not None else None
        meta2_26 = compute_metaplot(df2, ids2_26, window=args.window, bin_size=args.bin) if df2 is not None else None
        log(f"[METAPLOT] built in {now()-t0:.2f}s", args.profile)
    else:
        meta1_N = meta1_C = meta1_26 = None
        meta2_N = meta2_C = meta2_26 = None

    # Plot standard
   # Standard
    if args.export_metaplot_n_open:
        plot_metaplot(
            lines=[meta1_N, meta2_N],
            labels=['Condition 1', 'Condition 2'] if df2 is not None else ['Condition 1'],
            title='Fraction unmethylated — Group 1: (ZF1/2 unbound) & (ZF6/7 bound)',
            outpath=args.export_metaplot_n_open,
            smooth=args.smooth,
            xwin=args.plot_xwin
        )

    if args.export_metaplot_c_open:
        plot_metaplot(
            lines=[meta1_C, meta2_C],
            labels=['Condition 1', 'Condition 2'] if df2 is not None else ['Condition 1'],
            title='Fraction unmethylated — Group 2: (ZF10/11 unbound) & (ZF6/7 bound)',
            outpath=args.export_metaplot_c_open,
            smooth=args.smooth,
            xwin=args.plot_xwin
        )


    if args.export_metaplot_2u6b:
        plot_metaplot(
            lines=[meta1_26, meta2_26],
            labels=['Condition 1', 'Condition 2'] if df2 is not None else ['Condition 1'],
            title='Fraction unmethylated — Group 3: (ZF2 unbound) & (ZF6 bound)',
            outpath=args.export_metaplot_2u6b,
            smooth=args.smooth,
            xwin=args.plot_xwin
    )





    # ── Stratified overlays (Condition 1) ─────────────────────────────
    if args.metaplot_nuc_stratify:
        def _plot_strat(ids_all, label, out_pdf):
            if not out_pdf:
                return
            strat = compute_metaplot_stratified(df1, ids_all, nuc_pairs1,
                                                window=args.window, bin_size=args.bin)
            if strat.empty:
                log(f"[NUC-STRAT] {label}: no data to plot", True)
                return

            piv = strat.pivot(index='rel_bin', columns='label', values='frac_unmeth').reset_index()

            # Two lines: retained vs nucleosome-flagged
            ret = piv[['rel_bin']].copy()
            ret['frac_unmeth'] = piv.get('retained')

            nuc = piv[['rel_bin']].copy()
            nuc['frac_unmeth'] = piv.get('nucleosome-flagged')

            # Counts for legend
            flagged = len([t for t in ids_all if t in nuc_pairs1])
            kept = len(ids_all) - flagged

            plot_metaplot(
                lines=[ret.dropna(), nuc.dropna()],
                labels=[f"{label} — retained (n={kept})",
                        f"{label} — nucleosome-flagged (n={flagged})"],
                title=(f"Fraction unmethylated — {label} (nucleosome flank "
                       f"[{args.nuc_ndr_lo},{args.nuc_ndr_hi}] bp"
                       f"{' up' if args.exclude_nucleosome_up else ''}"
                       f"{' and down' if args.exclude_nucleosome_up and args.exclude_nucleosome_down else (' down' if args.exclude_nucleosome_down else '')})"),
                outpath=out_pdf,
                smooth=args.smooth,
                xwin=args.plot_xwin,          # ← add this
            )




        _plot_strat(ids1_N_all,  "N_open",      args.export_metaplot_n_open_nuc)
        _plot_strat(ids1_C_all,  "C_open",      args.export_metaplot_c_open_nuc)
        _plot_strat(ids1_26_all, "ZF2U & ZF6B", args.export_metaplot_2u6b_nuc)
        
                # ── Combined included vs excluded across ALL selected pairs (cond1) ──
        if args.metaplot_nuc_mix and args.export_metaplot_nuc_mix:
            all_ids = list({*ids1_N_all, *ids1_C_all, *ids1_26_all})
            if not all_ids:
                log("[NUC-STRAT] combined: no pairs to plot", True)
            else:
                strat_all = compute_metaplot_stratified(
                    df1, all_ids, nuc_pairs1, window=args.window, bin_size=args.bin
                )
                if strat_all.empty:
                    log("[NUC-STRAT] combined: no data in window", True)
                else:
                    piv = strat_all.pivot(index='rel_bin', columns='label', values='frac_unmeth').reset_index()
                    inc = piv[['rel_bin']].copy(); inc['frac_unmeth'] = piv.get('retained')
                    exc = piv[['rel_bin']].copy(); exc['frac_unmeth'] = piv.get('nucleosome-flagged')

                    flagged = len([t for t in all_ids if t in nuc_pairs1])
                    kept = len(all_ids) - flagged

                    plot_metaplot(
                        lines=[inc.dropna(), exc.dropna()],
                        labels=[f"Included (n={kept})", f"Excluded (n={flagged})"],
                        title=(f"Fraction unmethylated — ALL selected pairs — included vs excluded ..."),
                        outpath=args.export_metaplot_nuc_mix,
                        smooth=args.smooth,
                        xwin=args.plot_xwin,          # ← add this
                    )


    # ── Static/Dynamic-only plots (C1/C2 lines) ─────────────────────────────
    if args.plot_static_dynamic:
        def _make_sd_plots(group_name, ids1, ids2, base_outpath):
            if not base_outpath:
                return
            # Classify on POST-filter pivots (pivot1/pivot2)
            st1, dy1, nk1, ms1 = split_static_dynamic(pivot1, ids1)
            st2, dy2, nk2, ms2 = split_static_dynamic(pivot2, ids2) if pivot2 is not None else ([], [], [], [])

            # Static
            meta_st1 = compute_metaplot(df1, st1, window=args.window, bin_size=args.bin)
            meta_st2 = compute_metaplot(df2, st2, window=args.window, bin_size=args.bin) if df2 is not None else None
            if (meta_st1 is not None and not meta_st1.empty) or (meta_st2 is not None and not meta_st2.empty):
                plot_metaplot(
                    lines=[meta_st1, meta_st2],
                    labels=['Condition 1', 'Condition 2'] if df2 is not None else ['Condition 1'],
                    title=f'Fraction unmethylated — {group_name} — STATIC (n={len(st1)}{f"/{len(st2)}" if df2 is not None else ""})',
                    outpath=variant_pdf(base_outpath, '_static'),
                    smooth=args.smooth,
                    xwin=args.plot_xwin,          # ← add this
                )

            # Dynamic
            meta_dy1 = compute_metaplot(df1, dy1, window=args.window, bin_size=args.bin)
            meta_dy2 = compute_metaplot(df2, dy2, window=args.window, bin_size=args.bin) if df2 is not None else None
            if (meta_dy1 is not None and not meta_dy1.empty) or (meta_dy2 is not None and not meta_dy2.empty):
                # Dynamic
                plot_metaplot(
                    lines=[meta_dy1, meta_dy2],
                    labels=['Condition 1', 'Condition 2'] if df2 is not None else ['Condition 1'],
                    title=f'Fraction unmethylated — {group_name} — DYNAMIC (n={len(dy1)}{f"/{len(dy2)}" if df2 is not None else ""})',
                    outpath=variant_pdf(base_outpath, '_dynamic'),
                    smooth=args.smooth,
                    xwin=args.plot_xwin,          # ← add this
                )

            # (Optional) print quick counts
            log(f"[SD] {group_name}: static C1={len(st1)} C2={len(st2)}; dynamic C1={len(dy1)} C2={len(dy2)}; "
                f"naked C1={len(nk1)} C2={len(nk2)}; missing C1={len(ms1)} C2={len(ms2)}", True)

        if args.export_metaplot_n_open:
            _make_sd_plots('N_open', ids1_N, ids2_N, args.export_metaplot_n_open)

        if args.export_metaplot_c_open:
            _make_sd_plots('C_open', ids1_C, ids2_C, args.export_metaplot_c_open)

        if args.export_metaplot_2u6b:
            _make_sd_plots('ZF2U & ZF6B', ids1_26, ids2_26, args.export_metaplot_2u6b)

    # ── Mixture of ALL molecules (C1/C2 lines) ─────────────────────────────
    if args.export_metaplot_all:
        all_ids1 = list({*ids1_N, *ids1_C, *ids1_26})
        all_ids2 = list({*ids2_N, *ids2_C, *ids2_26}) if df2 is not None else []
        meta_all1 = compute_metaplot(df1, all_ids1, window=args.window, bin_size=args.bin)
        meta_all2 = compute_metaplot(df2, all_ids2, window=args.window, bin_size=args.bin) if df2 is not None else None

        plot_metaplot(
            lines=[meta_all1, meta_all2],
            labels=['Condition 1', 'Condition 2'] if df2 is not None else ['Condition 1'],
            title='Fraction unmethylated — ALL selected pairs (mixture)',
            outpath=args.export_metaplot_all,
            smooth=args.smooth,
            xwin=args.plot_xwin,          # ← add this
        )


    # ── Global (non-grouped) classification & plots ─────────────────────────────
    if args.global_plots:
        # Nucleosome flags for condition 2 as well (if present)
        nuc_pairs2 = flag_nucleosome_pairs(
            df2, lo=nuc_lo, hi=nuc_hi, up=nuc_up, down=nuc_down
        ) if (df2 is not None and (nuc_up or nuc_down)) else set()

        # All observed pairs (by core ZF pivot)
        all_pairs1 = list(pivot1.index) if pivot1 is not None and not pivot1.empty else []
        all_pairs2 = list(pivot2.index) if pivot2 is not None and not pivot2.empty else []

        # Split (static/dynamic/naked) using core ZF states
        st1, dy1, nk1, ms1 = split_static_dynamic(pivot1, all_pairs1)
        st2, dy2, nk2, ms2 = split_static_dynamic(pivot2, all_pairs2) if pivot2 is not None else ([],[],[],[])

        # Enforce exclusivity by removing nucleosome pairs from SDN sets
        nuc1 = set(nuc_pairs1)
        nuc2 = set(nuc_pairs2)

        st1 = [t for t in st1 if t not in nuc1]
        dy1 = [t for t in dy1 if t not in nuc1]
        nk1 = [t for t in nk1 if t not in nuc1]

        st2 = [t for t in st2 if t not in nuc2]
        dy2 = [t for t in dy2 if t not in nuc2]
        nk2 = [t for t in nk2 if t not in nuc2]

        # Nucleosome-only sets (exclude anything that also falls into static/dynamic/naked)
        sdn1 = set().union(st1, dy1, nk1)
        sdn2 = set().union(st2, dy2, nk2)
        nuc_only1 = [t for t in nuc1 if t not in sdn1]
        nuc_only2 = [t for t in nuc2 if t not in sdn2]

        # Mixture (static + dynamic), no nuc/naked
        mix1 = list(set().union(st1, dy1))
        mix2 = list(set().union(st2, dy2))

        # "All molecules" = union of four disjoint sets
        all1 = list(set().union(st1, dy1, nk1, nuc_only1))
        all2 = list(set().union(st2, dy2, nk2, nuc_only2))

        log(f"[GLOBAL] C1 counts — static={len(st1)} dynamic={len(dy1)} naked={len(nk1)} nucOnly={len(nuc_only1)} all={len(all1)}", True)
        if df2 is not None:
            log(f"[GLOBAL] C2 counts — static={len(st2)} dynamic={len(dy2)} naked={len(nk2)} nucOnly={len(nuc_only2)} all={len(all2)}", True)
        
                # Overlay (single plot) for condition 1: nucleosomes / unbound / static / dynamic
        if args.export_global_overlay:
            pair_to_label = {}
            for t in nuc_only1: pair_to_label[t] = 'Nucleosomes'
            for t in nk1:       pair_to_label[t] = 'Unbound'
            for t in st1:       pair_to_label[t] = 'Static'
            for t in dy1:       pair_to_label[t] = 'Dynamic'

            strat = compute_metaplot_labeled(df1, pair_to_label, window=args.window, bin_size=args.bin)

            all_bins = pd.DataFrame({'rel_bin': np.arange(-args.window, args.window + 1, args.bin, dtype=int)})

            def _line(label):
                df_lab = strat[strat['label'] == label][['rel_bin','frac_unmeth']]
                return all_bins.merge(df_lab, on='rel_bin', how='left')

            lines  = [_line('Nucleosomes'), _line('Unbound'), _line('Static'), _line('Dynamic')]
            labels = ['Nucleosomes', 'Unbound', 'Static', 'Dynamic']
            colors = ['green', 'dodgerblue', 'black', 'darkred']

            plot_metaplot(
                lines=lines,
                labels=labels,
                title='Fig.1SE',
                outpath=args.export_global_overlay,
                smooth=args.smooth,
                xwin=args.plot_xwin,
                colors=colors,
                lw=2
            )

        # Helper to plot two-condition lines if C2 provided
        def _plot_global(pairs1, pairs2, title, out_pdf):
            if not out_pdf:
                return
            m1 = compute_metaplot(df1, pairs1, window=args.window, bin_size=args.bin)
            m2 = compute_metaplot(df2, pairs2, window=args.window, bin_size=args.bin) if df2 is not None else None
            plot_metaplot(
                lines=[m1, m2],
                labels=['Condition 1', 'Condition 2'] if df2 is not None else ['Condition 1'],
                title=title,
                outpath=out_pdf,
                smooth=args.smooth,
                xwin=args.plot_xwin,      # ← add this
            )


        # 1) Dynamic-only
        _plot_global(
            dy1, dy2,
            "Fraction unmethylated — Dynamic molecules (exclude nucleosome & naked & static)",
            args.export_global_dynamic
        )

        # 2) Static-only
        _plot_global(
            st1, st2,
            "Fraction unmethylated — Static molecules (exclude nucleosome & naked & dynamic)",
            args.export_global_static
        )

        # 3) Mixture of 1+2
        _plot_global(
            mix1, mix2,
            "Fraction unmethylated — Mixture (Static + Dynamic; exclude nucleosome & naked)",
            args.export_global_mix_sd
        )

        # 4) Nucleosome-only
        _plot_global(
            nuc_only1, nuc_only2,
            f"Fraction unmethylated — Nucleosome-only (flank [{nuc_lo},{nuc_hi}] bp"
            f"{' up' if nuc_up else ''}"
            f"{' and down' if nuc_up and nuc_down else (' down' if nuc_down else '')})",
            args.export_global_nucleosome
        )

        # 5) All molecules
        _plot_global(
            all1, all2,
            "Fraction unmethylated — All molecules (nucleosome + naked + static + dynamic)",
            args.export_global_all
        )
        
        # Naked-only
        _plot_global(
            nk1, nk2,
            "Fraction unmethylated — Naked-only (exclude nucleosome & static & dynamic)",
            args.export_global_naked
        )



    
    # Export curves (optional)
    if args.export_curves_tsv:
        # pick a master rel_bin from any non-empty curve
        base = None
        for cand in [meta1_N, meta1_C, meta1_26, meta2_N, meta2_C, meta2_26]:
            if cand is not None and not cand.empty:
                base = cand[['rel_bin']].copy()
                break
        if base is None:
            base = pd.DataFrame({'rel_bin': np.arange(-args.window, args.window + 1, args.bin, dtype=int)})
        df_out = base.copy()
        # C1
        if meta1_N is not None:
            df_out = df_out.merge(meta1_N.rename(columns={'frac_unmeth':'C1_Group1_fracU'})[['rel_bin','C1_Group1_fracU']], on='rel_bin', how='left')
        if meta1_C is not None:
            df_out = df_out.merge(meta1_C.rename(columns={'frac_unmeth':'C1_Group2_fracU'})[['rel_bin','C1_Group2_fracU']], on='rel_bin', how='left')
        if meta1_26 is not None:
            df_out = df_out.merge(meta1_26.rename(columns={'frac_unmeth':'C1_Group3_fracU'})[['rel_bin','C1_Group3_fracU']], on='rel_bin', how='left')
        # C2
        if df2 is not None:
            if meta2_N is not None:
                df_out = df_out.merge(meta2_N.rename(columns={'frac_unmeth':'C2_Group1_fracU'})[['rel_bin','C2_Group1_fracU']], on='rel_bin', how='left')
            if meta2_C is not None:
                df_out = df_out.merge(meta2_C.rename(columns={'frac_unmeth':'C2_Group2_fracU'})[['rel_bin','C2_Group2_fracU']], on='rel_bin', how='left')
            if meta2_26 is not None:
                df_out = df_out.merge(meta2_26.rename(columns={'frac_unmeth':'C2_Group3_fracU'})[['rel_bin','C2_Group3_fracU']], on='rel_bin', how='left')
        outp = Path(args.export_curves_tsv)
        if outp.parent and not outp.parent.exists():
            outp.parent.mkdir(parents=True, exist_ok=True)
        df_out.to_csv(str(outp), sep='\t', index=False)
        print(f"Saved curves to {outp}")

if __name__ == '__main__':
    main()


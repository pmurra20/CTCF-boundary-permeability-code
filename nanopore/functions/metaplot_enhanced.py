#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import tempfile
import os
from multiprocessing import Pool

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.family'] = 'Arial'
# 6.25.25 - added window option + bypass limit of ≤4 GiB for bedtools.

# Utility: smoothing
def moving_average(series, window):
    return series.rolling(window=window, center=True, min_periods=1).mean()

# Bedtools intersect wrapper: returns temp-file path only
def run_intersect(args_tuple):
    calls_file, motif_bed = args_tuple
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.bed')
    out_path = tmp.name
    tmp.close()

    cmd = ['bedtools', 'intersect', '-a', calls_file, '-b', motif_bed, '-wa', '-wb']
    with open(out_path, 'w') as out:
        subprocess.run(cmd, stdout=out, check=True)

    return out_path

# Compute fraction unmethylated per bin matrix
def compute_fraction_matrix(df, labels, bin_size, window):
    cols = [
        'chr_call','start_call','end_call','call_pos','strand_call',
        'read_id','llr_ratio','llr_met','llr_unmet','status',
        'chr_motif','start_motif','end_motif','strand_motif','score','specific_pos'
    ]
    df.columns = cols
    
# upstream (N→C) → negative; downstream → positive
    sf = {'+': -1, '-': 1}
    df['sf'] = df['strand_motif'].map(sf).fillna(1).astype(int)
    df['rel_pos'] = (df['call_pos'] - df['specific_pos']) * df['sf']

    cond = (df['rel_pos'] >= -window) & (df['rel_pos'] <= window)
    df = df.loc[cond].copy()  # <- avoids SettingWithCopyWarning

    bins = np.arange(-window, window + bin_size, bin_size)
    df['bin'] = pd.cut(
        df['rel_pos'],
        bins=bins,
        labels=np.arange(-window, window, bin_size)
    )

    df = df.dropna(subset=['bin']).copy()  # <- also safe
    df['bin'] = df['bin'].astype(int)


    df['motif_id'] = (
        df['chr_motif'].astype(str) + '_' +
        df['start_motif'].astype(str) + '_' +
        df['end_motif'].astype(str) + '_' +
        df['strand_motif']
    )

    mat = (
        df.groupby(['motif_id','bin'])['status']
          .apply(lambda x: (x == 'U').mean())
          .unstack(fill_value=np.nan)
    )
    return mat.reindex(columns=labels, fill_value=np.nan)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Enhanced metaplot: overlay multiple motif sets.'
    )
    parser.add_argument('--calls',     required=True, help='Calls expanded TSV file')
    parser.add_argument('--motif-beds', nargs='+', required=True,
                        help='One or more motif BED files (extended)')
    parser.add_argument('--labels',    nargs='+', default=None,
                        help='Labels for each motif bed (optional)')
    parser.add_argument('--csv',       type=str, default=None,
                        help='Optional path to output CSV of bin-wise mean and SD')
    parser.add_argument('-o','--output', required=True, help='Output PDF filename')
    parser.add_argument('-t','--threads',type=int, default=1,
                        help='CPU threads for parallel intersect')
    parser.add_argument('--bin-size', type=int, default=1, help='Bin size in bp')
    parser.add_argument('--smooth',   type=int, default=5,
                        help='Smoothing window in bins')
    parser.add_argument('--error-alpha', type=float, default=0.3,
                        help='Alpha for SD shading')
    parser.add_argument('--window',   type=int, default=65,
                        help='Half-window around motif center (bp)')
    parser.add_argument('--highlight', type=int, default=10,
                        help='Highlight ±bp around center')
    parser.add_argument('--ymin',     type=float, default=None,
                        help='Min y-axis limit')
    parser.add_argument('--ymax',     type=float, default=None,
                        help='Max y-axis limit')
    args = parser.parse_args()

    # Labels for legend
    if args.labels and len(args.labels) == len(args.motif_beds):
        labels_list = args.labels
    else:
        labels_list = [os.path.basename(b).split('.')[0] for b in args.motif_beds]

    # Bin labels
    labels = np.arange(-args.window, args.window, args.bin_size)

    # 1) launch all intersects, get back temp-file names
    tasks = [(args.calls, bed) for bed in args.motif_beds]
    with Pool(args.threads) as pool:
        temp_files = pool.map(run_intersect, tasks)

    # 2) read each temp-file into a DataFrame, then delete it
    dfs = []
    for tf in temp_files:
        # read with tab separator to avoid regex warning
        df = pd.read_csv(tf, sep='\t', engine='c', header=None, comment='#')
        dfs.append(df)
        os.remove(tf)

    # 3) build fraction matrices
    mats = [
        compute_fraction_matrix(df, labels, args.bin_size, args.window)
        for df in dfs
    ]

    # 4) compute stats
    stats = []
    for mat in mats:
        m = mat.mean(axis=0, skipna=True)
        s = mat.std(axis=0, skipna=True, ddof=1)
        stats.append((m, s))

    # Optional CSV export
    if args.csv:
        export_df = pd.DataFrame({'position': labels})
        for label, (mean_frac, sd_frac) in zip(labels_list, stats):
            export_df[f'{label}_mean'] = mean_frac.values
            export_df[f'{label}_sd']   = sd_frac.values
        export_df.to_csv(args.csv, index=False)
        print(f"Data exported to CSV: {args.csv}")

    # 5) plot
    plt.figure(figsize=(8,4), dpi=300)
    colors = plt.cm.tab10.colors
    for idx, (label, (mean_frac, sd_frac)) in enumerate(zip(labels_list, stats)):
        x = labels
        y = moving_average(mean_frac, args.smooth) if args.smooth > 1 else mean_frac
        sd = moving_average(sd_frac, args.smooth) if args.smooth > 1 else sd_frac
        clr = colors[idx % len(colors)]
        plt.fill_between(x, y - sd, y + sd, color=clr, alpha=args.error_alpha)
        plt.plot(x, y, color=clr, linewidth=1.5, label=label)

    # center & highlights
#    plt.axvline(0, color='black', linestyle='--', linewidth=1)
#    hl = args.highlight
#    plt.axvline(-hl, color='black', linestyle=':', linewidth=1)
#    plt.axvline( hl, color='black', linestyle=':', linewidth=1)

    plt.xlabel('Position relative to motif center (bp)')
    plt.ylabel('Fraction unmethylated')
    plt.title('Metaplot')
    plt.legend(frameon=False)
    if args.ymin is not None and args.ymax is not None:
        plt.ylim(args.ymin, args.ymax)
    plt.tight_layout()
    plt.savefig(args.output)
    print(f"Overlay plot saved to {args.output}")

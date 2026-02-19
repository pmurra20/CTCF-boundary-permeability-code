#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cluster ZF matrix -> per-motif cluster probabilities.
Optionally compare two matrices (e.g., High vs Low), run Mann-Whitney (two-sided),
compute Hodges-Lehmann difference with 95% CI, and plot MEDIAN with 95% CI error bars
plus significance stars.
"""

import argparse, re, math
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# Fonts/embedding
mpl.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.family'] = 'Arial'

def parse_clusters(spec: str):
    clusters, labels = [], []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        m = re.fullmatch(r"(\d+)-(\d+)", chunk)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            if a > b: a, b = b, a
            if not (1 <= a <= b <= 11):
                raise ValueError("cluster bounds must be within 1..11")
            clusters.append(list(range(a, b+1)))
            labels.append(f"ZF{a}-{b}")
        else:
            if re.fullmatch(r"\d+", chunk):
                z = int(chunk)
                if not (1 <= z <= 11):
                    raise ValueError("cluster elements must be within 1..11")
                clusters.append([z])
                labels.append(f"ZF{z}")
            else:
                raise ValueError(f"Bad cluster chunk: {chunk}")
    return clusters, labels

def detect_zf_columns(df: pd.DataFrame):
    zf_map = {}
    for c in df.columns:
        if isinstance(c, str) and c.isdigit():
            z = int(c)
            if 1 <= z <= 11:
                zf_map[c] = z
            continue
        if isinstance(c, (int, np.integer)):
            z = int(c)
            if 1 <= z <= 11:
                zf_map[str(c)] = z
            continue
        m = re.fullmatch(r"ZF(\d{1,2})", str(c))
        if m:
            z = int(m.group(1))
            if 1 <= z <= 11:
                zf_map[str(c)] = z
    if not zf_map:
        raise ValueError("No ZF columns found (expected '1'..'11' or 'ZF1'..'ZF11').")
    return zf_map

def build_clusters(df: pd.DataFrame, clusters: list, labels: list, zf_map: dict) -> pd.DataFrame:
    out = df[["motif_id"]].copy()
    for label, members in zip(labels, clusters):
        cols = [c for c, z in zf_map.items() if z in members]
        out[label] = df[cols].mean(axis=1, skipna=True) if cols else np.nan
    return out

def _p_to_stars(p: float) -> str:
    try:
        if p < 0.0001: return '****'
        if p < 0.001:  return '***'
        if p < 0.01:   return '**'
        if p < 0.05:   return '*'
    except Exception:
        pass
    return ''

def _sum_of_ranks_and_u(x, y):
    x = np.asarray(x, dtype=float); x = x[np.isfinite(x)]
    y = np.asarray(y, dtype=float); y = y[np.isfinite(y)]
    n1, n2 = len(x), len(y)
    z = np.concatenate([x, y])
    order = np.argsort(z, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(z)+1, dtype=float)
    _, inv, counts = np.unique(z, return_inverse=True, return_counts=True)
    for k, c in enumerate(counts):
        if c > 1:
            idx = np.where(inv == k)[0]
            ranks[idx] = ranks[idx].mean()
    R1 = ranks[:n1].sum()
    R2 = ranks[n1:].sum()
    U1 = R1 - n1*(n1+1)/2.0
    U2 = n1*n2 - U1
    U  = min(U1, U2)
    tie_term = np.sum(counts**3 - counts)
    sd = math.sqrt(n1*n2/12.0 * ((n1+n2+1) - tie_term/((n1+n2)*(n1+n2-1))))
    zscore = (U - n1*n2/2.0) / sd if sd > 0 else 0.0
    from math import erf, sqrt
    p = 2.0 * (1.0 - 0.5*(1.0 + erf(abs(zscore)/sqrt(2.0))))
    return R1, R2, U, U1, U2, p, counts

def _hl_and_ci(x, y, counts, alpha=0.05):
    x = np.asarray(x, dtype=float); x = x[np.isfinite(x)]
    y = np.asarray(y, dtype=float); y = y[np.isfinite(y)]
    n1, n2 = len(x), len(y)
    diffs = (y.reshape(1, -1) - x.reshape(-1, 1)).ravel(order="C")
    diffs.sort()
    HL = float(np.median(diffs))
    tie_term = np.sum(counts**3 - counts)
    sd_u = math.sqrt(n1*n2/12.0 * ((n1+n2+1) - tie_term/((n1+n2)*(n1+n2-1))))
    z = 1.959963984540054
    Uc = n1*n2/2.0 - z*sd_u
    k = int(math.floor(Uc))
    N = n1*n2
    L = max(1, k+1)
    Uind = min(N, N - k)
    ci_low  = float(diffs[L-1])
    ci_high = float(diffs[Uind-1])
    return HL, (ci_low, ci_high)

def _median_and_ci(x, alpha=0.05):
    x = np.asarray(x, dtype=float); x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return float('nan'), float('nan'), float('nan')
    xs = np.sort(x)
    med = float(np.median(xs))
    z = 1.959963984540054
    half = 0.5*n
    rad = 0.5*z*math.sqrt(n)
    L = max(1, int(math.floor(half - rad)))
    U = min(n, int(math.ceil (half + rad)))
    lo = float(xs[L-1])
    hi = float(xs[U-1])
    return med, lo, hi

def mwu_prism_like_table(high_df, low_df, labels, decimals: int = 5):
    rows = []
    for label in labels:
        x = high_df[label].values
        y = low_df[label].values
        x = x[np.isfinite(x)]; y = y[np.isfinite(y)]
        med_hi, lo_hi, hi_hi = _median_and_ci(x)
        med_lo, lo_lo, hi_lo = _median_and_ci(y)
        R_high, R_low, U_min, U_high, U_low, p, counts = _sum_of_ranks_and_u(x, y)
        HL, (ci_lo, ci_hi) = _hl_and_ci(x, y, counts)
        rows.append({
            "cluster": label,
            "Median High (n)": f"{med_hi:.{decimals}f} (n={len(x)})",
            "Median Low (n)":  f"{med_lo:.{decimals}f} (n={len(y)})",
            "Median 95% CI High [lo,hi]": f"[{lo_hi:.{decimals}f}, {hi_hi:.{decimals}f}]",
            "Median 95% CI Low  [lo,hi]": f"[{lo_lo:.{decimals}f}, {hi_lo:.{decimals}f}]",
            "Difference: Actual (Low-High)": float(med_lo - med_hi) if np.isfinite(med_lo) and np.isfinite(med_hi) else np.nan,
            "Difference: Hodges-Lehmann (Low-High)": HL,
            "HL 95% CI low": ci_lo,
            "HL 95% CI high": ci_hi,
            "Sum of ranks High, Low": f"{int(round(R_high))} , {int(round(R_low))}",
            "Mann-Whitney U (min)": float(U_min),
            "p value (two-sided, approx)": float(p),
            "Stars": _p_to_stars(p),
            "Significant (p<0.05)": "Yes" if (np.isfinite(p) and p < 0.05) else "No"
        })
    return pd.DataFrame(rows)

def make_plot_median_ci(high_df, low_df, labels, out_prefix, group_labels=("High PDS5A","Low PDS5A"), colors=("#3b82f6","#ef4444")) :
    med_hi, lo_hi, hi_hi = [], [], []
    med_lo, lo_lo, hi_lo = [], [], []
    for lab in labels:
        x = high_df[lab].dropna().values
        y = low_df[lab].dropna().values
        mh, lh, hh = _median_and_ci(x)
        ml, ll, hl = _median_and_ci(y)
        med_hi.append(mh); lo_hi.append(lh); hi_hi.append(hh)
        med_lo.append(ml); lo_lo.append(ll); hi_lo.append(hl)

    xpos = np.arange(len(labels), dtype=float)
    width = 0.38
    fig, ax = plt.subplots(figsize=(3.6, 5.2))

    ax.bar(xpos - width/2, med_hi, width, color=colors[0], label=group_labels[0])
    ax.bar(xpos + width/2, med_lo, width, color=colors[1], label=group_labels[1])

    yerr_hi = np.vstack([np.array(med_hi) - np.array(lo_hi), np.array(hi_hi) - np.array(med_hi)])
    yerr_lo = np.vstack([np.array(med_lo) - np.array(lo_lo), np.array(hi_lo) - np.array(med_lo)])
    ax.errorbar(xpos - width/2, med_hi, yerr=yerr_hi, fmt="none", capsize=0, lw=1, color="black")
    ax.errorbar(xpos + width/2, med_lo, yerr=yerr_lo, fmt="none", capsize=0, lw=1, color="black")

    ax.set_xticks(xpos); ax.set_xticklabels([lab.replace("ZF","").replace("-", "–") for lab in labels])
    ax.set_xlabel("ZF")
    ax.set_ylabel("Fraction of molecules\ninaccessible")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, loc="upper right")

    for i, lab in enumerate(labels):
        x = high_df[lab].dropna().values
        y = low_df[lab].dropna().values
        _, _, _, _, _, p, counts = _sum_of_ranks_and_u(x, y)
        stars = _p_to_stars(p)
        top = max(hi_hi[i], hi_lo[i])
        ystar = min(1.02, top + 0.04)
        ax.text(xpos[i], ystar, stars, ha='center', va='bottom', fontsize=12)

    fig.tight_layout()
    pdf = f"{out_prefix}.pdf"
    png = f"{out_prefix}.png"
    fig.savefig(pdf, dpi=300)
    fig.savefig(png, dpi=300)
    plt.close(fig)
    return pdf, png

def main():
    ap = argparse.ArgumentParser(description="Cluster ZF matrix and (optionally) compare two groups with MWU + HL CI; plot medians with 95% CI + stars.")
    ap.add_argument("--input", required=True, help="Matrix TSV with motif_id and ZF columns (1..11 or ZF1..ZF11). This is 'High' unless labels overridden.")
    ap.add_argument("--out", help="Output TSV path for clustered per-motif values of --input.")
    ap.add_argument("--clusters", default="1-2,3-8,9-11", help='Cluster spec (default: \"1-2,3-8,9-11\"). Use contiguous ranges like 1-2,3-8,9-11.')
    ap.add_argument("--compare-with", dest="cmp", help="Optional matrix TSV for the comparison group (e.g., Low)." )
    ap.add_argument("--out-compare", help="Optional output TSV for clustered per-motif values of --compare-with.")
    ap.add_argument("--plot-out-prefix", help="If set and --compare-with is provided, write <prefix>.pdf/.png for the bar plot (medians with 95% CI)." )
    ap.add_argument("--stats-out", help="If set and --compare-with is provided, write Prism-like MWU table (TSV)." )
    ap.add_argument("--stats-decimals", type=int, default=5, help="Decimal places for Median/CI values in MWU table (default: 5; minimum: 5)." )
    ap.add_argument("--group-labels", default="High PDS5A,Low PDS5A", help="Comma-separated labels for the groups (default: 'High PDS5A,Low PDS5A')." )
    ap.add_argument("--group-colors", default="#3b82f6,#ef4444", help="Comma-separated hex colors for the groups (default blue, red)." )

    args = ap.parse_args()

    df1 = pd.read_csv(args.input, sep="\t")
    if "motif_id" not in df1.columns:
        raise SystemExit("Input file must contain 'motif_id'.")
    zf1 = detect_zf_columns(df1)
    for c in list(zf1.keys()):
        df1[c] = pd.to_numeric(df1[c], errors="coerce" )

    clusters, labels = parse_clusters(args.clusters)
    c1 = build_clusters(df1, clusters, labels, zf1)
    if args.out:
        c1.to_csv(args.out, sep="\t", index=False)
        print(f"Wrote {args.out}")

    if not args.cmp:
        return

    df2 = pd.read_csv(args.cmp, sep="\t")
    if "motif_id" not in df2.columns:
        raise SystemExit("--compare-with file must contain 'motif_id'.")
    zf2 = detect_zf_columns(df2)
    for c in list(zf2.keys()):
        df2[c] = pd.to_numeric(df2[c], errors="coerce" )
    c2 = build_clusters(df2, clusters, labels, zf2)
    if args.out_compare:
        c2.to_csv(args.out_compare, sep="\t", index=False)
        print(f"Wrote {args.out_compare}")

    stats_dec = max(5, int(args.stats_decimals))
    stats = mwu_prism_like_table(c1, c2, labels, decimals=stats_dec)
    if args.stats_out:
        stats.to_csv(args.stats_out, sep="\t", index=False)
        print(f"Wrote {args.stats_out}")

    if args.plot_out_prefix:
        glabels = tuple([s.strip() for s in args.group_labels.split(",")])
        gcolors = tuple([s.strip() for s in args.group_colors.split(",")])
        pdf, png = make_plot_median_ci(c1, c2, labels, args.plot_out_prefix, group_labels=glabels, colors=gcolors)
        print(f"Wrote {pdf} and {png}")

if __name__ == "__main__":
    main()

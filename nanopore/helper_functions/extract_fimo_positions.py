#!/usr/bin/env python3
"""
Extract and sort sequence_name, start, stop, and strand columns from a cleaned FIMO TSV.

Usage:
    python extract_fimo_positions.py --input fimo_cleaned.tsv --output fimo_positions_sorted.bed
"""
import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser(description="Extract and sort FIMO position columns.")
    parser.add_argument("-i", "--input", required=True, help="Path to cleaned FIMO TSV file")
    parser.add_argument("-o", "--output", required=True, help="Path to output BED-like file")
    args = parser.parse_args()

    # Read the cleaned FIMO table
    df = pd.read_csv(args.input, sep='\t')

    # Select and sort the desired columns
    df_pos = df[['sequence_name', 'start', 'stop', 'strand']]
    df_sorted = df_pos.sort_values(['sequence_name', 'start', 'stop'])

    # Write out as BED-like file (no header)
    df_sorted.to_csv(args.output, sep='\t', index=False, header=False)

if __name__ == "__main__":
    main()

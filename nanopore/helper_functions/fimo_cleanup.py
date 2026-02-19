#!/usr/bin/env python3
"""
Clean a FIMO TSV file by removing empty columns and rows, keeping only populated entries.

Usage:
    python fimo_cleanup.py --input fimo.tsv --output fimo_cleaned.tsv
"""
import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser(description="Clean FIMO TSV output.")
    parser.add_argument("-i", "--input", required=True, help="Path to input FIMO TSV file")
    parser.add_argument("-o", "--output", required=True, help="Path to output cleaned TSV file")
    args = parser.parse_args()

    # Read the FIMO output, skipping comment lines
    df = pd.read_csv(args.input, sep='\t', comment='#')

    # Replace placeholder "." with NA to identify empty fields
    df.replace('.', pd.NA, inplace=True)

    # Drop columns that are entirely empty
    df.dropna(axis=1, how='all', inplace=True)

    # Drop rows that have any empty fields
    df.dropna(axis=0, how='any', inplace=True)

    # Save the cleaned DataFrame
    df.to_csv(args.output, sep='\t', index=False)

if __name__ == "__main__":
    main()

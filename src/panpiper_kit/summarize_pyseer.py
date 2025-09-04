#!/usr/bin/env python3
import argparse
import os
import re
import glob
import gzip
import logging
from typing import Tuple, Optional, Callable, Iterable

import pandas as pd
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)

def parse_name(fp: str) -> Tuple[str, str, str]:
    """
    Parse pyseer filename to extract species, metadata, and basename.
    
    Expected format: <species>__<metadata>.pyseer.tsv[.gz]
    
    Args:
        fp: File path to parse
        
    Returns:
        Tuple of (species, metadata, basename)
    """
    base = os.path.basename(fp)
    base = re.sub(r'\.gz$', '', base)
    if base.endswith('.pyseer.tsv'):
        stem = base[:-len('.pyseer.tsv')]
    else:
        stem = base
    if '__' in stem:
        species, metadata = stem.split('__', 1)
    else:
        species, metadata = "", stem
    return species, metadata, os.path.basename(fp)

def open_text(fp: str):
    """
    Open file for reading, handling both regular and gzipped files.
    
    Args:
        fp: File path to open
        
    Returns:
        File handle for reading
    """
    return gzip.open(fp, 'rt') if fp.endswith('.gz') else open(fp, 'r')

def build_bh_lookup(p_values: Iterable[float]) -> Optional[Callable[[float], float]]:
    """
    Build Benjamini-Hochberg FDR correction lookup function.
    
    Given iterable of p-values (NaNs ignored), return a function p -> q_BH (Benjamini–Hochberg).
    Uses the monotone step-down transform; lookup is O(log m). Returns None if no valid p-values.
    
    Args:
        p_values: Iterable of p-values (NaNs will be ignored)
        
    Returns:
        Function that maps p-values to q-values, or None if no valid p-values
    """
    p = np.array([x for x in p_values if pd.notna(x)], dtype=float)
    m = p.size
    if m == 0:
        return None
    p_sorted = np.sort(p)  # ascending
    ranks = np.arange(1, m+1, dtype=float)
    bh = (m / ranks) * p_sorted
    # make monotone (from right to left)
    q_sorted = np.minimum.accumulate(bh[::-1])[::-1]
    # ensure in [0,1]
    q_sorted = np.clip(q_sorted, 0.0, 1.0)

    def q_of(p_single: float) -> float:
        if not np.isfinite(p_single):
            return np.nan
        # index of first p_j >= p_single
        idx = np.searchsorted(p_sorted, p_single, side='left')
        if idx >= m:  # p_single larger than any observed p
            return q_sorted[-1]
        return q_sorted[idx]

    return q_of

def summarize_pyseer(indir: str, out: str, alpha: float = 0.05, 
                     pattern: str = "*.pyseer.tsv*") -> None:
    """
    Summarize significant pyseer hits across files with BH/FDR correction.
    
    Uses streaming approach for memory efficiency with large files.
    Applies per-file FDR correction and filters for significant results.
    
    Args:
        indir: Directory containing *.pyseer.tsv(.gz) files
        out: Output TSV path
        alpha: FDR threshold (default 0.05)
        pattern: Glob pattern inside indir (default: *.pyseer.tsv*)
    """
    logger.info(f"Starting pyseer summarization from {indir}")
    logger.info(f"Output will be written to {out}")
    logger.info(f"FDR threshold: {alpha}")
    
    files = sorted(glob.glob(os.path.join(indir, pattern)))
    if not files:
        logger.warning(f"No files matching {pattern} under {indir}")
        raise SystemExit(f"No files matching {pattern} under {indir}")

    logger.info(f"Found {len(files)} pyseer files to process")
    wrote_header = False
    total_significant = 0

    for fp in files:
        species, metadata, bname = parse_name(fp)
        logger.debug(f"Processing {bname} (species: {species}, metadata: {metadata})")

        # ---------- PASS 1: read only p-value columns ----------
        usecols_p = ['filter-pvalue', 'lrt-pvalue']
        p_filter_vals, p_lrt_vals = [], []
        try:
            for chunk in pd.read_csv(open_text(fp), sep='\t', usecols=usecols_p,
                                     chunksize=500_000, low_memory=False):
                if 'filter-pvalue' in chunk:
                    p_filter_vals.extend(chunk['filter-pvalue'].tolist())
                if 'lrt-pvalue' in chunk:
                    p_lrt_vals.extend(chunk['lrt-pvalue'].tolist())
        except ValueError as e:
            logger.warning(f"{bname}: {e}; skipping.")
            continue

        qf_lookup = build_bh_lookup(p_filter_vals)
        ql_lookup = build_bh_lookup(p_lrt_vals)
        if qf_lookup is None or ql_lookup is None:
            logger.warning(f"{bname}: no valid p-values; skipping.")
            continue

        # ---------- PASS 2: stream, compute q's, filter, append ----------
        # Columns we'd like to keep if present
        keep_cols = [
            'variant','af','filter-pvalue','lrt-pvalue',
            'beta','beta-std-err','intercept','notes'
        ]
        file_significant = 0
        
        for chunk in pd.read_csv(open_text(fp), sep='\t', chunksize=250_000, low_memory=False):
            # ensure columns exist (robust across pyseer versions)
            cols_present = [c for c in keep_cols if c in chunk.columns]
            # compute q's on the fly
            if 'filter-pvalue' in chunk.columns:
                chunk['q_filter_bh'] = [qf_lookup(x) for x in chunk['filter-pvalue'].to_numpy()]
            else:
                chunk['q_filter_bh'] = np.nan
            if 'lrt-pvalue' in chunk.columns:
                chunk['q_lrt_bh'] = [ql_lookup(x) for x in chunk['lrt-pvalue'].to_numpy()]
            else:
                chunk['q_lrt_bh'] = np.nan

            # significance: both q's <= alpha
            mask = (chunk['q_lrt_bh'] <= alpha) & (chunk['q_filter_bh'] <= alpha)
            mask = mask.fillna(False)
            if not mask.any():
                continue

            sub = chunk.loc[mask, cols_present + ['q_filter_bh','q_lrt_bh']].copy()
            sub.insert(0, '__file__', bname)
            sub.insert(1, 'species', species)
            sub.insert(2, 'metadata', metadata)

            # write (append) with header once per entire run
            sub.to_csv(out, sep='\t', index=False, mode='a', header=not wrote_header)
            wrote_header = True
            file_significant += len(sub)
        
        if file_significant > 0:
            logger.info(f"{bname}: found {file_significant} significant results")
            total_significant += file_significant

    if not wrote_header:
        # write an empty header (no hits)
        header = ['__file__','species','metadata',
                  'variant','af','filter-pvalue','q_filter_bh','lrt-pvalue','q_lrt_bh',
                  'beta','beta-std-err','intercept','notes']
        pd.DataFrame(columns=header).to_csv(out, sep='\t', index=False)
        logger.info(f"Wrote 0 significant rows to {out}")
    else:
        logger.info(f"Summary complete: {total_significant} significant results written to {out}")

def main() -> None:
    """
    Command line interface for summarize_pyseer.
    
    Stream-summarize significant pyseer hits with per-file BH/FDR correction.
    Uses memory-efficient streaming approach for large files.
    """
    ap = argparse.ArgumentParser(
        description="Summarize significant pyseer hits across files with BH/FDR correction (memory-efficient streaming)"
    )
    ap.add_argument("-i", "--indir", required=True, 
                    help="Directory containing *.pyseer.tsv(.gz) files")
    ap.add_argument("-o", "--out", required=True, 
                    help="Output TSV path")
    ap.add_argument("--alpha", type=float, default=0.05, 
                    help="FDR threshold (default 0.05)")
    ap.add_argument("--pattern", default="*.pyseer.tsv*", 
                    help="Glob pattern inside indir (default: *.pyseer.tsv*)")
    args = ap.parse_args()
    
    # Configure logging for standalone usage
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    summarize_pyseer(
        indir=args.indir,
        out=args.out,
        alpha=args.alpha,
        pattern=args.pattern
    )

if __name__ == "__main__":
    main()

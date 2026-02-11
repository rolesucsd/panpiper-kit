from typing import Tuple

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import fdrcorrection


def compute_bh_qvalues(pvalues: np.ndarray) -> np.ndarray:
    """
    Compute Benjamini-Hochberg q-values from p-values.

    This is a canonical implementation used throughout panpiper-kit to ensure
    consistent FDR correction across all modules.

    Args:
        pvalues: Array of p-values (NaNs are preserved in output)

    Returns:
        Array of q-values (same shape as pvalues, NaN where input was NaN)

    Example:
        >>> p = np.array([0.01, 0.04, np.nan, 0.03, 0.05])
        >>> q = compute_bh_qvalues(p)
        >>> # q will have adjusted p-values, with NaN preserved at index 2
    """
    # Handle empty or all-NaN input
    pvalues = np.asarray(pvalues, dtype=float)
    is_nan = np.isnan(pvalues)
    valid_p = pvalues[~is_nan]

    if len(valid_p) == 0:
        return np.full_like(pvalues, np.nan)

    m = len(valid_p)
    sorted_idx = np.argsort(valid_p)
    sorted_p = valid_p[sorted_idx]

    # BH correction: q_i = p_i * m / rank_i
    ranks = np.arange(1, m + 1, dtype=float)
    bh_values = sorted_p * m / ranks

    # Make monotone decreasing from right to left (ensures q_i <= q_j for i < j)
    q_sorted = np.minimum.accumulate(bh_values[::-1])[::-1]

    # Clip to [0, 1] range
    q_sorted = np.clip(q_sorted, 0.0, 1.0)

    # Restore original order
    q_values = np.empty(m, dtype=float)
    q_values[sorted_idx] = q_sorted

    # Restore NaNs in original positions
    result = np.full_like(pvalues, np.nan, dtype=float)
    result[~is_nan] = q_values

    return result


def add_bh(in_fp: str, out_fp: str, pcol_guess: Tuple[str, ...] = ('pvalue', 'lrt-pvalue')) -> None:
    """
    Add Benjamini-Hochberg FDR correction to a results file.
    
    Reads a TSV file with p-values, applies BH correction, and writes the results
    with additional columns for q-values and significance flags.
    
    Args:
        in_fp: Input file path containing p-values
        out_fp: Output file path for FDR-corrected results
        pcol_guess: Tuple of possible p-value column names to search for
        
    Raises:
        RuntimeError: If no p-value column is found in the input file
    """
    df = pd.read_csv(in_fp, sep='\t')
    pcol = next((c for c in pcol_guess if c in df.columns), None)
    if pcol is None:
        raise RuntimeError(f'No p-value column in {in_fp}')
    mask = df[pcol].notna()

    # Handle case where all p-values are NaN
    if not mask.any():
        df['qvalue_bh'] = pd.NA
        df['significant_bh_0.05'] = False
    else:
        rej, q = fdrcorrection(df.loc[mask, pcol].astype(float).values, alpha=0.05, method='indep')
        # Initialize with NaN (not 1.0) to preserve missing values
        df['qvalue_bh'] = pd.NA
        df.loc[mask, 'qvalue_bh'] = q
        df['significant_bh_0.05'] = False
        df.loc[mask, 'significant_bh_0.05'] = rej
    df.to_csv(out_fp, sep='\t', index=False)

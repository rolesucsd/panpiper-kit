from typing import Tuple

import pandas as pd
from statsmodels.stats.multitest import fdrcorrection


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
    rej, q = fdrcorrection(df.loc[mask, pcol].astype(float).values, alpha=0.05, method='indep')
    df['qvalue_bh'] = 1.0; df.loc[mask,'qvalue_bh'] = q
    df['significant_bh_0.05'] = False; df.loc[mask,'significant_bh_0.05'] = rej
    df.to_csv(out_fp, sep='\t', index=False)

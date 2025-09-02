import pandas as pd
from statsmodels.stats.multitest import fdrcorrection

def add_bh(in_fp: str, out_fp: str, pcol_guess=('pvalue','lrt-pvalue')):
    df = pd.read_csv(in_fp, sep='\t')
    pcol = next((c for c in pcol_guess if c in df.columns), None)
    if pcol is None:
        raise RuntimeError(f'No p-value column in {in_fp}')
    mask = df[pcol].notna()
    rej, q = fdrcorrection(df.loc[mask, pcol].astype(float).values, alpha=0.05, method='indep')
    df['qvalue_bh'] = 1.0; df.loc[mask,'qvalue_bh'] = q
    df['significant_bh_0.05'] = False; df.loc[mask,'significant_bh_0.05'] = rej
    df.to_csv(out_fp, sep='\t', index=False)

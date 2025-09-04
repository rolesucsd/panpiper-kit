#!/usr/bin/env python3
import argparse, os, re, gzip, glob
import pandas as pd
from statsmodels.stats.multitest import fdrcorrection

def parse_name(fp: str):
    """
    Expect: <species>__<metadata>.pyseer.tsv[.gz]
    Return (species, metadata, basename)
    """
    base = os.path.basename(fp)
    base = re.sub(r'\.gz$', '', base)
    if not base.endswith('.pyseer.tsv'):
        # fallback for odd names: try to peel suffix
        return ("", "", os.path.basename(fp))
    stem = base[:-len('.pyseer.tsv')]
    if '__' in stem:
        species, metadata = stem.split('__', 1)
    else:
        species, metadata = ("", stem)
    return species, metadata, os.path.basename(fp)

def read_table(fp: str) -> pd.DataFrame:
    # tolerate gz
    opener = gzip.open if fp.endswith('.gz') else open
    with opener(fp, 'rt') as fh:
        return pd.read_csv(fh, sep='\t')

def add_fdr(df: pd.DataFrame, alpha: float, scope_key=None):
    """
    Add BH q-values for filter-pvalue and lrt-pvalue.
    If scope_key is provided (e.g., a column tuple), do groupwise correction.
    """
    def _apply(group: pd.DataFrame) -> pd.DataFrame:
        out = group.copy()
        for col, qcol in [('filter-pvalue','q_filter_bh'), ('lrt-pvalue','q_lrt_bh')]:
            if col in out.columns:
                mask = out[col].notna()
                if mask.any():
                    rej, q = fdrcorrection(out.loc[mask, col].astype(float).values, alpha=alpha, method='indep')
                    out[qcol] = 1.0
                    out.loc[mask, qcol] = q
                else:
                    out[qcol] = pd.NA
            else:
                out[qcol] = pd.NA
        return out

    if scope_key is None:
        return _apply(df)
    else:
        return df.groupby(list(scope_key), dropna=False, group_keys=False).apply(_apply)

def summarize_pyseer(indir: str, out: str, alpha: float = 0.05, 
                     pattern: str = "*.pyseer.tsv*"):
    """
    Summarize significant pyseer hits across files with BH/FDR.
    
    Args:
        indir: Directory containing *.pyseer.tsv(.gz) files
        out: Output TSV path
        alpha: FDR threshold (default 0.05)
        pattern: Glob pattern inside indir (default: *.pyseer.tsv*)
    """
    files = sorted(glob.glob(os.path.join(indir, pattern)))
    if not files:
        raise SystemExit(f"No files matching {pattern} under {indir}")

    rows = []
    for fp in files:
        try:
            df = read_table(fp)
        except Exception as e:
            print(f"[warn] skip unreadable {fp}: {e}")
            continue
        species, metadata, bname = parse_name(fp)
        # sanity for required cols
        required = ['variant','af','lrt-pvalue','beta']
        missing = [c for c in required if c not in df.columns]
        if missing:
            print(f"[warn] {fp} missing columns: {missing}; keeping available ones.")
        df['__file__'] = bname
        df['species'] = species
        df['metadata'] = metadata
        rows.append(df)

    if not rows:
        raise SystemExit("No usable pyseer tables.")

    big = pd.concat(rows, ignore_index=True)

    # Add FDRs (per-file)
    big = add_fdr(big, alpha, scope_key=['__file__'])

    # Apply significance filter 
    def sig_mask(df):
        ql = df.get('q_lrt_bh')
        qf = df.get('q_filter_bh')
        return (ql.notna()) & (qf.notna()) & (ql <= alpha) & (qf <= alpha)

    mask = sig_mask(big)
    sig = big.loc[mask].copy()

    # Select and order columns (fall back if some missing)
    keep_cols = [
        '__file__','species','metadata','variant','af',
        'filter-pvalue','q_filter_bh','lrt-pvalue','q_lrt_bh',
        'beta','beta-std-err','intercept','notes'
    ]
    cols = [c for c in keep_cols if c in sig.columns]
    sig = sig[cols].sort_values(['metadata','__file__','q_lrt_bh'], na_position='last')

    # Write
    sig.to_csv(out, sep='\t', index=False)
    print(f"[ok] wrote {len(sig)} significant rows to {out}")
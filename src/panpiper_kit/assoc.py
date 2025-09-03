from typing import Tuple

import pandas as pd
import numpy as np
from skbio.stats.distance import DistanceMatrix, permanova, mantel
from scipy.stats import f_oneway, chi2, f as f_dist

# ---------------------------
# Exact tests (Permutation)
# ---------------------------

def distance_assoc_one(mash_tsv: str, pheno_tsv: str, typ: str, perms: int) -> pd.DataFrame:
    """
    Perform exact distance-based association tests with permutations.
    
    EXACT mode:
      - binary/categorical -> PERMANOVA (permutations)
      - continuous         -> Mantel (Spearman, permutations)
    
    Args:
        mash_tsv: Path to Mash distance matrix TSV file
        pheno_tsv: Path to phenotype TSV file
        typ: Phenotype type ('binary', 'categorical', or 'continuous')
        perms: Number of permutations for statistical tests
        
    Returns:
        Single-row DataFrame with columns: species, metadata, n_samples, test, stat, pvalue, permutations
    """
    Dm = pd.read_csv(mash_tsv, sep='\t', index_col=0)
    # symmetrize and zero diagonal
    Dm = (Dm + Dm.T) / 2
    np.fill_diagonal(Dm.values, 0.0)

    ph = pd.read_csv(pheno_tsv, sep='\t').dropna(subset=['phenotype'])
    # Align samples
    keep = [s for s in Dm.index if s in set(ph['sample'])]
    species = pheno_tsv.split('/')[-1].split('__')[0]
    variable = '__'.join(pheno_tsv.split('/')[-1].split('__')[1:]).replace('.pheno.tsv','')

    if len(keep) < 4:
        return pd.DataFrame([{
            'species': species, 'metadata': variable, 'n_samples': len(keep),
            'test': 'NA', 'stat': np.nan, 'R2': np.nan, 'pvalue': np.nan,
            'permutations': perms
        }])

    Dm = Dm.loc[keep, keep]
    DM = DistanceMatrix(Dm.values, keep)

    if typ in ('binary','categorical'):
        grp = ph.set_index('sample').loc[keep, 'phenotype'].astype(str).values
        res = permanova(dm=DM, grouping=grp, permutations=perms)
        row = dict(species=species, metadata=variable, n_samples=len(keep),
                   test='PERMANOVA', stat=float(res['test statistic']),
                   R2=np.nan, pvalue=float(res['p-value']), permutations=perms)
    else:
        v = ph.set_index('sample').loc[keep, 'phenotype']
        # Check for zero variance, handling NaN values properly
        try:
            std_val = v.std(ddof=0)
            if v.dropna().nunique() < 2 or pd.isna(std_val) or float(std_val) == 0.0:
                row = dict(species=species, metadata=variable, n_samples=len(keep),
                           test='Mantel_spearman', stat=np.nan, R2=np.nan, pvalue=np.nan, permutations=perms)
            else:
                DX = DistanceMatrix(_cont_distance(v), keep)
                r, p, n = mantel(DM, DX, method='spearman', permutations=perms, alternative='two-sided')
                row = dict(species=species, metadata=variable, n_samples=len(keep),
                           test='Mantel_spearman', stat=float(r), R2=np.nan, pvalue=float(p), permutations=perms)
        except (ValueError, TypeError):
            row = dict(species=species, metadata=variable, n_samples=len(keep),
                       test='Mantel_spearman', stat=np.nan, R2=np.nan, pvalue=np.nan, permutations=perms)
    return pd.DataFrame([row])

# ---------------------------
# FAST tests (No permutations)
# ---------------------------

def fast_distance_tests(mash_tsv: str, pheno_tsv: str, typ: str, max_axes: int = 10) -> pd.DataFrame:
    """
    Perform fast distance-based association tests without permutations.
    
    FAST mode (no permutations):
      - Compute PCoA (via Gower centering + eigendecomposition).
      - Categorical/Binary: ANOVA on top K axes; combine p-values with Fisher's method.
      - Continuous: OLS of phenotype ~ top K axes; F-test (model vs null).

    Args:
        mash_tsv: Path to Mash distance matrix TSV file
        pheno_tsv: Path to phenotype TSV file
        typ: Phenotype type ('binary', 'categorical', or 'continuous')
        max_axes: Maximum number of PCoA axes to use in analysis
        
    Returns:
        Single-row DataFrame with columns: n_samples, test, stat, pvalue
    """
    # Load and symmetrize distances
    Dm = pd.read_csv(mash_tsv, sep='\t', index_col=0)
    Dm = (Dm + Dm.T) / 2
    np.fill_diagonal(Dm.values, 0.0)
    labels = list(Dm.index)
    n = len(labels)
    if n < 4:
        return pd.DataFrame([{'n_samples': n, 'test': 'FAST', 'stat': np.nan, 'pvalue': np.nan}])

    # Align phenotype
    ph = pd.read_csv(pheno_tsv, sep='\t').dropna(subset=['phenotype'])
    ph = ph[ph['sample'].isin(labels)].set_index('sample').loc[labels]
    keep_mask = ph['phenotype'].notna().values
    if keep_mask.sum() < 4:
        return pd.DataFrame([{'n_samples': int(keep_mask.sum()), 'test': 'FAST', 'stat': np.nan, 'pvalue': np.nan}])

    # Subset to non-missing samples consistently
    D = Dm.values[np.ix_(keep_mask, keep_mask)]
    y = ph['phenotype'].values[keep_mask]
    labs = np.array(labels)[keep_mask]
    n_eff = len(labs)

    # PCoA: double-center and eigendecompose
    X_pc, eigvals = _pcoa_scores(D, max_axes=max_axes)

    if typ in ('binary','categorical'):
        groups = y.astype(str)
        pvals = []
        for a in range(X_pc.shape[1]):
            buckets = [X_pc[groups == g, a] for g in np.unique(groups)]
            # valid ANOVA only if ≥2 non-empty groups
            if len(buckets) < 2 or any(len(b)==0 for b in buckets):
                continue
            fstat, p = f_oneway(*buckets)
            pvals.append(p)
        if not pvals:
            stat = np.nan; p_comb = np.nan
        else:
            stat = -2.0 * np.sum(np.log(pvals))
            p_comb = 1.0 - chi2.cdf(stat, 2 * len(pvals))
        return pd.DataFrame([{
            'n_samples': n_eff, 'test': 'FAST_ANOVA_PC', 'stat': stat, 'pvalue': p_comb
        }])
    else:
        # Continuous: OLS on PCs, F-test
        y = y.astype(float)
        # drop if zero variance
        if np.nanstd(y) == 0.0:
            return pd.DataFrame([{'n_samples': n_eff, 'test':'FAST_OLS_PC', 'stat': np.nan, 'pvalue': np.nan}])
        X = np.column_stack([np.ones(n_eff), X_pc])  # intercept + PCs
        XtX = X.T @ X
        try:
            beta = np.linalg.solve(XtX, X.T @ y)
        except np.linalg.LinAlgError:
            # fallback: pseudo-inverse
            beta = np.linalg.pinv(XtX) @ (X.T @ y)
        yhat = X @ beta
        rss = np.sum((y - yhat)**2)
        tss = np.sum((y - y.mean())**2)
        p = X_pc.shape[1]
        df1 = p
        df2 = n_eff - (p + 1)
        if df2 <= 0 or tss == 0:
            return pd.DataFrame([{'n_samples': n_eff, 'test':'FAST_OLS_PC', 'stat': np.nan, 'pvalue': np.nan}])
        R2 = 1 - rss / tss
        F = (R2/df1) / ((1-R2)/df2) if (1-R2) > 0 else np.inf
        pval = 1.0 - f_dist.cdf(F, df1, df2)
        return pd.DataFrame([{'n_samples': n_eff, 'test':'FAST_OLS_PC', 'stat': F, 'pvalue': pval}])

# ---------------------------
# Helpers
# ---------------------------

def _cont_distance(vec: pd.Series) -> np.ndarray:
    """
    Compute Euclidean distance matrix from standardized continuous vector.
    
    Args:
        vec: Input pandas Series with continuous values
        
    Returns:
        Distance matrix as numpy array
    """
    # Handle NaN values and zero variance
    vec_clean = vec.dropna()
    if len(vec_clean) == 0 or vec_clean.std(ddof=0) == 0.0:
        # Return zero distance matrix if no valid data or zero variance
        n = len(vec)
        return np.zeros((n, n))
    
    z = (vec_clean - vec_clean.mean()) / vec_clean.std(ddof=0)
    return np.sqrt((z.values[:,None] - z.values[None,:])**2)

def _pcoa_scores(D: np.ndarray, max_axes: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform classical MDS/PCoA on a distance matrix.
    
    Args:
        D: Input distance matrix
        max_axes: Maximum number of axes to return
        
    Returns:
        Tuple of (scores, eigenvalues) keeping only positive-eigenvalue axes (up to max_axes)
    """
    n = D.shape[0]
    # Gower centering
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * (J @ (D**2) @ J)
    # symmetric -> eigendecomposition
    eigvals, eigvecs = np.linalg.eigh(B)       # ascending order
    # keep positive eigenvalues (numerical tolerance)
    tol = 1e-12
    pos = eigvals > tol
    if not np.any(pos):
        # pathological: return first axis anyway to avoid crashes
        idx = np.argsort(eigvals)[::-1][:1]
        L = np.abs(eigvals[idx])
        V = eigvecs[:, idx]
        scores = V * np.sqrt(L)
        return scores[:, :min(max_axes, scores.shape[1])], L
    eigvals = eigvals[pos]
    eigvecs = eigvecs[:, pos]
    # sort descending
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    # scores = V * sqrt(lambda)
    scores = eigvecs * np.sqrt(eigvals)
    k = max(1, min(max_axes, scores.shape[1]))
    return scores[:, :k], eigvals[:k]

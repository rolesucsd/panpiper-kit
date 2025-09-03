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
    print(f"[DEBUG] Starting distance_assoc_one with:")
    print(f"  mash_tsv: {mash_tsv}")
    print(f"  pheno_tsv: {pheno_tsv}")
    print(f"  typ: {typ}")
    print(f"  perms: {perms}")
    
    Dm = pd.read_csv(mash_tsv, sep='\t', index_col=0)
    # Ensure labels are consistently strings to avoid alignment/sort issues
    Dm.index = Dm.index.astype(str)
    Dm.columns = Dm.columns.astype(str)
    print(f"[DEBUG] Loaded distance matrix: {Dm.shape}, samples: {list(Dm.index)[:5]}...")
    # symmetrize and zero diagonal
    Dm = (Dm + Dm.T) / 2
    np.fill_diagonal(Dm.values, 0.0)

    ph = pd.read_csv(pheno_tsv, sep='\t').dropna(subset=['phenotype'])
    print(f"[DEBUG] Loaded phenotype data: {ph.shape}, columns: {list(ph.columns)}")
    print(f"[DEBUG] Phenotype samples: {list(ph['sample'])[:5]}...")
    print(f"[DEBUG] Phenotype values: {ph['phenotype'].value_counts().head()}")
    
    # Align samples
    keep = [s for s in Dm.index if s in set(ph['sample'])]
    species = pheno_tsv.split('/')[-1].split('__')[0]
    variable = '__'.join(pheno_tsv.split('/')[-1].split('__')[1:]).replace('.pheno.tsv','')
    
    print(f"[DEBUG] After alignment: {len(keep)} samples kept out of {len(Dm.index)} distance matrix samples")
    print(f"[DEBUG] Species: {species}, Variable: {variable}")

    if len(keep) < 4:
        print(f"[DEBUG] Too few samples ({len(keep)}) for analysis, returning NA result")
        return pd.DataFrame([{
            'species': species, 'metadata': variable, 'n_samples': len(keep),
            'test': 'NA', 'stat': np.nan, 'R2': np.nan, 'pvalue': np.nan,
            'permutations': perms
        }])

    Dm = Dm.loc[keep, keep]
    DM = DistanceMatrix(Dm.values, keep)

    if typ in ('binary','categorical'):
        print(f"[DEBUG] Running PERMANOVA for {typ} phenotype")
        grp = ph.set_index('sample').loc[keep, 'phenotype'].astype(str).values
        print(f"[DEBUG] Group values: {np.unique(grp, return_counts=True)}")
        res = permanova(dm=DM, grouping=grp, permutations=perms)
        print(f"[DEBUG] PERMANOVA result: {res}")
        row = dict(species=species, metadata=variable, n_samples=len(keep),
                   test='PERMANOVA', stat=float(res['test statistic']),
                   R2=np.nan, pvalue=float(res['p-value']), permutations=perms)
    else:
        print(f"[DEBUG] Running Mantel test for continuous phenotype")
        v = ph.set_index('sample').loc[keep, 'phenotype']
        print(f"[DEBUG] Continuous values: min={v.min()}, max={v.max()}, mean={v.mean()}, std={v.std()}")
        # Check for zero variance, handling NaN values properly
        try:
            std_val = v.std(ddof=0)
            print(f"[DEBUG] Standard deviation: {std_val}, unique values: {v.dropna().nunique()}")
            if v.dropna().nunique() < 2 or pd.isna(std_val) or float(std_val) == 0.0:
                print(f"[DEBUG] Zero variance detected, returning NA result")
                row = dict(species=species, metadata=variable, n_samples=len(keep),
                           test='Mantel_spearman', stat=np.nan, R2=np.nan, pvalue=np.nan, permutations=perms)
            else:
                print(f"[DEBUG] Computing continuous distance matrix")
                DX = DistanceMatrix(_cont_distance(v), keep)
                print(f"[DEBUG] Running Mantel test with {perms} permutations")
                r, p, n = mantel(DM, DX, method='spearman', permutations=perms, alternative='two-sided')
                print(f"[DEBUG] Mantel result: r={r}, p={p}, n={n}")
                row = dict(species=species, metadata=variable, n_samples=len(keep),
                           test='Mantel_spearman', stat=float(r), R2=np.nan, pvalue=float(p), permutations=perms)
        except (ValueError, TypeError) as e:
            print(f"[DEBUG] Error in Mantel test: {e}")
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
    print(f"[DEBUG] Starting fast_distance_tests with:")
    print(f"  mash_tsv: {mash_tsv}")
    print(f"  pheno_tsv: {pheno_tsv}")
    print(f"  typ: {typ}")
    print(f"  max_axes: {max_axes}")
    
    # Load and symmetrize distances
    Dm = pd.read_csv(mash_tsv, sep='\t', index_col=0)
    # Ensure labels are consistently strings to avoid alignment/sort issues
    Dm.index = Dm.index.astype(str)
    Dm.columns = Dm.columns.astype(str)
    print(f"[DEBUG] Loaded distance matrix: {Dm.shape}")
    Dm = (Dm + Dm.T) / 2
    np.fill_diagonal(Dm.values, 0.0)
    labels = list(Dm.index)
    n = len(labels)
    print(f"[DEBUG] Distance matrix has {n} samples")
    if n < 4:
        print(f"[DEBUG] Too few samples ({n}) for analysis")
        return pd.DataFrame([{'n_samples': n, 'test': 'FAST', 'stat': np.nan, 'pvalue': np.nan}])

    # Align phenotype
    ph = pd.read_csv(pheno_tsv, sep='\t').dropna(subset=['phenotype'])
    print(f"[DEBUG] Loaded phenotype data: {ph.shape}")
    ph = ph[ph['sample'].isin(labels)].set_index('sample').loc[labels]
    keep_mask = ph['phenotype'].notna().values
    print(f"[DEBUG] After alignment: {keep_mask.sum()} samples with valid phenotypes")
    if keep_mask.sum() < 4:
        print(f"[DEBUG] Too few samples with valid phenotypes ({keep_mask.sum()})")
        return pd.DataFrame([{'n_samples': int(keep_mask.sum()), 'test': 'FAST', 'stat': np.nan, 'pvalue': np.nan}])

    # Subset to non-missing samples consistently
    D = Dm.values[np.ix_(keep_mask, keep_mask)]
    y = ph['phenotype'].values[keep_mask]
    labs = np.array(labels)[keep_mask]
    n_eff = len(labs)

    # PCoA: double-center and eigendecompose
    print(f"[DEBUG] Computing PCoA with max_axes={max_axes}")
    X_pc, eigvals = _pcoa_scores(D, max_axes=max_axes)
    print(f"[DEBUG] PCoA computed: {X_pc.shape} scores, {len(eigvals)} eigenvalues")

    if typ in ('binary','categorical'):
        print(f"[DEBUG] Running FAST ANOVA for {typ} phenotype")
        groups = y.astype(str)
        print(f"[DEBUG] Group values: {np.unique(groups, return_counts=True)}")
        pvals = []
        for a in range(X_pc.shape[1]):
            buckets = [X_pc[groups == g, a] for g in np.unique(groups)]
            # valid ANOVA only if ≥2 non-empty groups
            if len(buckets) < 2 or any(len(b)==0 for b in buckets):
                print(f"[DEBUG] Skipping axis {a}: insufficient groups")
                continue
            fstat, p = f_oneway(*buckets)
            print(f"[DEBUG] Axis {a}: F={fstat}, p={p}")
            pvals.append(p)
        if not pvals:
            print(f"[DEBUG] No valid ANOVA tests, returning NA")
            stat = np.nan; p_comb = np.nan
        else:
            stat = -2.0 * np.sum(np.log(pvals))
            p_comb = 1.0 - chi2.cdf(stat, 2 * len(pvals))
            print(f"[DEBUG] Combined test: stat={stat}, p={p_comb}")
        return pd.DataFrame([{
            'n_samples': n_eff, 'test': 'FAST_ANOVA_PC', 'stat': stat, 'pvalue': p_comb
        }])
    else:
        print(f"[DEBUG] Running FAST OLS for continuous phenotype")
        # Continuous: OLS on PCs, F-test
        y = y.astype(float)
        print(f"[DEBUG] Continuous values: min={y.min()}, max={y.max()}, mean={y.mean()}, std={y.std()}")
        # drop if zero variance
        if np.nanstd(y) == 0.0:
            print(f"[DEBUG] Zero variance detected, returning NA")
            return pd.DataFrame([{'n_samples': n_eff, 'test':'FAST_OLS_PC', 'stat': np.nan, 'pvalue': np.nan}])
        X = np.column_stack([np.ones(n_eff), X_pc])  # intercept + PCs
        XtX = X.T @ X
        try:
            beta = np.linalg.solve(XtX, X.T @ y)
        except np.linalg.LinAlgError:
            # fallback: pseudo-inverse
            print(f"[DEBUG] Using pseudo-inverse for OLS")
            beta = np.linalg.pinv(XtX) @ (X.T @ y)
        yhat = X @ beta
        rss = np.sum((y - yhat)**2)
        tss = np.sum((y - y.mean())**2)
        p = X_pc.shape[1]
        df1 = p
        df2 = n_eff - (p + 1)
        print(f"[DEBUG] OLS: RSS={rss}, TSS={tss}, df1={df1}, df2={df2}")
        if df2 <= 0 or tss == 0:
            print(f"[DEBUG] Invalid degrees of freedom or zero TSS")
            return pd.DataFrame([{'n_samples': n_eff, 'test':'FAST_OLS_PC', 'stat': np.nan, 'pvalue': np.nan}])
        R2 = 1 - rss / tss
        F = (R2/df1) / ((1-R2)/df2) if (1-R2) > 0 else np.inf
        pval = 1.0 - f_dist.cdf(F, df1, df2)
        print(f"[DEBUG] OLS result: R2={R2}, F={F}, p={pval}")
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

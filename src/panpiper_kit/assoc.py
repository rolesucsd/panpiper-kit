from __future__ import annotations
import os, logging, json
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Iterable
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib

from skbio.stats.distance import DistanceMatrix, permanova, mantel
from scipy.stats import f_oneway, chi2, f as f_dist

# ----------------- logging -----------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ----------------- globals per worker (for backward compatibility) -----------------
# These are maintained for backward compatibility with the multiprocessing worker pool
# In new code, prefer using the DistanceMatrixAnalyzer class directly
_DM: Optional[pd.DataFrame] = None
_DM_IDS: Optional[List[str]] = None
_PCOA_CACHE: Dict[str, Tuple[np.ndarray, np.ndarray, Dict]] = {}  # Cache PCoA results


class DistanceMatrixAnalyzer:
    """
    Encapsulates distance matrix analysis to avoid global state.

    This class provides a cleaner interface for distance-based statistical tests,
    managing distance matrix data and PCoA caching internally.

    Attributes:
        distance_matrix: DataFrame containing pairwise distances
        sample_ids: List of sample identifiers
        pcoa_cache: Dictionary caching PCoA results by distance matrix hash

    Example:
        >>> analyzer = DistanceMatrixAnalyzer(distance_df)
        >>> result = analyzer.permanova_test(phenotype_file, "binary", permutations=999)
    """

    def __init__(self, distance_matrix: pd.DataFrame):
        """
        Initialize analyzer with a distance matrix.

        Args:
            distance_matrix: Square DataFrame of pairwise distances
        """
        # Ensure symmetric and zero diagonal
        dm = distance_matrix.copy()
        dm.index = dm.index.astype(str)
        dm.columns = dm.columns.astype(str)
        dm = (dm + dm.T) / 2.0
        np.fill_diagonal(dm.values, 0.0)

        self.distance_matrix = dm
        self.sample_ids = list(dm.index)
        self.pcoa_cache: Dict[str, Tuple[np.ndarray, np.ndarray, Dict]] = {}

    def align_common(self, phenotype_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Align distance matrix with phenotype data.

        Args:
            phenotype_df: DataFrame with 'sample' and 'phenotype' columns

        Returns:
            Tuple of (distance_submatrix, phenotype_values, common_sample_ids)
        """
        ph = phenotype_df.dropna(subset=['phenotype']).copy()
        ph['sample'] = ph['sample'].astype(str)

        # Keep samples in matrix order
        ph_samples = set(ph['sample'])
        keep_ids = [sid for sid in self.sample_ids if sid in ph_samples]

        if len(keep_ids) < MIN_SAMPLES_DISTANCE_TEST:
            return np.empty((0, 0)), np.array([]), []

        y = ph.set_index('sample').loc[keep_ids, 'phenotype'].values
        D = self.distance_matrix.loc[keep_ids, keep_ids].values
        return D, y, keep_ids

# ----------------- merge constants -----------------
MERGE_KEY_COLS = ["species", "metadata", "type", "pheno_tsv"]
EXACT_PREFIXED_KEY_COLS = ["exact_species", "exact_metadata", "exact_type", "exact_pheno_tsv"]


def _prepare_exact_df_for_merge(exact_df: pd.DataFrame, exact_rows: List[Dict]) -> pd.DataFrame:
    """Return a DataFrame with prefixed columns expected by merge.

    If exact_df is empty or missing required keys, return an empty
    DataFrame with the expected prefixed key columns so the merge
    succeeds with left joins producing NaNs.
    """
    if exact_df.empty:
        logger.debug("exact_df is empty, creating empty pref with expected columns")
        return pd.DataFrame(columns=EXACT_PREFIXED_KEY_COLS)

    missing_cols = [c for c in MERGE_KEY_COLS if c not in exact_df.columns]
    if missing_cols:
        logger.error(f"exact_df missing required columns: {missing_cols}. Available: {list(exact_df.columns)}")
        logger.error(f"exact_rows sample (first 2): {exact_rows[:2] if exact_rows else '[]'}")
        return pd.DataFrame(columns=EXACT_PREFIXED_KEY_COLS)

    pref = exact_df.add_prefix("exact_")
    logger.debug(f"pref shape after add_prefix: {pref.shape}, columns: {list(pref.columns)}")
    return pref

# ----------------- constants -----------------
# Minimum samples for distance-based test (4 needed for df > 0 in most tests)
MIN_SAMPLES_DISTANCE_TEST = 4

# FAST screening p-value threshold
# Rationale: Balance sensitivity (keep promising tests) vs compute cost
DEFAULT_FAST_P_THRESH = 0.50

# Early stopping threshold - stop permutations if clearly not significant
DEFAULT_EARLY_STOP_P = 0.20

# Escalation thresholds for adaptive permutation ladder
DEFAULT_ESCALATE_P1 = 0.10           # if p <= this at 199, go to 999
DEFAULT_ESCALATE_P2 = 0.05           # if p <= this at 999, go to 9999
DEFAULT_ESCALATE_P3 = 0.01           # if p <= this at 9999, go to 99999

# Default permutation ladder (can be extended to millions)
DEFAULT_PERM_LADDER = (199, 999, 9999)
DEFAULT_PERM_LADDER_EXTENDED = (199, 999, 9999, 99999, 999999)

# PCoA settings
DEFAULT_MAX_AXES = 10
EIGEN_TOL = 1e-12
PCOA_LARGE_CORRECTION_WARN_THRESHOLD = 0.1  # Warn if Lingoes correction > 10% of max eigenvalue

# Large permutation settings
LARGE_PERM_THRESHOLD = 10000  # Use custom implementation above this
LARGE_PERM_CHUNK_SIZE = 10000  # Process permutations in chunks of this size
ANALYTICAL_N_THRESHOLD = 500  # Use analytical approximation for n > this

# --------------- worker init ----------------
def _init_worker(mash_tsv: str):
    """
    Load Mash distance matrix once per process.

    This function initializes global state for multiprocessing workers.
    For new code, prefer using DistanceMatrixAnalyzer class directly.

    Args:
        mash_tsv: Path to distance matrix TSV file
    """
    global _DM, _DM_IDS, _PCOA_CACHE
    Dm = pd.read_csv(mash_tsv, sep="\t", index_col=0)
    Dm.index = Dm.index.astype(str)
    Dm.columns = Dm.columns.astype(str)
    Dm = (Dm + Dm.T) / 2.0
    np.fill_diagonal(Dm.values, 0.0)
    _DM = Dm
    _DM_IDS = list(Dm.index)
    _PCOA_CACHE = {}  # Reset cache for new worker

# --------------- small utils ----------------
def _align_common(phenotype_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Align distance matrix with phenotype data (legacy function using globals).

    Args:
        phenotype_df: DataFrame with 'sample' and 'phenotype' columns

    Returns:
        Tuple of (distance_submatrix, phenotype_values, common_sample_ids)

    Note:
        This function uses global _DM and _DM_IDS for backward compatibility.
        New code should use DistanceMatrixAnalyzer.align_common() instead.
    """
    assert _DM is not None and _DM_IDS is not None
    phenotype_df = phenotype_df.dropna(subset=['phenotype']).copy()
    phenotype_df['sample'] = phenotype_df['sample'].astype(str)
    # keep in matrix order for consistency
    # Pre-compute set for O(1) lookup instead of O(n)
    phenotype_samples = set(phenotype_df['sample'])
    keep_ids = [sample_id for sample_id in _DM_IDS if sample_id in phenotype_samples]
    if len(keep_ids) < MIN_SAMPLES_DISTANCE_TEST:
        return np.empty((0, 0)), np.array([]), []
    phenotype_values = phenotype_df.set_index('sample').loc[keep_ids, 'phenotype'].values
    distance_submatrix = _DM.loc[keep_ids, keep_ids].values
    return distance_submatrix, phenotype_values, keep_ids

def _hash_distance_matrix(D: np.ndarray, ids: List[str]) -> str:
    """Generate hash for distance matrix to enable caching."""
    # Combine matrix data and IDs for unique hash
    h = hashlib.sha256()
    h.update(D.tobytes())
    h.update(''.join(ids).encode())
    return h.hexdigest()[:16]  # Use first 16 chars for shorter key

def _continuous_phenotype_distance(phenotype_series: pd.Series) -> np.ndarray:
    """
    Compute Euclidean distance matrix from continuous phenotype values.

    Standardizes values to z-scores before computing pairwise distances.

    Args:
        phenotype_series: Series of continuous phenotype values

    Returns:
        Square distance matrix (n x n) based on standardized values
    """
    values = phenotype_series.dropna().astype(float).values
    if len(values) == 0 or np.nanstd(values) == 0.0:
        n = len(phenotype_series)
        return np.zeros((n, n))
    # Standardize to z-scores
    z_scores = (values - np.mean(values)) / (np.std(values, ddof=0))
    # Compute pairwise Euclidean distances
    return np.sqrt((z_scores[:, None] - z_scores[None, :]) ** 2)

def _pcoa_scores(D: np.ndarray, max_axes: int = DEFAULT_MAX_AXES, euclid_correction: str = "lingoes",
                 ids: Optional[List[str]] = None, use_cache: bool = True
                 ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    PCoA with optional Lingoes correction; returns (scores, eigvals, diag).

    Supports caching to avoid recomputing PCoA for the same distance matrix.
    Warns if Lingoes correction is substantial.
    """
    global _PCOA_CACHE

    # Check cache if enabled
    cache_key = None
    if use_cache and ids is not None:
        cache_key = _hash_distance_matrix(D, ids)
        if cache_key in _PCOA_CACHE:
            cached_scores, cached_eigvals, cached_diag = _PCOA_CACHE[cache_key]
            # Return cached results, but respect max_axes
            k = max(1, min(max_axes, cached_scores.shape[1]))
            return cached_scores[:, :k], cached_eigvals[:k], cached_diag

    n = D.shape[0]
    J = np.eye(n) - np.ones((n, n))/n
    B = -0.5 * (J @ (D**2) @ J)
    eigvals, eigvecs = np.linalg.eigh(B)  # ascending
    neg_sum = float(np.abs(eigvals[eigvals < 0]).sum()) if np.any(eigvals < 0) else 0.0

    correction_magnitude = 0.0
    if euclid_correction == "lingoes" and np.any(eigvals < -EIGEN_TOL):
        c = -float(eigvals.min()) + 1e-12
        correction_magnitude = c
        eigvals = eigvals + c

        # Warn if correction is substantial
        max_eigval = eigvals.max() if len(eigvals) > 0 else 1.0
        if max_eigval > 0 and (c / max_eigval) > PCOA_LARGE_CORRECTION_WARN_THRESHOLD:
            logger.warning(
                f"Large Lingoes correction applied (c={c:.4f}, {100*c/max_eigval:.1f}% of max eigenvalue). "
                f"Distance matrix may not be Euclidean. Consider checking for outliers or data quality issues."
            )

    pos_mask = eigvals > EIGEN_TOL
    if not np.any(pos_mask):
        # pathological; keep dominant
        idx = np.argsort(eigvals)[::-1][:1]
        L = np.abs(eigvals[idx])
        V = eigvecs[:, idx]
        scores = V * np.sqrt(L)
        diag = {"neg_inertia": neg_sum, "k": scores.shape[1], "correction": correction_magnitude}
        return scores, L, diag

    eigvals = eigvals[pos_mask]
    eigvecs = eigvecs[:, pos_mask]
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    scores = eigvecs * np.sqrt(eigvals)
    k = max(1, min(max_axes, scores.shape[1]))
    diag = {"neg_inertia": neg_sum, "k": k, "correction": correction_magnitude}

    # Cache full results
    if use_cache and cache_key is not None:
        _PCOA_CACHE[cache_key] = (scores, eigvals, diag)

    return scores[:, :k], eigvals[:k], diag

# --------------- FAST tests (screen) ----------------
def _fast_test(pheno_tsv: str, typ: str, max_axes: int = DEFAULT_MAX_AXES
               ) -> Dict[str, float]:
    ph = pd.read_csv(pheno_tsv, sep="\t")
    D, y, ids = _align_common(ph)
    if len(ids) < MIN_SAMPLES_DISTANCE_TEST:
        return {"n_samples": len(ids), "test": "FAST", "stat": np.nan, "pvalue": np.nan}

    # Use caching for PCoA
    Xpc, eigvals, diag = _pcoa_scores(D, max_axes=max_axes, euclid_correction="lingoes", ids=ids, use_cache=True)
    n = len(ids)
    if typ in ("binary", "categorical"):
        groups = y.astype(str)
        pvals = []
        for a in range(Xpc.shape[1]):
            buckets = [Xpc[groups == g, a] for g in np.unique(groups)]
            if len(buckets) < 2 or any(len(b) == 0 for b in buckets):
                continue
            fstat, p = f_oneway(*buckets)
            if np.isfinite(p):
                pvals.append(p)
        if not pvals:
            stat = np.nan; p_comb = np.nan
        else:
            stat = -2.0 * np.sum(np.log(pvals))
            p_comb = 1.0 - chi2.cdf(stat, 2 * len(pvals))
        return {"n_samples": n, "test": "FAST_ANOVA_PC", "stat": float(stat), "pvalue": float(p_comb),
                "neg_inertia": diag["neg_inertia"], "k_axes": diag["k"]}
    else:
        y = y.astype(float)
        if np.nanstd(y) == 0.0:
            return {"n_samples": n, "test": "FAST_OLS_PC", "stat": np.nan, "pvalue": np.nan,
                    "neg_inertia": diag["neg_inertia"], "k_axes": diag["k"]}
        X = np.column_stack([np.ones(n), Xpc])
        XtX = X.T @ X
        try:
            beta = np.linalg.solve(XtX, X.T @ y)
        except np.linalg.LinAlgError:
            beta = np.linalg.pinv(XtX) @ (X.T @ y)
        yhat = X @ beta
        rss = float(np.sum((y - yhat) ** 2))
        tss = float(np.sum((y - np.nanmean(y)) ** 2))
        p = Xpc.shape[1]
        df1 = p
        df2 = n - (p + 1)

        # Validate degrees of freedom
        if df1 <= 0 or df2 <= 0 or tss == 0:
            return {"n_samples": n, "test": "FAST_OLS_PC", "stat": np.nan, "pvalue": np.nan,
                    "neg_inertia": diag["neg_inertia"], "k_axes": diag["k"]}

        R2 = 1 - rss / tss

        # Calculate F-statistic with proper handling of edge cases
        if (1 - R2) <= 0:
            F = np.inf
            pval = 0.0  # Perfect fit -> p-value is 0
        else:
            F = (R2 / df1) / ((1 - R2) / df2)
            pval = 1.0 - f_dist.cdf(F, df1, df2)
        return {"n_samples": n, "test": "FAST_OLS_PC", "stat": float(F), "pvalue": float(pval),
                "neg_inertia": diag["neg_inertia"], "k_axes": diag["k"], "R2": float(R2)}

# --------------- Custom PERMANOVA for large permutations --------------
def _compute_permanova_F_stat(DM_array: np.ndarray, grouping: np.ndarray) -> float:
    """
    Compute pseudo-F statistic for PERMANOVA.

    This implements the core PERMANOVA calculation without using scikit-bio,
    allowing for custom permutation strategies.
    """
    n = len(grouping)
    groups = np.unique(grouping)
    k = len(groups)

    if k < 2 or n < MIN_SAMPLES_DISTANCE_TEST:
        return np.nan

    # Gower's centered matrix: G = -0.5 * D^2
    G = -0.5 * (DM_array ** 2)

    # Total sum of squares
    grand_mean = np.mean(G)
    total_SS = np.sum((G - grand_mean) ** 2) / n

    # Within-group sum of squares
    within_SS = 0.0
    for g in groups:
        mask = grouping == g
        ng = np.sum(mask)
        if ng > 0:
            G_sub = G[np.ix_(mask, mask)]
            group_mean = np.mean(G_sub)
            within_SS += np.sum((G_sub - group_mean) ** 2) / ng

    # Between-group sum of squares
    between_SS = total_SS - within_SS

    # Degrees of freedom
    df_between = k - 1
    df_within = n - k

    if df_within <= 0 or within_SS == 0:
        return np.nan

    # Pseudo-F statistic
    F = (between_SS / df_between) / (within_SS / df_within)
    return float(F)


def _permanova_chunk(DM_array: np.ndarray, grouping: np.ndarray, n_perms: int,
                     obs_F: float, seed: Optional[int] = None) -> int:
    """
    Run a chunk of permutations and return count of exceedances.

    This function is designed to be called in parallel for large permutation tests.
    """
    rng = np.random.default_rng(seed)
    exceedance_count = 0

    for _ in range(n_perms):
        perm_grouping = rng.permutation(grouping)
        perm_F = _compute_permanova_F_stat(DM_array, perm_grouping)
        if np.isfinite(perm_F) and perm_F >= obs_F:
            exceedance_count += 1

    return exceedance_count


def _execute_permutation_chunks(DM_array: np.ndarray, grouping: np.ndarray,
                                 n_chunks: int, chunk_size: int, obs_F: float,
                                 n_jobs: int) -> int:
    """
    Execute permutation chunks either in parallel or sequentially.

    Args:
        DM_array: Distance matrix as numpy array
        grouping: Group labels as numpy array
        n_chunks: Number of chunks to execute
        chunk_size: Number of permutations per chunk
        obs_F: Observed F-statistic
        n_jobs: Number of parallel jobs (1 = sequential)

    Returns:
        Total count of permutations that exceeded observed statistic
    """
    total_exceedances = 0

    if n_jobs > 1:
        # Parallel execution
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = []
            for _ in range(n_chunks):
                seed = np.random.randint(0, 2**31 - 1)
                fut = executor.submit(_permanova_chunk, DM_array, grouping, chunk_size, obs_F, seed)
                futures.append(fut)

            for fut in as_completed(futures):
                try:
                    total_exceedances += fut.result()
                except Exception as e:
                    logger.error(f"Permutation chunk failed: {e}")
    else:
        # Sequential execution
        for _ in range(n_chunks):
            seed = np.random.randint(0, 2**31 - 1)
            total_exceedances += _permanova_chunk(DM_array, grouping, chunk_size, obs_F, seed)

    return total_exceedances


def _permanova_large_perms(DM_array: np.ndarray, grouping: np.ndarray,
                           permutations: int = 100000,
                           chunk_size: int = LARGE_PERM_CHUNK_SIZE,
                           n_jobs: int = 1) -> Dict[str, float]:
    """
    Custom PERMANOVA implementation supporting very large permutation counts (1M+).

    Uses chunked permutations with optional parallelization to handle millions
    of permutations efficiently without memory overflow.

    Args:
        DM_array: Distance matrix as numpy array
        grouping: Group labels as numpy array
        permutations: Total number of permutations to run
        chunk_size: Number of permutations per chunk
        n_jobs: Number of parallel jobs (1 = sequential)

    Returns:
        Dictionary with test statistic, p-value, and metadata
    """
    try:
        # Compute observed statistic
        obs_F = _compute_permanova_F_stat(DM_array, grouping)

        if not np.isfinite(obs_F):
            return {
                "n_samples": len(grouping),
                "test": "PERMANOVA_large",
                "stat": np.nan,
                "pvalue": np.nan,
                "permutations": permutations,
                "error": "Could not compute observed F statistic"
            }

        # Calculate number of chunks
        n_chunks = max(1, permutations // chunk_size)
        actual_perms = n_chunks * chunk_size

        if actual_perms != permutations:
            logger.debug(f"Adjusted permutations from {permutations} to {actual_perms} (multiple of chunk size)")

        # Execute permutation chunks
        total_exceedances = _execute_permutation_chunks(
            DM_array, grouping, n_chunks, chunk_size, obs_F, n_jobs
        )

        # Calculate p-value with "+1" correction
        pvalue = (total_exceedances + 1) / (actual_perms + 1)

        return {
            "n_samples": len(grouping),
            "test": "PERMANOVA_large",
            "stat": float(obs_F),
            "pvalue": float(pvalue),
            "permutations": actual_perms
        }

    except (ValueError, TypeError, np.linalg.LinAlgError) as e:
        # Log error but return structured result for robustness
        logger.error(f"Large PERMANOVA failed: {e}", exc_info=True)
        return {
            "n_samples": len(grouping),
            "test": "PERMANOVA_large",
            "stat": np.nan,
            "pvalue": np.nan,
            "permutations": permutations,
            "error": str(e)
        }


def _permanova_analytical(DM_array: np.ndarray, grouping: np.ndarray) -> Dict[str, float]:
    """
    Analytical PERMANOVA approximation using F-distribution.

    For large sample sizes (n > 500), the PERMANOVA pseudo-F statistic
    approximately follows an F-distribution. This provides instant results
    without permutations.

    Note: This is an approximation and may be slightly anticonservative for
    complex group structures or small effect sizes.
    """
    try:
        obs_F = _compute_permanova_F_stat(DM_array, grouping)

        if not np.isfinite(obs_F):
            return {
                "n_samples": len(grouping),
                "test": "PERMANOVA_analytical",
                "stat": np.nan,
                "pvalue": np.nan,
                "method": "F_approximation",
                "error": "Could not compute F statistic"
            }

        n = len(grouping)
        k = len(np.unique(grouping))
        df1 = k - 1
        df2 = n - k

        # Use F-distribution CDF
        pvalue = 1.0 - f_dist.cdf(obs_F, df1, df2)

        return {
            "n_samples": n,
            "test": "PERMANOVA_analytical",
            "stat": float(obs_F),
            "pvalue": float(pvalue),
            "method": "F_approximation",
            "df1": df1,
            "df2": df2
        }

    except (ValueError, TypeError, np.linalg.LinAlgError) as e:
        # Log error but return structured result for robustness
        logger.error(f"Analytical PERMANOVA failed: {e}", exc_info=True)
        return {
            "n_samples": len(grouping),
            "test": "PERMANOVA_analytical",
            "stat": np.nan,
            "pvalue": np.nan,
            "method": "F_approximation",
            "error": str(e)
        }


# --------------- EXACT tests (permutations) --------------
def _permutation_test(pheno_tsv: str, typ: str, perms: int, mode: str = "auto") -> Dict[str, float]:
    """
    One-shot exact test at a given permutations count.

    Args:
        pheno_tsv: Path to phenotype file
        typ: Type of phenotype (binary, categorical, continuous)
        perms: Number of permutations
        mode: Testing mode - "auto" (adaptive), "standard" (scikit-bio),
              "large" (custom for >10k), "analytical" (F-distribution approx)

    Returns:
        Dictionary with test results
    """
    assert _DM is not None
    ph = pd.read_csv(pheno_tsv, sep="\t").dropna(subset=['phenotype'])

    # Align in distance-matrix order
    keep = [sid for sid in _DM.index if sid in set(ph['sample'])]
    if len(keep) < MIN_SAMPLES_DISTANCE_TEST:
        return {"n_samples": len(keep), "test": "NA", "stat": np.nan, "pvalue": np.nan, "permutations": perms}

    mat = _DM.loc[keep, keep].to_numpy(dtype=float, copy=True)  # C-contiguous
    mat = 0.5 * (mat + mat.T)                                  # enforce symmetry
    np.fill_diagonal(mat, 0.0)

    if typ in ("binary", "categorical"):
        grp = ph.set_index('sample').loc[keep, 'phenotype'].astype(str).to_numpy()
        n = len(keep)

        # Choose method based on mode and sample size
        if mode == "analytical" or (mode == "auto" and n > ANALYTICAL_N_THRESHOLD):
            # Use analytical F-approximation for large n
            result = _permanova_analytical(mat, grp)
            logger.debug(f"Using analytical PERMANOVA (n={n})")
        elif mode == "large" or (mode == "auto" and perms > LARGE_PERM_THRESHOLD):
            # Use custom implementation for large permutation counts
            result = _permanova_large_perms(mat, grp, permutations=perms, n_jobs=1)
            logger.debug(f"Using large-permutation PERMANOVA (perms={perms})")
        else:
            # Use standard scikit-bio implementation
            try:
                DM = DistanceMatrix(mat, ids=keep)
                res = permanova(DM, grouping=grp, permutations=perms)
                result = {
                    "n_samples": len(keep),
                    "test": "PERMANOVA",
                    "stat": float(res['test statistic']),
                    "pvalue": float(res['p-value']),
                    "permutations": perms
                }
            except Exception as e:
                logger.error(f"Standard PERMANOVA failed: {e}")
                result = {
                    "n_samples": len(keep),
                    "test": "PERMANOVA",
                    "stat": np.nan,
                    "pvalue": np.nan,
                    "permutations": perms,
                    "error": str(e)
                }

        return result

    else:  # continuous
        v = ph.set_index('sample').loc[keep, 'phenotype']
        if v.dropna().nunique() < 2:
            return {"n_samples": len(keep), "test": "Mantel_spearman",
                    "stat": np.nan, "pvalue": np.nan, "permutations": perms}

        try:
            DM = DistanceMatrix(mat, ids=keep)
            phenotype_distances = _continuous_phenotype_distance(v)
            DX = DistanceMatrix(np.ascontiguousarray(phenotype_distances, dtype=float), ids=keep)
            r, p, n = mantel(DM, DX, method='spearman', permutations=perms, alternative='two-sided')
            return {"n_samples": int(n), "test": "Mantel_spearman", "stat": float(r),
                    "pvalue": float(p), "permutations": perms}
        except Exception as e:
            logger.error(f"Mantel test failed: {e}")
            return {"n_samples": len(keep), "test": "Mantel_spearman",
                    "stat": np.nan, "pvalue": np.nan, "permutations": perms, "error": str(e)}


def _should_escalate_permutations(rung_index: int, pvalue: float,
                                  early_stop_p: float, escalate_p1: float,
                                  escalate_p2: float, escalate_p3: float) -> Tuple[bool, str]:
    """
    Determine whether to escalate to next permutation level.

    Args:
        rung_index: Current position in permutation ladder (0-based)
        pvalue: P-value from current permutation test
        early_stop_p: Threshold for early stopping at rung 0
        escalate_p1-p3: Escalation thresholds for rungs 0-2

    Returns:
        Tuple of (should_escalate, reason_message)
    """
    # Early stop at first rung if not promising
    if rung_index == 0:
        if np.isnan(pvalue) or pvalue > early_stop_p:
            return False, f"Early stop: p={pvalue:.4f} > {early_stop_p}"
        if pvalue <= escalate_p1:
            return True, f"Escalating from rung {rung_index}: p={pvalue:.4f} <= {escalate_p1}"
        return False, f"Stopping at rung {rung_index}: p={pvalue:.4f}"

    # Escalation decisions for subsequent rungs
    if rung_index == 1 and pvalue <= escalate_p2:
        return True, f"Escalating from rung {rung_index}: p={pvalue:.4f} <= {escalate_p2}"
    if rung_index == 2 and pvalue <= escalate_p3:
        return True, f"Escalating from rung {rung_index}: p={pvalue:.4f} <= {escalate_p3}"

    # Default: stop if we're past first rung and no escalation threshold met
    if rung_index >= 1:
        return False, f"Stopping at rung {rung_index}: p={pvalue:.4f}"

    return False, "Unknown stopping condition"


def _adaptive_exact(pheno_tsv: str, typ: str,
                    perm_ladder: Iterable[int] = DEFAULT_PERM_LADDER,
                    early_stop_p: float = DEFAULT_EARLY_STOP_P,
                    escalate_p1: float = DEFAULT_ESCALATE_P1,
                    escalate_p2: float = DEFAULT_ESCALATE_P2,
                    escalate_p3: float = DEFAULT_ESCALATE_P3,
                    mode: str = "auto") -> Dict[str, float]:
    """
    Adaptive permutation ladder with intelligent escalation.

    Strategy:
      - Start with smallest permutation count (e.g., 199)
      - Stop early if clearly not significant
      - Escalate to higher permutation counts if result is promising
      - Supports extended ladder with multiple escalation thresholds

    Args:
        pheno_tsv: Path to phenotype file
        typ: Phenotype type (binary, categorical, continuous)
        perm_ladder: Permutation escalation ladder (e.g., [199, 999, 9999])
        early_stop_p: P-value threshold for early stopping at first rung
        escalate_p1-p3: P-value thresholds for escalating through ladder
        mode: Testing mode (auto, standard, large, analytical)

    Returns:
        Dictionary with final test results from highest rung reached
    """
    results = {}
    perm_list = list(perm_ladder)

    for rung_index, perms in enumerate(perm_list):
        try:
            res = _permutation_test(pheno_tsv, typ, perms=perms, mode=mode)
            p_value = res.get("pvalue", np.nan)
            results = res

            # Determine whether to escalate or stop
            should_escalate, reason = _should_escalate_permutations(
                rung_index, p_value, early_stop_p, escalate_p1, escalate_p2, escalate_p3
            )

            logger.debug(reason)

            if not should_escalate:
                break

        except Exception as e:
            logger.error(f"Adaptive exact test failed at permutation level {perms}: {e}")
            results = {
                "n_samples": np.nan,
                "test": "NA",
                "stat": np.nan,
                "pvalue": np.nan,
                "permutations": perms,
                "error": str(e)
            }
            break

    return results

# --------------- Orchestrator helpers -----------------
def _run_fast_screening(phenos: List['PhenotypeJob'], mash_tsv: str,
                        n_workers: int, fast_max_axes: int) -> pd.DataFrame:
    """
    Run FAST screening tests in parallel across all phenotypes.

    Args:
        phenos: List of phenotype jobs to test
        mash_tsv: Path to distance matrix file
        n_workers: Number of parallel workers
        fast_max_axes: Maximum PCoA axes to use

    Returns:
        DataFrame with FAST screening results
    """
    logger.info(f"[FAST] screening {len(phenos)} phenotypes using {n_workers} workers")
    fast_rows: List[Dict[str, object]] = []

    with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker, initargs=(mash_tsv,)) as ex:
        fut2meta = {}
        for job in phenos:
            fut = ex.submit(_fast_test, job.pheno_tsv, job.typ, fast_max_axes)
            fut2meta[fut] = job

        for fut in as_completed(fut2meta):
            job = fut2meta[fut]
            try:
                out = fut.result()
            except Exception as e:
                out = {"n_samples": np.nan, "test":"FAST", "stat": np.nan, "pvalue": np.nan, "error": str(e)}
            out.update({"species": job.species, "metadata": job.variable, "type": job.typ, "pheno_tsv": job.pheno_tsv})
            fast_rows.append(out)

    return pd.DataFrame(fast_rows)


def _run_exact_tests(promising_df: pd.DataFrame, mash_tsv: str, n_workers: int,
                     perm_ladder: Iterable[int], early_stop_p: float,
                     escalate_p1: float, escalate_p2: float, escalate_p3: float,
                     perm_mode: str) -> pd.DataFrame:
    """
    Run adaptive exact permutation tests on promising phenotypes.

    Args:
        promising_df: DataFrame of phenotypes that passed FAST screening
        mash_tsv: Path to distance matrix file
        n_workers: Number of parallel workers
        perm_ladder: Permutation escalation ladder
        early_stop_p: P-value threshold for early stopping
        escalate_p1-p3: P-value thresholds for escalating permutations
        perm_mode: Testing mode (auto, standard, large, analytical)

    Returns:
        DataFrame with exact test results
    """
    exact_rows: List[Dict[str, object]] = []

    if len(promising_df) == 0:
        return pd.DataFrame(exact_rows)

    with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker, initargs=(mash_tsv,)) as ex:
        fut2meta = {}
        for _, r in promising_df.iterrows():
            fut = ex.submit(_adaptive_exact, r['pheno_tsv'], r['type'],
                            perm_ladder, early_stop_p, escalate_p1, escalate_p2, escalate_p3, perm_mode)
            fut2meta[fut] = (r['species'], r['metadata'], r['type'], r['pheno_tsv'])

        for fut in as_completed(fut2meta):
            sp, var, typ, pth = fut2meta[fut]
            try:
                res = fut.result()
            except Exception as e:
                logger.error(f"Exact test failed for {sp}__{var}: {e}", exc_info=True)
                res = {"n_samples": np.nan, "test":"NA", "stat": np.nan, "pvalue": np.nan,
                       "permutations": np.nan, "error": str(e)}
            res.update({"species": sp, "metadata": var, "type": typ, "pheno_tsv": pth})
            exact_rows.append(res)

    exact_df = pd.DataFrame(exact_rows)

    if not exact_df.empty:
        logger.debug(f"exact_df shape: {exact_df.shape}, columns: {list(exact_df.columns)}")

    return exact_df


def _merge_fast_exact_results(fast_df: pd.DataFrame, exact_df: pd.DataFrame,
                               exact_rows: List[Dict[str, float]]) -> pd.DataFrame:
    """
    Merge FAST screening results with exact test results.

    Args:
        fast_df: DataFrame with FAST screening results
        exact_df: DataFrame with exact test results
        exact_rows: List of exact result dictionaries (for error handling)

    Returns:
        Merged DataFrame with standardized column order
    """
    pref = _prepare_exact_df_for_merge(exact_df, exact_rows)

    out = fast_df.merge(
        pref,
        left_on=MERGE_KEY_COLS,
        right_on=EXACT_PREFIXED_KEY_COLS,
        how="left"
    )

    # Standardize output columns
    keep_cols = [
        "species","metadata","type","pheno_tsv",
        "n_samples","test","stat","pvalue","neg_inertia","k_axes","R2",
        "exact_n_samples","exact_test","exact_stat","exact_pvalue","exact_permutations","exact_error"
    ]
    for c in keep_cols:
        if c not in out.columns:
            out[c] = np.nan

    return out[keep_cols].sort_values(["species","metadata"])


# --------------- Orchestrator -----------------
@dataclass
class PhenotypeJob:
    species: str
    variable: str
    typ: str
    pheno_tsv: str

def run_assoc(
    mash_tsv: str,
    phenos: List[PhenotypeJob],
    *,
    n_workers: int = os.cpu_count() or 4,
    fast_p_thresh: float = DEFAULT_FAST_P_THRESH,
    fast_max_axes: int = DEFAULT_MAX_AXES,
    perm_ladder: Iterable[int] = DEFAULT_PERM_LADDER,
    early_stop_p: float = DEFAULT_EARLY_STOP_P,
    escalate_p1: float = DEFAULT_ESCALATE_P1,
    escalate_p2: float = DEFAULT_ESCALATE_P2,
    escalate_p3: float = DEFAULT_ESCALATE_P3,
    perm_mode: str = "auto",
) -> pd.DataFrame:
    """
    Parallel pipeline for genome-wide association testing.

    Two-stage workflow:
      1) FAST screening: Quick PCoA-based tests across all phenotypes
      2) EXACT testing: Adaptive permutation tests for promising candidates

    Args:
        mash_tsv: Path to distance matrix TSV file
        phenos: List of PhenotypeJob objects defining tests to run
        n_workers: Number of parallel workers
        fast_p_thresh: P-value threshold for advancing to exact tests
        fast_max_axes: Maximum PCoA axes for FAST screening
        perm_ladder: Permutation escalation ladder (e.g., [199, 999, 9999])
        early_stop_p: P-value threshold for early stopping
        escalate_p1-p3: P-value thresholds for escalating through ladder
        perm_mode: Testing mode - auto, standard, large, or analytical

    Returns:
        DataFrame with merged FAST and EXACT results, sorted by species and metadata
    """
    # Stage 1: FAST screening
    fast_df = _run_fast_screening(phenos, mash_tsv, n_workers, fast_max_axes)

    # Filter to promising phenotypes
    keep_mask = (fast_df['pvalue'] <= fast_p_thresh) & fast_df['pvalue'].notna()
    promising_df = fast_df[keep_mask].copy()
    logger.info(f"[FAST] {keep_mask.sum()} / {len(fast_df)} pass (p ≤ {fast_p_thresh}) → exact tests")

    # Stage 2: EXACT testing on promising subset
    exact_df = _run_exact_tests(
        promising_df, mash_tsv, n_workers,
        perm_ladder, early_stop_p, escalate_p1, escalate_p2, escalate_p3, perm_mode
    )

    # Merge and return results
    exact_rows = exact_df.to_dict('records') if not exact_df.empty else []
    return _merge_fast_exact_results(fast_df, exact_df, exact_rows)

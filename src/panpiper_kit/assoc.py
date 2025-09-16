from __future__ import annotations
import os, logging, json
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Iterable
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

from skbio.stats.distance import DistanceMatrix, permanova, mantel
from scipy.stats import f_oneway, chi2, f as f_dist

# ----------------- logging -----------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ----------------- globals per worker -----------------
_DM: Optional[pd.DataFrame] = None
_DM_IDS: Optional[List[str]] = None

# ----------------- knobs -----------------
DEFAULT_FAST_P_THRESH = 0.50         # pass FAST screen if p <= this
DEFAULT_EARLY_STOP_P = 0.20          # after 199 perms, stop if p > this
DEFAULT_ESCALATE_P1 = 0.10           # if p <= this at 199, go to 999
DEFAULT_ESCALATE_P2 = 0.05           # if p <= this at 999, go to 9999
DEFAULT_PERM_LADDER = (199, 999, 9999)
DEFAULT_MAX_AXES = 10
EIGEN_TOL = 1e-12

# --------------- worker init ----------------
def _init_worker(mash_tsv: str):
    """Load Mash distance matrix once per process."""
    global _DM, _DM_IDS
    Dm = pd.read_csv(mash_tsv, sep="\t", index_col=0)
    Dm.index = Dm.index.astype(str)
    Dm.columns = Dm.columns.astype(str)
    Dm = (Dm + Dm.T) / 2.0
    np.fill_diagonal(Dm.values, 0.0)
    _DM = Dm
    _DM_IDS = list(Dm.index)

# --------------- small utils ----------------
def _align_common(ph: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Return (D_sub, y, ids) aligned to samples with non-missing phenotype."""
    assert _DM is not None and _DM_IDS is not None
    ph = ph.dropna(subset=['phenotype']).copy()
    ph['sample'] = ph['sample'].astype(str)
    # keep in matrix order for consistency
    keep_ids = [sid for sid in _DM_IDS if sid in set(ph['sample'])]
    if len(keep_ids) < 4:
        return np.empty((0,0)), np.array([]), []
    y = ph.set_index('sample').loc[keep_ids, 'phenotype'].values
    D = _DM.loc[keep_ids, keep_ids].values
    return D, y, keep_ids

def _cont_distance(vec: pd.Series) -> np.ndarray:
    x = vec.dropna().astype(float).values
    if len(x) == 0 or np.nanstd(x) == 0.0:
        n = len(vec)
        return np.zeros((n, n))
    z = (x - np.mean(x)) / (np.std(x, ddof=0))
    return np.sqrt((z[:, None] - z[None, :]) ** 2)

def _pcoa_scores(D: np.ndarray, max_axes: int = DEFAULT_MAX_AXES, euclid_correction: str = "lingoes"
                 ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """PCoA with optional Lingoes correction; returns (scores, eigvals, diag)."""
    n = D.shape[0]
    J = np.eye(n) - np.ones((n, n))/n
    B = -0.5 * (J @ (D**2) @ J)
    eigvals, eigvecs = np.linalg.eigh(B)  # ascending
    neg_sum = float(np.abs(eigvals[eigvals < 0]).sum()) if np.any(eigvals < 0) else 0.0

    if euclid_correction == "lingoes" and np.any(eigvals < -EIGEN_TOL):
        c = -float(eigvals.min()) + 1e-12
        eigvals = eigvals + c

    pos_mask = eigvals > EIGEN_TOL
    if not np.any(pos_mask):
        # pathological; keep dominant
        idx = np.argsort(eigvals)[::-1][:1]
        L = np.abs(eigvals[idx])
        V = eigvecs[:, idx]
        scores = V * np.sqrt(L)
        return scores, L, {"neg_inertia": neg_sum, "k": scores.shape[1]}

    eigvals = eigvals[pos_mask]
    eigvecs = eigvecs[:, pos_mask]
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    scores = eigvecs * np.sqrt(eigvals)
    k = max(1, min(max_axes, scores.shape[1]))
    return scores[:, :k], eigvals[:k], {"neg_inertia": neg_sum, "k": k}

# --------------- FAST tests (screen) ----------------
def _fast_test(pheno_tsv: str, typ: str, max_axes: int = DEFAULT_MAX_AXES
               ) -> Dict[str, object]:
    ph = pd.read_csv(pheno_tsv, sep="\t")
    D, y, ids = _align_common(ph)
    if len(ids) < 4:
        return {"n_samples": len(ids), "test": "FAST", "stat": np.nan, "pvalue": np.nan}

    Xpc, eigvals, diag = _pcoa_scores(D, max_axes=max_axes, euclid_correction="lingoes")
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
        if df2 <= 0 or tss == 0:
            return {"n_samples": n, "test": "FAST_OLS_PC", "stat": np.nan, "pvalue": np.nan,
                    "neg_inertia": diag["neg_inertia"], "k_axes": diag["k"]}
        R2 = 1 - rss / tss
        F = (R2/df1) / ((1-R2)/df2) if (1-R2) > 0 else np.inf
        pval = 1.0 - f_dist.cdf(F, df1, df2)
        return {"n_samples": n, "test": "FAST_OLS_PC", "stat": float(F), "pvalue": float(pval),
                "neg_inertia": diag["neg_inertia"], "k_axes": diag["k"], "R2": float(R2)}

# --------------- EXACT tests (permutations) --------------
def _permutation_test(pheno_tsv: str, typ: str, perms: int) -> Dict[str, object]:
    """One-shot exact test at a given permutations count."""
    assert _DM is not None
    ph = pd.read_csv(pheno_tsv, sep="\t").dropna(subset=['phenotype'])
    # align
    keep = [sid for sid in _DM.index if sid in set(ph['sample'])]
    if len(keep) < 4:
        return {"n_samples": len(keep), "test": "NA", "stat": np.nan, "pvalue": np.nan, "permutations": perms}
    DM = DistanceMatrix(_DM.loc[keep, keep].values, keep)
    if typ in ("binary", "categorical"):
        grp = ph.set_index('sample').loc[keep, 'phenotype'].astype(str).values
        res = permanova(DM, grouping=grp, permutations=perms)
        return {
            "n_samples": len(keep),
            "test": "PERMANOVA",
            "stat": float(res['test statistic']),
            "pvalue": float(res['p-value']),
            "permutations": perms
        }
    else:
        v = ph.set_index('sample').loc[keep, 'phenotype']
        if v.dropna().nunique() < 2:
            return {"n_samples": len(keep), "test": "Mantel_spearman",
                    "stat": np.nan, "pvalue": np.nan, "permutations": perms}
        DX = DistanceMatrix(_cont_distance(v), keep)
        r, p, n = mantel(DM, DX, method='spearman', permutations=perms, alternative='two-sided')
        return {"n_samples": n, "test": "Mantel_spearman", "stat": float(r),
                "pvalue": float(p), "permutations": perms}

def _adaptive_exact(pheno_tsv: str, typ: str,
                    perm_ladder: Iterable[int] = DEFAULT_PERM_LADDER,
                    early_stop_p: float = DEFAULT_EARLY_STOP_P,
                    escalate_p1: float = DEFAULT_ESCALATE_P1,
                    escalate_p2: float = DEFAULT_ESCALATE_P2) -> Dict[str, object]:
    """
    Adaptive permutation ladder:
      - run smallest perms,
      - stop early if clearly not significant,
      - escalate if promising.
    """
    results = {}
    p_last = None
    for i, perms in enumerate(perm_ladder):
        res = _permutation_test(pheno_tsv, typ, perms=perms)
        p_last = res.get("pvalue", np.nan)
        results = res
        # early stop after first rung if unpromising
        if i == 0 and (np.isnan(p_last) or p_last > early_stop_p):
            break
        # escalate decisions
        if i == 0 and p_last <= escalate_p1:
            continue
        if i == 1 and p_last <= escalate_p2:
            continue
        # otherwise stop
        if i >= 1:
            break
    return results

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
) -> pd.DataFrame:
    """
    Parallel pipeline:
      1) FAST screen across all phenotypes
      2) Adaptive exact test for promising phenotypes
    Returns one DataFrame with FAST + EXACT fields merged.
    """
    # --- FAST in parallel ---
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
    fast_df = pd.DataFrame(fast_rows)

    # --- choose promising ---
    keep_mask = (fast_df['pvalue'] <= fast_p_thresh) & fast_df['pvalue'].notna()
    todo = fast_df[keep_mask].copy()
    logger.info(f"[FAST] {keep_mask.sum()} / {len(fast_df)} pass (p ≤ {fast_p_thresh}) → exact tests")

    # --- EXACT (adaptive) in parallel ---
    exact_rows: List[Dict[str, object]] = []
    if len(todo):
        with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker, initargs=(mash_tsv,)) as ex:
            fut2meta = {}
            for _, r in todo.iterrows():
                fut = ex.submit(_adaptive_exact, r['pheno_tsv'], r['type'],
                                perm_ladder, early_stop_p, escalate_p1, escalate_p2)
                fut2meta[fut] = (r['species'], r['metadata'], r['type'], r['pheno_tsv'])
            for fut in as_completed(fut2meta):
                sp, var, typ, pth = fut2meta[fut]
                try:
                    res = fut.result()
                except Exception as e:
                    res = {"n_samples": np.nan, "test":"NA", "stat": np.nan, "pvalue": np.nan,
                           "permutations": np.nan, "error": str(e)}
                res.update({"species": sp, "metadata": var, "type": typ, "pheno_tsv": pth})
                exact_rows.append(res)
    exact_df = pd.DataFrame(exact_rows)

    # --- merge & return ---
    out = fast_df.merge(
        exact_df.add_prefix("exact_"),
        left_on=["species","metadata","type","pheno_tsv"],
        right_on=["exact_species","exact_metadata","exact_type","exact_pheno_tsv"],
        how="left"
    )
    # pretty columns
    keep_cols = [
        "species","metadata","type","pheno_tsv",
        "n_samples","test","stat","pvalue","neg_inertia","k_axes","R2",
        "exact_n_samples","exact_test","exact_stat","exact_pvalue","exact_permutations","exact_error"
    ]
    for c in keep_cols:
        if c not in out.columns: out[c] = np.nan
    return out[keep_cols].sort_values(["species","metadata"])

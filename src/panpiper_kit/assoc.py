import pandas as pd, numpy as np
from skbio.stats.distance import DistanceMatrix, permanova, mantel

def _meta_matrix_cont(v: pd.Series):
    z = (v - v.mean())/v.std(ddof=0)
    M = np.sqrt((z.values[:,None]-z.values[None,:])**2)
    return M

def distance_assoc_one(mash_tsv: str, pheno_tsv: str, typ: str, perms: int) -> pd.DataFrame:
    Dm = pd.read_csv(mash_tsv, sep='\t', index_col=0)
    Dm = (Dm + Dm.T)/2; np.fill_diagonal(Dm.values, 0.0)
    ph = pd.read_csv(pheno_tsv, sep='\t').dropna(subset=['phenotype'])
    keep = [s for s in Dm.index if s in set(ph['sample'])]
    species = pheno_tsv.split('/')[-1].split('__')[0]
    variable = '__'.join(pheno_tsv.split('/')[-1].split('__')[1:]).replace('.pheno.tsv','')
    if len(keep) < 4:
        return pd.DataFrame([{'species': species, 'metadata': variable, 'n_samples': len(keep),
                              'test': 'NA', 'stat': np.nan, 'R2': np.nan, 'pvalue': np.nan,
                              'permutations': perms}])
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
        if v.dropna().nunique() < 2 or float(v.std(ddof=0)) == 0.0:
            row = dict(species=species, metadata=variable, n_samples=len(keep),
                       test='Mantel_spearman', stat=np.nan, R2=np.nan, pvalue=np.nan, permutations=perms)
        else:
            DX = DistanceMatrix(_meta_matrix_cont(v), keep)
            r,p,n = mantel(DM, DX, method='spearman', permutations=perms, alternative='two-sided')
            row = dict(species=species, metadata=variable, n_samples=len(keep),
                       test='Mantel_spearman', stat=float(r), R2=np.nan, pvalue=float(p), permutations=perms)
    return pd.DataFrame([row])

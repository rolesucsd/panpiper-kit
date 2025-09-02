import os
import re
import pathlib
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np


def _classify(s: pd.Series, min_unique_cont: int) -> str:
    """
    Classify a pandas Series as binary, continuous, or categorical.
    
    Args:
        s: Input pandas Series to classify
        min_unique_cont: Minimum number of unique values required for continuous classification
        
    Returns:
        Classification string: 'binary', 'continuous', or 'categorical'
    """
    vals = s.dropna().unique()
    if len(vals) == 2: 
        return 'binary'
    if pd.api.types.is_numeric_dtype(s) and s.dropna().nunique() >= min_unique_cont:
        return 'continuous'
    return 'categorical'

def filter_metadata_per_species(
        metadata_fp: str, 
        ani_map_fp: str, 
        out_dir: str,
        min_n: int = 6, 
        max_missing_frac: float = 0.2,
        min_level_n: int = 3, 
        min_unique_cont: int = 6
    ) -> Dict[str, List[Tuple[str, str, str]]]:
    """
    Filter metadata per species and create phenotype files for analysis.
    
    This function processes metadata and ANI mapping files to create filtered
    phenotype files for each species that meet the specified criteria.
    
    Args:
        metadata_fp: Path to metadata TSV file containing sample information
        ani_map_fp: Path to ANI mapping TSV file with sample->species assignments
        out_dir: Output directory for filtered phenotype files
        min_n: Minimum number of samples required per species
        max_missing_frac: Maximum fraction of missing values allowed per phenotype
        min_level_n: Minimum number of samples required per category level
        min_unique_cont: Minimum number of unique values for continuous phenotypes
        
    Returns:
        Dictionary mapping species names to lists of (variable, type, filepath) tuples
        
    Raises:
        RuntimeError: If metadata file doesn't contain 'sample' column
    """
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(metadata_fp, sep='\t').drop_duplicates(subset=['sample'])
    if 'sample' not in meta.columns:
        raise RuntimeError("metadata must contain 'sample' which maps to to the FASTA filename")
    ani = pd.read_csv(ani_map_fp, sep='\t', names=['sample','species'])
    df = ani.merge(meta, on='sample', how='inner')
    out_index = {}
    for species, sub in df.groupby('species', sort=False):
        rows = []
        usable = [c for c in sub.columns if c not in ('sample','species')]
        for col in usable:
            s = sub[['sample', col]].copy()
            miss_frac = s[col].isna().mean()
            if miss_frac > max_missing_frac: continue
            s = s.dropna(subset=[col])
            if len(s) < min_n: continue
            typ = _classify(s[col], min_unique_cont)
            if typ == 'binary':
                vc = s[col].astype(str).value_counts()
                if (vc < min_level_n).any(): continue
            elif typ == 'categorical':
                vc = s[col].astype(str).value_counts()
                keep_levels = vc[vc >= min_level_n].index
                s = s[s[col].astype(str).isin(keep_levels)]
                if s[col].nunique() < 2 or len(s) < min_n: continue
            else:
                if s[col].dropna().nunique() < min_unique_cont: continue
                if float(s[col].std(ddof=0)) == 0.0: continue
            p = pathlib.Path(out_dir)/f"{species}__{re.sub(r'[^A-Za-z0-9_.-]+','_',col)}.pheno.tsv"
            s.rename(columns={col:'phenotype'}).to_csv(p, sep='\t', index=False)
            rows.append((col, typ, str(p)))
        out_index[species] = rows
        idxp = pathlib.Path(out_dir)/f"{species}.list.tsv"
        with open(idxp,'w') as fh:
            fh.write('species\tvariable\ttype\tpheno_tsv\n')
            for (v,t,pth) in rows: fh.write(f"{species}\t{v}\t{t}\t{pth}\n")
    return out_index

import os
import re
import pathlib
from typing import Dict, List, Tuple, Optional

import pandas as pd
import numpy as np

# Constants for phenotype filtering
DEFAULT_MIN_N = 6
DEFAULT_MAX_MISSING_FRAC = 0.2
DEFAULT_MIN_LEVEL_N = 3
DEFAULT_MIN_UNIQUE_CONT = 6


def _pick_col(cols_lower_map: Dict[str, str], candidates: List[str]) -> str:
    """
    Find the first column in df whose lowercased name contains any of the candidate substrings.
    Raises KeyError if not found.
    
    Args:
        cols_lower_map: Dictionary mapping lowercase column names to original names
        candidates: List of candidate substrings to search for
        
    Returns:
        Original column name that matches one of the candidates
        
    Raises:
        KeyError: If no column matches any of the candidates
    """
    for low, orig in cols_lower_map.items():
        if any(cand in low for cand in candidates):
            return orig
    raise KeyError(f"Missing required column matching any of: {candidates}")


def _extract_patient_from_bin(bin_name: str) -> str:
    """
    Extract patient name from bin identifier.
    
    Handles multiple formats:
    - {patient}_{binner}_{bin_identifier} -> returns {patient}
    - {patient}.{id}_{binner}_{bin_identifier} -> returns {patient}.{id}
    
    Examples:
    - Patient1_metabat_001 -> Patient1
    - 10317.X00179178_CONCOCT_bin.40 -> 10317.X00179178
    
    Args:
        bin_name: Bin identifier string
        
    Returns:
        Patient name extracted from bin identifier
    """
    # Split by underscore
    parts = bin_name.split('_')
    
    # If we have at least 2 parts, the patient is everything before the last 2 parts
    # This handles both formats:
    # - Patient1_metabat_001 -> Patient1
    # - 10317.X00179178_CONCOCT_bin.40 -> 10317.X00179178
    if len(parts) >= 2:
        return '_'.join(parts[:-2])
    else:
        # Fallback: if only one part, return the whole thing
        return bin_name


def _clean_metadata_values(series: pd.Series, custom_missing_values: Optional[List[str]] = None) -> pd.Series:
    """
    Clean metadata values by converting common "missing" indicators to NaN.
    
    Converts values like "Not collected", "Not available", "Unknown", etc. to NaN
    so they are properly treated as missing values rather than valid categories.
    
    Args:
        series: Input pandas Series to clean
        custom_missing_values: Additional missing value indicators to treat as NaN
        
    Returns:
        Series with cleaned values (missing indicators converted to NaN)
    """
    # Common missing value indicators (case-insensitive)
    missing_indicators = {
        'not collected', 'not available', 'unknown', 'n/a', 'na', 'none', 
        'missing', 'not specified', 'not provided', 'not reported',
        'not applicable', 'not determined', 'not tested', 'not done',
        'pending', 'tbd', 'to be determined', 'not recorded',
        'not documented', 'not assessed', 'not evaluated'
    }
    
    # Add custom missing values if provided
    if custom_missing_values:
        missing_indicators.update({v.lower().strip() for v in custom_missing_values})
    
    # Convert to lowercase for comparison
    series_str = series.astype(str).str.lower().str.strip()
    
    # Replace missing indicators with NaN
    mask = series_str.isin(missing_indicators)
    cleaned_series = series.copy()
    # Explicitly cast to object dtype to avoid dtype incompatibility warning
    if cleaned_series.dtype == 'bool':
        cleaned_series = cleaned_series.astype('object')
    cleaned_series.loc[mask] = pd.NA
    
    return cleaned_series


def filter_by_checkm(s2p: Dict[str, str], checkm_fp: str, comp_min: float, cont_max: float) -> Dict[str, str]:
    """
    Filter samples based on CheckM quality metrics.
    
    Given a sample->fasta map and a CheckM-like TSV file, returns a filtered sample->fasta map
    containing only samples that meet the specified completeness and contamination thresholds.
    Accepts flexible headers, e.g. sample/bin/genome, completeness/comp, contamination/contam.
    
    Args:
        s2p: Dictionary mapping sample names to FASTA file paths
        checkm_fp: Path to CheckM TSV file with quality metrics
        comp_min: Minimum completeness percentage to keep (default 80.0)
        cont_max: Maximum contamination percentage to keep (default 10.0)
        
    Returns:
        Filtered dictionary mapping sample names to FASTA file paths
        
    Raises:
        KeyError: If required columns are not found in CheckM file
    """
    df = pd.read_csv(checkm_fp, sep="\t", dtype=str, low_memory=False)
    if df.empty:
        return {}

    cols_map = {c.lower(): c for c in df.columns}
    s_col = _pick_col(cols_map, ["sample", "bin", "genome", "id", "name"])
    c_col = _pick_col(cols_map, ["completeness", "comp"])
    t_col = _pick_col(cols_map, ["contamination", "contam"])

    comp = pd.to_numeric(df[c_col], errors="coerce")
    cont = pd.to_numeric(df[t_col], errors="coerce")
    keep_names = set(df.loc[(comp >= comp_min) & (cont <= cont_max), s_col].astype(str))
    if not keep_names:
        return {}

    # Keep only samples present in s2p and in keep_names
    kept = {s: p for s, p in s2p.items() if s in keep_names}
    return kept

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


def _is_valid_phenotype(s: pd.Series, typ: str, max_missing_frac: float, min_n: int, 
                       min_level_n: int, min_unique_cont: int) -> bool:
    """
    Check if a phenotype series meets the filtering criteria.
    
    Args:
        s: Phenotype series to validate
        typ: Phenotype type ('binary', 'categorical', 'continuous')
        max_missing_frac: Maximum fraction of missing values allowed
        min_n: Minimum number of samples required
        min_level_n: Minimum samples per category level
        min_unique_cont: Minimum unique values for continuous phenotypes
        
    Returns:
        True if phenotype meets criteria, False otherwise
    """
    # Check missing value fraction
    if s.isna().mean() > max_missing_frac:
        return False
    
    # Check minimum sample count
    s_clean = s.dropna()
    if len(s_clean) < min_n:
        return False
    
    # Type-specific validation
    if typ == 'binary':
        vc = s_clean.astype(str).value_counts()
        return not (vc < min_level_n).any()
    elif typ == 'categorical':
        vc = s_clean.astype(str).value_counts()
        keep_levels = vc[vc >= min_level_n].index
        s_filtered = s_clean[s_clean.astype(str).isin(keep_levels)]
        return s_filtered.nunique() >= 2 and len(s_filtered) >= min_n
    else:  # continuous
        if s_clean.nunique() < min_unique_cont:
            return False
        try:
            std_val = s_clean.std(ddof=0, skipna=True)
            return not (pd.isna(std_val) or float(std_val) == 0.0)
        except (ValueError, TypeError):
            return False


def _process_species_phenotypes(sub: pd.DataFrame, species: str, sample_col: str, 
                               out_dir: str, max_missing_frac: float, min_n: int,
                               min_level_n: int, min_unique_cont: int) -> List[Tuple[str, str, str]]:
    """
    Process phenotypes for a single species and create phenotype files.
    
    Args:
        sub: DataFrame subset for this species
        species: Species name
        sample_col: Name of the sample column
        out_dir: Output directory for phenotype files
        max_missing_frac: Maximum fraction of missing values allowed
        min_n: Minimum number of samples required
        min_level_n: Minimum samples per category level
        min_unique_cont: Minimum unique values for continuous phenotypes
        
    Returns:
        List of (variable, type, filepath) tuples for valid phenotypes
    """
    rows = []
    exclude_cols = {'species', 'bin_identifier', 'patient', sample_col}
    usable_cols = [c for c in sub.columns if c not in exclude_cols]
    
    for col in usable_cols:
        s = sub[['bin_identifier', col]].copy()
        typ = _classify(s[col], min_unique_cont)
        
        if not _is_valid_phenotype(s[col], typ, max_missing_frac, min_n, min_level_n, min_unique_cont):
            continue
            
        # Handle categorical filtering
        if typ == 'categorical':
            vc = s[col].astype(str).value_counts()
            keep_levels = vc[vc >= min_level_n].index
            s = s[s[col].astype(str).isin(keep_levels)]
        
        # Create phenotype file
        p = pathlib.Path(out_dir) / f"{species}__{re.sub(r'[^A-Za-z0-9_.-]+','_',col)}.pheno.tsv"
        s.rename(columns={'bin_identifier': 'sample', col: 'phenotype'}).to_csv(p, sep='\t', index=False)
        rows.append((col, typ, str(p)))
    
    return rows

def filter_metadata_per_species(
        metadata_fp: str, 
        ani_map_fp: str, 
        out_dir: str,
        min_n: int = DEFAULT_MIN_N, 
        max_missing_frac: float = DEFAULT_MAX_MISSING_FRAC,
        min_level_n: int = DEFAULT_MIN_LEVEL_N, 
        min_unique_cont: int = DEFAULT_MIN_UNIQUE_CONT,
        custom_missing_values: Optional[List[str]] = None
    ) -> Dict[str, List[Tuple[str, str, str]]]:
    """
    Filter metadata per species and create phenotype files for analysis.
    
    This function processes metadata and ANI mapping files to create filtered
    phenotype files for each species that meet the specified criteria.
    
    Handles naming scheme where:
    - Metadata has SampleID column with patient names
    - FASTA files are named like {patient}_{binner}_{bin_identifier}.fa
    - ANI map maps bin identifiers to species
    
    Args:
        metadata_fp: Path to metadata TSV file containing patient information
        ani_map_fp: Path to ANI mapping TSV file with species->bin_identifier assignments
        out_dir: Output directory for filtered phenotype files
        min_n: Minimum number of samples required per species
        max_missing_frac: Maximum fraction of missing values allowed per phenotype
        min_level_n: Minimum number of samples required per category level
        min_unique_cont: Minimum number of unique values for continuous phenotypes
        custom_missing_values: Additional missing value indicators to treat as NaN
        
    Returns:
        Dictionary mapping species names to lists of (variable, type, filepath) tuples
        
    Raises:
        RuntimeError: If metadata file doesn't contain 'SampleID' column
    """
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    
    # Load metadata and handle flexible column naming
    meta = pd.read_csv(metadata_fp, sep='\t', low_memory=False)
    cols_map = {c.lower(): c for c in meta.columns}
    
    # Find the sample/patient ID column
    try:
        sample_col = _pick_col(cols_map, ['sampleid', 'sample', 'patient', 'id'])
    except KeyError:
        raise RuntimeError("metadata must contain 'SampleID' or similar column")
    
    meta = meta.drop_duplicates(subset=[sample_col])
    
    # Clean metadata values - convert "missing" indicators to NaN
    for col in meta.columns:
        if col != sample_col:  # Don't clean the sample ID column
            meta[col] = _clean_metadata_values(meta[col], custom_missing_values)
    
    # Load ANI mapping (species -> bin_identifier)
    ani = pd.read_csv(ani_map_fp, sep='\t', names=['species','bin_identifier'])
    
    # Extract patient names from bin identifiers and merge
    ani['patient'] = ani['bin_identifier'].apply(_extract_patient_from_bin)
    df = ani.merge(meta, left_on='patient', right_on=sample_col, how='inner')
    
    out_index = {}
    for species, sub in df.groupby('species', sort=False):
        rows = _process_species_phenotypes(sub, species, sample_col, out_dir, 
                                         max_missing_frac, min_n, min_level_n, min_unique_cont)
        out_index[species] = rows
        
        # Create manifest file
        idxp = pathlib.Path(out_dir) / f"{species}.list.tsv"
        with open(idxp, 'w') as fh:
            fh.write('species\tvariable\ttype\tpheno_tsv\n')
            for (v, t, pth) in rows:
                fh.write(f"{species}\t{v}\t{t}\t{pth}\n")
    
    return out_index

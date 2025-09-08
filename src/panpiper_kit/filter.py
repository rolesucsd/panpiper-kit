import os
import re
import pathlib
from typing import Dict, List, Tuple, Optional

import pandas as pd
import numpy as np
import json

# Constants for phenotype filtering
DEFAULT_MIN_N = 6
DEFAULT_MAX_MISSING_FRAC = 0.2
DEFAULT_MIN_LEVEL_N = 3
DEFAULT_MIN_UNIQUE_CONT = 10

def _filter_numeric_outliers(
    s: pd.Series,
    iqr_factor: float = 3.0,
) -> Tuple[pd.Series, int, Dict[str, float]]:
    """
    For numeric series, mark extreme outliers as NaN.
    Returns (cleaned_series, n_outliers, meta).
    Non-numeric input is returned unchanged with n_outliers=0.
    """
    if not pd.api.types.is_numeric_dtype(s):
        return s, 0, {}

    x = s.astype(float)
    # Only IQR-based filtering is supported currently
    meta = {"method": "iqr"}

    q1 = x.quantile(0.25)
    q3 = x.quantile(0.75)
    iqr = q3 - q1
    meta.update({"q1": float(q1) if pd.notna(q1) else None,
                 "q3": float(q3) if pd.notna(q3) else None,
                 "iqr": float(iqr) if pd.notna(iqr) else None,
                 "iqr_factor": float(iqr_factor)})
    if pd.isna(iqr) or iqr == 0:
        return s, 0, meta
    lo = q1 - iqr_factor * iqr
    hi = q3 + iqr_factor * iqr
    mask = (x < lo) | (x > hi)

    cleaned = s.copy()
    n_out = int(mask.sum(skipna=True))
    if n_out > 0:
        cleaned.loc[mask] = pd.NA
    return cleaned, n_out, meta

def _series_numeric_coerce(s: pd.Series) -> pd.Series:
    """Try numeric coercion without throwing off non-numeric columns."""
    if pd.api.types.is_numeric_dtype(s):
        return s
    # If "mostly" numeric, coerce; else return as-is
    coerced = pd.to_numeric(s, errors="coerce")
    # Heuristic: if at least half of non-null values convert, treat as numeric
    non_null = s.notna().sum()
    if non_null > 0 and coerced.notna().sum() >= 0.5 * non_null:
        return coerced
    return s

def _classify(s: pd.Series, min_unique_cont: int) -> str:
    """
    Classify a pandas Series as binary, continuous, or categorical.
    Tries gentle numeric coercion for better detection of continuous vars.
    """
    s2 = _series_numeric_coerce(s)
    vals = s2.dropna().unique()
    if len(vals) == 2:
        return 'binary'
    if pd.api.types.is_numeric_dtype(s2) and s2.dropna().nunique() >= min_unique_cont:
        return 'continuous'
    return 'categorical'

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


def _process_species_phenotypes(
    sub: pd.DataFrame,
    species: str,
    sample_col: str,
    out_dir: str,
    max_missing_frac: float,
    min_n: int,
    min_level_n: int,
    min_unique_cont: int,
    iqr_factor: float = 3.0,
) -> List[Tuple[str, str, str]]:
    """
    Process phenotypes for a single species and create phenotype files.
    Also writes a comprehensive per-variable summary TSV:
      <out_dir>/<species>.pheno_summary.tsv
    """
    rows: List[Tuple[str, str, str]] = []
    exclude_cols = {'species', 'bin_identifier', 'patient', sample_col}
    usable_cols = [c for c in sub.columns if c not in exclude_cols]

    summary_records = []

    for col in usable_cols:
        rec = {
            "species": species,
            "variable": col,
            "type": None,
            "n_total": int(len(sub)),
            "n_missing": None,
            "missing_frac": None,
            "n_kept_after_level_filter": None,
            "n_levels": None,
            "level_counts_json": None,
            "continuous_mean": None,
            "continuous_sd": None,
            "continuous_min": None,
            "continuous_max": None,
            "outlier_method": None,
            "outliers_removed": 0,
            "passes_filters": False,
            "fail_reason": "",
            "pheno_tsv": None,
        }

        s = sub[['bin_identifier', col]].copy()
        typ = _classify(s[col], min_unique_cont)
        rec["type"] = typ

        # If continuous, (optionally) filter outliers -> set to NaN
        if typ == "continuous":
            s_num = _series_numeric_coerce(s[col])
            cleaned, n_out, meta = _filter_numeric_outliers(
                s_num, iqr_factor=iqr_factor
            )
            # keep original col name; update the dataframe
            s[col] = cleaned
            rec["outlier_method"] = meta.get("method")
            rec["outliers_removed"] = int(n_out)

        # Compute basic missing stats *after* outlier filtering
        s_clean = s[col]
        rec["n_missing"] = int(s_clean.isna().sum())
        rec["missing_frac"] = float(s_clean.isna().mean())

        # Early missing-fraction fail
        if rec["missing_frac"] > max_missing_frac:
            rec["fail_reason"] = f"missing_frac>{max_missing_frac}"
            summary_records.append(rec)
            continue

        # Now drop NaNs for type-specific checks
        s_nonnull = s.dropna(subset=[col])

        # Type-specific stats/filters
        if typ in ("binary", "categorical"):
            vc = s_nonnull[col].astype(str).value_counts().sort_index()
            rec["n_levels"] = int(len(vc))
            rec["level_counts_json"] = json.dumps(vc.to_dict())

            # For binary, both levels must have >= min_level_n and overall N >= min_n
            if typ == "binary":
                if (len(vc) != 2) or (vc.min() < min_level_n) or (len(s_nonnull) < min_n):
                    rec["fail_reason"] = "insufficient per-level counts or total N for binary"
                    summary_records.append(rec)
                    continue

            # For categorical, drop levels < min_level_n then require >=2 levels and N>=min_n
            if typ == "categorical":
                keep_levels = vc[vc >= min_level_n].index
                s_nonnull = s_nonnull[s_nonnull[col].astype(str).isin(keep_levels)]
                rec["n_kept_after_level_filter"] = int(len(s_nonnull))
                vc2 = s_nonnull[col].astype(str).value_counts().sort_index()
                # refresh counts after pruning
                rec["n_levels"] = int(len(vc2))
                rec["level_counts_json"] = json.dumps(vc2.to_dict())
                if len(vc2) < 2 or len(s_nonnull) < min_n:
                    rec["fail_reason"] = "insufficient levels or total N after pruning"
                    summary_records.append(rec)
                    continue

        else:  # continuous
            s_vals = _series_numeric_coerce(s_nonnull[col])
            if s_vals.nunique() < min_unique_cont or len(s_vals) < min_n:
                rec["fail_reason"] = "insufficient unique values or total N for continuous"
                summary_records.append(rec)
                continue
            mu = s_vals.mean()
            sd = s_vals.std(ddof=0)
            rec["continuous_mean"] = float(mu) if pd.notna(mu) else None
            rec["continuous_sd"] = float(sd) if pd.notna(sd) else None
            rec["continuous_min"] = float(s_vals.min()) if len(s_vals) else None
            rec["continuous_max"] = float(s_vals.max()) if len(s_vals) else None

        # If we got here, it passes type-specific checks
        rec["passes_filters"] = True

        # Write the phenotype file (binary/categorical may be pruned)
        if typ == "categorical":
            # s_nonnull already pruned; write that
            out_df = s_nonnull.rename(columns={'bin_identifier': 'sample', col: 'phenotype'})
        else:
            # binary/continuous: write non-null rows
            out_df = s_nonnull.rename(columns={'bin_identifier': 'sample', col: 'phenotype'})

        p = pathlib.Path(out_dir) / f"{species}__{re.sub(r'[^A-Za-z0-9_.-]+','_',col)}.pheno.tsv"
        out_df.to_csv(p, sep='\t', index=False)
        rec["pheno_tsv"] = str(p)
        rows.append((col, typ, str(p)))

        summary_records.append(rec)

    # Write species summary
    sum_df = pd.DataFrame.from_records(summary_records)
    sum_p = pathlib.Path(out_dir) / f"{species}.pheno_summary.tsv"
    sum_df.to_csv(sum_p, sep="\t", index=False)

    return rows


def filter_metadata_per_species(
        metadata_fp: str, 
        ani_map_fp: str, 
        out_dir: str,
        min_n: int = DEFAULT_MIN_N, 
        max_missing_frac: float = DEFAULT_MAX_MISSING_FRAC,
        min_level_n: int = DEFAULT_MIN_LEVEL_N, 
        min_unique_cont: int = DEFAULT_MIN_UNIQUE_CONT,
        custom_missing_values: Optional[List[str]] = None,
        iqr_factor: float = 3.0,       # Tukey fence multiplier
    ) -> Dict[str, List[Tuple[str, str, str]]]:
    """
    Filter metadata per species and create phenotype files for analysis.
    Also writes <species>.pheno_summary.tsv for each species.
    """
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(metadata_fp, sep='\t', low_memory=False)
    cols_map = {c.lower(): c for c in meta.columns}

    try:
        sample_col = _pick_col(cols_map, ['sampleid', 'sample', 'patient', 'id'])
    except KeyError:
        raise RuntimeError("metadata must contain 'SampleID' or similar column")

    meta = meta.drop_duplicates(subset=[sample_col])

    for col in meta.columns:
        if col != sample_col:
            meta[col] = _clean_metadata_values(meta[col], custom_missing_values)

    ani = pd.read_csv(ani_map_fp, sep='\t', names=['species','bin_identifier'])
    ani['patient'] = ani['bin_identifier'].apply(_extract_patient_from_bin)
    df = ani.merge(meta, left_on='patient', right_on=sample_col, how='inner')

    out_index: Dict[str, List[Tuple[str, str, str]]] = {}
    for species, sub in df.groupby('species', sort=False):
        rows = _process_species_phenotypes(
            sub=sub,
            species=species,
            sample_col=sample_col,
            out_dir=out_dir,
            max_missing_frac=max_missing_frac,
            min_n=min_n,
            min_level_n=min_level_n,
            min_unique_cont=min_unique_cont,
            iqr_factor=iqr_factor,
        )
        out_index[species] = rows

        # manifest (unchanged)
        idxp = pathlib.Path(out_dir) / f"{species}.list.tsv"
        with open(idxp, 'w') as fh:
            fh.write('species\tvariable\ttype\tpheno_tsv\n')
            for (v, t, pth) in rows:
                fh.write(f"{species}\t{v}\t{t}\t{pth}\n")

    return out_index

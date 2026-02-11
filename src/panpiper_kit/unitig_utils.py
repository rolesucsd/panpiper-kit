"""
Shared utilities for unitig processing and annotation.

This module provides common functions used by unitig annotation scripts
to avoid code duplication.
"""

import gzip
from typing import Dict, Set, List
from collections import defaultdict

import numpy as np

from .fdr import compute_bh_qvalues


def open_maybe_gz(path: str):
    """
    Open file for reading, handling both regular and gzipped files.

    Args:
        path: File path to open

    Returns:
        File handle for reading
    """
    return gzip.open(path, "rt") if str(path).endswith(".gz") else open(path, "r")


def parse_unitig_map(path: str) -> Dict[str, Set[str]]:
    """
    Parse unitig-to-samples mapping file.

    Expected format (no header):
        UNITIG | sampleA:1 sampleB:1 ...

    Args:
        path: Path to unitig mapping file

    Returns:
        Dictionary mapping unitig sequences to sets of sample names

    Example:
        >>> unitig_map = parse_unitig_map("unitigs.txt")
        >>> unitig_map["ACGTACGT"]
        {'sample1', 'sample2'}
    """
    unitig_to_samples = defaultdict(set)
    with open_maybe_gz(path) as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln or ln.startswith("#") or " | " not in ln:
                continue
            unitig, rhs = ln.split(" | ", 1)
            unitig = unitig.strip()
            if not unitig:
                continue
            for tok in rhs.strip().split():
                s = tok.split(":", 1)[0].strip()
                if s:
                    unitig_to_samples[unitig].add(s)
    return unitig_to_samples


def bh_fdr(pvalues: List[float]) -> List[float]:
    """
    Compute Benjamini-Hochberg q-values from list of p-values.

    This is a compatibility wrapper around the canonical compute_bh_qvalues function.
    Handles invalid p-values by converting them to 1.0.

    Args:
        pvalues: List or array of p-values

    Returns:
        List of q-values (same length as input)

    Example:
        >>> p_vals = [0.01, 0.04, 0.03, 0.05]
        >>> q_vals = bh_fdr(p_vals)
        >>> # q_vals contains FDR-corrected values
    """
    # Convert to numpy array, handling invalid values
    p_array = []
    for p in pvalues:
        try:
            pv = float(p)
            # Check for valid p-value range and NaN
            if pv < 0 or pv > 1 or not (pv == pv):  # NaN check
                pv = 1.0
        except (ValueError, TypeError):
            pv = 1.0
        p_array.append(pv)

    # Use canonical implementation
    q_array = compute_bh_qvalues(np.array(p_array))
    return q_array.tolist()

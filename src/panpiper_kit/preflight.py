"""
Preflight identifier validation.

Three files must agree on sample identity before any analysis is meaningful:

1. ``--genomes``   FASTA basenames, which become the mash matrix index and the
                   unitig-caller sample names.
2. ``--ani-map``   column 2, which becomes the ``sample`` column of every
                   phenotype file.
3. ``--metadata``  the SampleID column, joined via the patient ID parsed out of
                   the bin identifier.

If any adjacent pair fails to overlap, every downstream test silently reports
``n_samples=0`` and pyseer finds no samples -- results that read as "no signal"
rather than as a failure. This module checks the joins up front and aborts with
a message naming the offending IDs.
"""

import logging
from typing import Dict, List, Optional, Sequence

import pandas as pd

from .filter import _extract_patient_from_bin, _pick_col

logger = logging.getLogger(__name__)

# Fraction of IDs that must join before a partial mismatch is escalated to an error.
DEFAULT_MIN_OVERLAP_FRAC = 0.0


class PreflightError(RuntimeError):
    """Raised when the input files do not share a common identifier namespace."""


def _fmt(ids: Sequence[str], n: int = 3) -> str:
    """Format a few example IDs for an error message."""
    return ', '.join(repr(str(i)) for i in list(ids)[:n]) or '(none)'


def _check_join(
    name: str,
    left_label: str,
    left_ids: Sequence[str],
    right_label: str,
    right_ids: Sequence[str],
    remedy: str,
    min_overlap_frac: float,
    errors: List[str],
) -> int:
    """
    Check that two ID sets overlap and record a message if they do not.

    Args:
        name: Short name of the join being checked
        left_label: Human-readable source of left_ids
        left_ids: Identifiers from the left source
        right_label: Human-readable source of right_ids
        right_ids: Identifiers from the right source
        remedy: Instruction telling the user how to fix a mismatch
        min_overlap_frac: Minimum fraction of left_ids that must join
        errors: List that failure messages are appended to

    Returns:
        Number of overlapping identifiers
    """
    left = set(map(str, left_ids))
    right = set(map(str, right_ids))
    shared = left & right
    frac = len(shared) / len(left) if left else 0.0

    logger.info(
        f"  {name}: {len(shared)}/{len(left)} {left_label} IDs found in {right_label} "
        f"({frac:.1%})"
    )

    if not shared:
        errors.append(
            f"{name}: 0 of {len(left)} {left_label} IDs match any of the "
            f"{len(right)} {right_label} IDs.\n"
            f"    {left_label} examples: {_fmt(sorted(left))}\n"
            f"    {right_label} examples: {_fmt(sorted(right))}\n"
            f"    {remedy}"
        )
    elif frac < min_overlap_frac:
        missing = sorted(left - shared)
        errors.append(
            f"{name}: only {frac:.1%} of {left_label} IDs match {right_label} "
            f"(required: {min_overlap_frac:.1%}).\n"
            f"    Unmatched examples: {_fmt(missing, 5)}\n"
            f"    {remedy}"
        )
    elif len(shared) < len(left):
        missing = sorted(left - shared)
        logger.warning(
            f"  {name}: {len(missing)} {left_label} IDs have no match in {right_label}; "
            f"they will be dropped. Examples: {_fmt(missing, 5)}"
        )

    return len(shared)


def validate_identifiers(
    sample_to_path: Dict[str, str],
    ani: pd.DataFrame,
    metadata_fp: str,
    min_overlap_frac: float = DEFAULT_MIN_OVERLAP_FRAC,
) -> Dict[str, int]:
    """
    Verify that genomes, ANI map and metadata share a common ID namespace.

    Args:
        sample_to_path: Mapping of FASTA basename to file path
        ani: ANI mapping DataFrame with 'species' and 'sample' columns
        metadata_fp: Path to the metadata TSV
        min_overlap_frac: Minimum fraction of IDs that must join at each step

    Returns:
        Dictionary of overlap counts per checked join

    Raises:
        PreflightError: If any join is empty or below min_overlap_frac
    """
    logger.info("Preflight: validating sample identifiers across inputs")
    errors: List[str] = []

    ani_ids = ani['sample'].astype(str).unique()
    fasta_ids = list(sample_to_path.keys())

    n_genomes = _check_join(
        name="ani_map -> genomes",
        left_label="ANI map bin_identifier",
        left_ids=ani_ids,
        right_label="FASTA basename",
        right_ids=fasta_ids,
        remedy=(
            "Column 2 of --ani-map must be the FASTA basename without its extension "
            "(see --ani-map help). Rename the ANI map entries or point --genomes at "
            "the directory whose filenames match."
        ),
        min_overlap_frac=min_overlap_frac,
        errors=errors,
    )

    patients = sorted({_extract_patient_from_bin(b) for b in ani_ids})
    degenerate = [b for b in ani_ids if _extract_patient_from_bin(b) == b]
    if degenerate:
        logger.warning(
            f"  {len(degenerate)} bin identifiers yielded no patient prefix "
            f"(too few '_'-separated fields). Examples: {_fmt(degenerate, 5)}"
        )

    meta = pd.read_csv(metadata_fp, sep='\t', low_memory=False, nrows=None)
    cols_map = {c.lower(): c for c in meta.columns}
    try:
        sample_col = _pick_col(cols_map, ['sampleid', 'sample', 'patient', 'id'])
    except KeyError:
        raise PreflightError(
            f"Metadata {metadata_fp} has no SampleID-like column. "
            f"Found columns: {list(meta.columns)}"
        )

    n_meta = _check_join(
        name="ani_map -> metadata",
        left_label="patient ID parsed from bin_identifier",
        left_ids=patients,
        right_label=f"metadata '{sample_col}'",
        right_ids=meta[sample_col].astype(str).unique(),
        remedy=(
            "Patient IDs are parsed as everything before the final two "
            "'_'-separated fields of the bin identifier (a trailing '_sub' is "
            "stripped first). Make the metadata SampleID column use those same IDs."
        ),
        min_overlap_frac=min_overlap_frac,
        errors=errors,
    )

    if errors:
        raise PreflightError(
            "Input files do not share a common sample-ID namespace:\n\n"
            + "\n\n".join(f"  [{i}] {e}" for i, e in enumerate(errors, 1))
            + "\n\nNo analysis was run. Fix the identifiers above, or pass "
              "--skip-preflight to proceed anyway (results will be empty)."
        )

    logger.info("Preflight: identifier checks passed")
    return {"ani_to_genomes": n_genomes, "ani_to_metadata": n_meta}

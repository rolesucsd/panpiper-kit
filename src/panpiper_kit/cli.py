import argparse
import logging
import pathlib
import pandas as pd
import os
import sys
import subprocess
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, NamedTuple
from typing_extensions import Literal

from .files import list_fastas, ensure_dir, run, safe_sample_name
from .filter import filter_metadata_per_species, filter_by_checkm
from .mash import mash_within_species
from .assoc import run_assoc, PhenotypeJob, _init_worker as _assoc_init_worker, _permutation_test as _assoc_permutation_test
from .gwas import ensure_unitigs
from .fdr import add_bh
from .summarize_pyseer import summarize_pyseer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ----- Constants -----

# Default filtering thresholds
DEFAULT_MIN_SAMPLES = 30
DEFAULT_MAX_MISSING_FRAC = 0.2
DEFAULT_MIN_LEVEL_N = 5
DEFAULT_MIN_UNIQUE_CONT = 3
VALID_PHENOTYPE_TYPES = {'binary', 'continuous', 'categorical'}

# Pyseer constants
PYSEER_MIN_SAMPLES_PER_GROUP = 2  # Minimum samples required per group for Pyseer analysis
PYSEER_MIN_SAMPLES_TOTAL = 2  # Minimum total samples for any Pyseer analysis

# Association testing constants
FAST_P_THRESHOLD = 0.10  # p-value threshold for fast mode screening
FAST_MODE_PERMS = 199  # Number of permutations used in fast test mode
EARLY_STOP_P = 0.20  # p-value above which to stop early in permutation testing
ESCALATE_P1 = 0.10  # First p-value threshold for escalating permutations
ESCALATE_P2 = 0.05  # Second p-value threshold for escalating permutations
ESCALATE_P3 = 0.01  # Third p-value threshold for escalating permutations

# Significance threshold
SIGNIFICANCE_ALPHA = 0.05  # Alpha level for statistical significance


# ----- Type Definitions -----

class PhenotypeInfo(NamedTuple):
    """Information about a phenotype variable."""
    variable: str
    phenotype_type: str
    pheno_tsv_path: str


PhenotypeDict = Dict[str, List[PhenotypeInfo]]


@dataclass
class AnalysisConfig:
    """Configuration for species-level analysis."""
    args: argparse.Namespace
    phenos: PhenotypeDict
    mash_dir: pathlib.Path
    pyseer_dir: pathlib.Path
    distance_assoc_dir: pathlib.Path
    unitig_dir: pathlib.Path
    phenos_dir: pathlib.Path
    s2p: Dict[str, str]


# ----- Helper Functions -----

def _generate_expected_pyseer_runs(phenos: PhenotypeDict,
                                  pyseer_dir: pathlib.Path) -> List[Dict[str, str]]:
    """
    Generate a list of all expected Pyseer runs based on phenotype data.

    Args:
        phenos: Dictionary of phenotype information per species
        pyseer_dir: Directory for pyseer GWAS results

    Returns:
        List of dictionaries containing expected run information
    """
    expected_runs = []

    for species, phenotypes in phenos.items():
        for pheno_info in phenotypes:
            var, typ, pheno_tsv = pheno_info
            if typ in ('binary', 'continuous'):
                # Main phenotype Pyseer run
                expected_runs.append({
                    'species': species,
                    'variable': var,
                    'type': typ,
                    'pheno_tsv': pheno_tsv,
                    'output_file': str(pyseer_dir / species / f'{species}__{var}.pyseer.tsv'),
                    'fdr_file': str(pyseer_dir / species / f'{species}__{var}.pyseer.fdr.tsv'),
                    'run_type': 'main_phenotype'
                })

    return expected_runs


def _check_pyseer_preconditions(pheno_tsv: str,
                                min_samples: int = PYSEER_MIN_SAMPLES_TOTAL) -> Tuple[bool, str]:
    """
    Check if a phenotype file meets basic preconditions for Pyseer analysis.

    Args:
        pheno_tsv: Path to phenotype TSV file
        min_samples: Minimum number of samples required

    Returns:
        Tuple of (can_run, reason_if_not)
    """
    try:
        df = pd.read_csv(pheno_tsv, sep='\t')

        if df.empty:
            return False, "Phenotype file is empty"

        if 'sample' not in df.columns or 'phenotype' not in df.columns:
            return False, "Missing required columns (sample, phenotype)"

        if len(df) < min_samples:
            return False, f"Insufficient samples ({len(df)} < {min_samples})"

        # Check for valid phenotype values
        valid_phenos = df['phenotype'].dropna()
        if len(valid_phenos) < min_samples:
            return False, f"Insufficient valid phenotype values ({len(valid_phenos)} < {min_samples})"

        # For binary phenotypes, check both levels have sufficient samples
        if len(valid_phenos.unique()) == 2:
            value_counts = valid_phenos.value_counts()
            min_count = value_counts.min()
            if min_count < PYSEER_MIN_SAMPLES_PER_GROUP:
                return False, f"Insufficient samples per group (min: {min_count})"

        return True, ""

    except (OSError, IOError) as e:
        return False, f"Error reading phenotype file: {e}"
    except pd.errors.ParserError as e:
        return False, f"Invalid TSV format: {e}"
    except Exception as e:
        logger.error(f"Unexpected error checking preconditions for {pheno_tsv}: {e}", exc_info=True)
        return False, f"Unexpected error: {e}"


def _validate_pyseer_inputs(pheno_tsv: str, unitig_file: str) -> Tuple[bool, List[str]]:
    """
    Validate that input files exist and are readable for Pyseer.

    Args:
        pheno_tsv: Path to phenotype TSV file
        unitig_file: Path to unitig file

    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []

    # Check phenotype file
    if not os.path.exists(pheno_tsv):
        errors.append(f"Phenotype file does not exist: {pheno_tsv}")
    else:
        try:
            df = pd.read_csv(pheno_tsv, sep='\t')
            if df.empty:
                errors.append(f"Phenotype file is empty: {pheno_tsv}")
            elif 'sample' not in df.columns or 'phenotype' not in df.columns:
                errors.append(f"Phenotype file missing required columns (sample, phenotype): {pheno_tsv}")
            elif len(df) < PYSEER_MIN_SAMPLES_TOTAL:
                errors.append(f"Phenotype file has insufficient samples (<{PYSEER_MIN_SAMPLES_TOTAL}): {pheno_tsv}")
        except (OSError, IOError) as e:
            errors.append(f"Cannot read phenotype file: {pheno_tsv} - {e}")
        except pd.errors.ParserError as e:
            errors.append(f"Invalid TSV format in {pheno_tsv}: {e}")

    # Check unitig file
    if not os.path.exists(unitig_file):
        errors.append(f"Unitig file does not exist: {unitig_file}")
    else:
        try:
            if os.path.getsize(unitig_file) == 0:
                errors.append(f"Unitig file is empty: {unitig_file}")
        except (OSError, IOError) as e:
            errors.append(f"Unitig file is not accessible: {unitig_file} - {e}")

    return len(errors) == 0, errors


def _track_pyseer_run(species: str, variable: str, run_type: str,
                     pheno_tsv: str, unitig_file: str, output_file: str,
                     status: str, error_msg: str = None) -> Dict[str, str]:
    """
    Track a Pyseer run with its status and details.

    Args:
        species: Species name
        variable: Variable name
        run_type: Type of run (main_phenotype, pairwise, etc.)
        pheno_tsv: Input phenotype file
        unitig_file: Input unitig file
        output_file: Expected output file
        status: Run status (expected, started, completed, failed, skipped)
        error_msg: Error message if failed

    Returns:
        Dictionary with run tracking information
    """
    return {
        'species': species,
        'variable': variable,
        'run_type': run_type,
        'pheno_tsv': pheno_tsv,
        'unitig_file': unitig_file,
        'output_file': output_file,
        'status': status,
        'error_msg': error_msg or '',
        'timestamp': pd.Timestamp.now().isoformat()
    }


def _generate_analysis_summary(phenos: PhenotypeDict,
                              sp_to_samples: Dict[str, List[str]],
                              remaining_species: Dict[str, List[str]],
                              args: argparse.Namespace,
                              pyseer_dir: pathlib.Path,
                              distance_assoc_dir: pathlib.Path) -> None:
    """
    Generate a comprehensive summary of all input files and their analysis status.

    Args:
        phenos: Dictionary of phenotype information per species
        sp_to_samples: Dictionary mapping species to sample lists
        remaining_species: Dictionary of species that need processing
        args: Command line arguments
        pyseer_dir: Directory for pyseer GWAS results
        distance_assoc_dir: Directory for distance-based association results
    """
    logger.info("=" * 80)
    logger.info("COMPREHENSIVE ANALYSIS SUMMARY")
    logger.info("=" * 80)

    # Summary statistics
    total_species = len(sp_to_samples)
    remaining_count = len(remaining_species)
    completed_count = total_species - remaining_count

    logger.info(f"Total species: {total_species}")
    logger.info(f"Already completed: {completed_count}")
    logger.info(f"Will process: {remaining_count}")
    logger.info("")

    # Analyze each species and phenotype
    for species in sorted(sp_to_samples.keys()):
        species_phenos = phenos.get(species, [])
        n_samples = len(sp_to_samples[species])
        is_remaining = species in remaining_species

        logger.info(f"SPECIES: {species} (N={n_samples}) {'[REMAINING]' if is_remaining else '[COMPLETED]'}")
        logger.info("-" * 60)

        if not species_phenos:
            logger.info("  No phenotypes found for this species")
            continue

        for pheno_info in species_phenos:
            var, typ, pheno_tsv = pheno_info
            # Check if phenotype file exists and analyze it
            file_exists = os.path.exists(pheno_tsv)
            if not file_exists:
                logger.info(f"  {var} ({typ}): MISSING FILE - {pheno_tsv}")
                continue

            try:
                df = pd.read_csv(pheno_tsv, sep='\t')
                n_total = len(df)
                n_valid = len(df.dropna(subset=['phenotype']))
                valid_phenos = df['phenotype'].dropna()

                # Analyze phenotype distribution
                if typ == 'binary':
                    if len(valid_phenos.unique()) == 2:
                        value_counts = valid_phenos.value_counts()
                        min_group = value_counts.min()
                        logger.info(f"  {var} ({typ}): N={n_valid} | Groups: {dict(value_counts)} | Min group: {min_group}")

                        # Check Pyseer requirements
                        if min_group < PYSEER_MIN_SAMPLES_PER_GROUP:
                            logger.info(f"    → SKIP PYSEER: Insufficient samples per group (min: {min_group})")
                        else:
                            logger.info(f"    → RUN PYSEER: Sufficient samples per group")
                    else:
                        logger.info(f"  {var} ({typ}): N={n_valid} | NOT BINARY - {len(valid_phenos.unique())} unique values")

                elif typ == 'continuous':
                    nunique = valid_phenos.nunique()
                    logger.info(f"  {var} ({typ}): N={n_valid} | Unique values: {nunique}")

                    if nunique < args.min_unique_cont:
                        logger.info(f"    → SKIP PYSEER: Insufficient unique values ({nunique} < {args.min_unique_cont})")
                    else:
                        logger.info(f"    → RUN PYSEER: Sufficient unique values")

                elif typ == 'categorical':
                    value_counts = valid_phenos.value_counts()
                    n_groups = len(value_counts)
                    min_group = value_counts.min()
                    logger.info(f"  {var} ({typ}): N={n_valid} | Groups: {n_groups} | Min group: {min_group}")
                    logger.info(f"    → DISTANCE TEST: Will run PERMANOVA")

                    # Check pairwise requirements
                    if n_groups >= 2 and min_group >= args.pair_min_n:
                        logger.info(f"    → PAIRWISE TESTS: Will run (min group: {min_group} >= {args.pair_min_n})")

                        # Check pairwise Pyseer requirements
                        large_groups = value_counts[value_counts >= args.pair_pyseer_min_n]
                        if len(large_groups) >= 2:
                            logger.info(f"    → PAIRWISE PYSEER: Will run for {len(large_groups)} groups (>= {args.pair_pyseer_min_n})")
                        else:
                            logger.info(f"    → SKIP PAIRWISE PYSEER: Only {len(large_groups)} groups >= {args.pair_pyseer_min_n}")
                    else:
                        logger.info(f"    → SKIP PAIRWISE: Insufficient groups or samples (groups: {n_groups}, min: {min_group})")

            except (OSError, IOError) as e:
                logger.warning(f"  {var} ({typ}): ERROR reading file - {e}")
            except pd.errors.ParserError as e:
                logger.warning(f"  {var} ({typ}): ERROR invalid TSV format - {e}")

        logger.info("")

    # Summary of parameters
    logger.info("ANALYSIS PARAMETERS:")
    logger.info("-" * 30)
    logger.info(f"Min samples per species: {args.min_n}")
    logger.info(f"Min samples per group (pairwise): {args.pair_min_n}")
    logger.info(f"Min samples per group (pairwise Pyseer): {args.pair_pyseer_min_n}")
    logger.info(f"Min unique values (continuous): {args.min_unique_cont}")
    logger.info(f"Max missing fraction: {args.max_missing_frac}")
    logger.info("=" * 80)


def _write_pyseer_tracking_report(tracking_data: List[Dict[str, str]],
                                 output_file: pathlib.Path) -> None:
    """
    Write a comprehensive Pyseer tracking report.

    Args:
        tracking_data: List of tracking dictionaries
        output_file: Path to write the report
    """
    if not tracking_data:
        logger.warning("No Pyseer tracking data to write")
        return

    df = pd.DataFrame(tracking_data)

    # Add summary statistics
    summary = {
        'total_expected': len(df),
        'completed': len(df[df['status'] == 'completed']),
        'failed': len(df[df['status'] == 'failed']),
        'skipped': len(df[df['status'] == 'skipped']),
        'started': len(df[df['status'] == 'started']),
        'expected': len(df[df['status'] == 'expected'])
    }

    # Write detailed report
    df.to_csv(output_file, sep='\t', index=False)

    # Write summary
    summary_file = output_file.parent / f"{output_file.stem}_summary.tsv"
    pd.DataFrame([summary]).to_csv(summary_file, sep='\t', index=False)

    logger.info(f"Pyseer tracking report written to: {output_file}")
    logger.info(f"Pyseer summary: {summary}")

    # Log failed runs
    failed_runs = df[df['status'] == 'failed']
    if not failed_runs.empty:
        logger.warning(f"{len(failed_runs)} Pyseer runs failed:")
        for _, run in failed_runs.iterrows():
            logger.warning(f"  {run['species']}__{run['variable']}: {run['error_msg']}")


def _build_species_sample_map(ani: pd.DataFrame, s2p: Dict[str, str], min_n: int) -> Dict[str, List[str]]:
    """
    Build mapping from species to sample lists, filtering by minimum sample count.

    Args:
        ani: ANI mapping DataFrame with 'species' and 'sample' columns
        s2p: Dictionary mapping sample names to file paths
        min_n: Minimum number of samples required per species

    Returns:
        Dictionary mapping species names to lists of sample names
    """
    return {sp: [s for s in ani.loc[ani['species']==sp, 'sample'] if s in s2p]
            for sp in ani['species'].unique()
            if len([s for s in ani.loc[ani['species']==sp, 'sample'] if s in s2p]) >= min_n}


def _get_remaining_species(sp_to_samples: Dict[str, List[str]], args: argparse.Namespace,
                          phenos: PhenotypeDict,
                          mash_dir: pathlib.Path, pyseer_dir: pathlib.Path,
                          distance_assoc_dir: pathlib.Path, unitig_dir: pathlib.Path) -> Dict[str, List[str]]:
    """
    Determine which species need to be processed based on file existence.

    Args:
        sp_to_samples: Dictionary mapping species to sample lists
        args: Command line arguments
        phenos: Dictionary of phenotype information per species
        mash_dir: Directory for Mash output files
        pyseer_dir: Directory for pyseer GWAS results
        distance_assoc_dir: Directory for distance-based association results
        unitig_dir: Directory for unitig files

    Returns:
        Dictionary of species that need to be processed
    """
    if args.resume:
        # Check which species are actually complete (file-based verification)
        actually_complete = set()
        for sp in sp_to_samples.keys():
            if is_species_complete(sp, phenos, mash_dir, pyseer_dir, distance_assoc_dir, unitig_dir):
                actually_complete.add(sp)

        logger.info(f"Found {len(actually_complete)} previously completed species based on file existence")

        # Filter out completed species
        remaining = {sp: sams for sp, sams in sp_to_samples.items()
                    if sp not in actually_complete}
        logger.info(f"Will process {len(remaining)} remaining species")
        return remaining
    else:
        logger.info(f"Processing all {len(sp_to_samples)} species")
        return sp_to_samples


def _collect_existing_results(actually_complete: set, phenos: PhenotypeDict,
                             distance_assoc_dir: pathlib.Path) -> List[pd.DataFrame]:
    """
    Collect existing results from previously completed species.

    Args:
        actually_complete: Set of species that are actually complete
        phenos: Dictionary of phenotype information per species
        distance_assoc_dir: Directory for distance-based association results

    Returns:
        List of DataFrames containing existing results
    """
    results = []
    for sp in actually_complete:
        # Load the combined distance association file for this species
        combined_file = distance_assoc_dir / sp / f'{sp}.combined.dist_assoc.tsv'
        if combined_file.exists():
            try:
                df = pd.read_csv(combined_file, sep='\t')
                results.append(df)
            except (OSError, IOError, pd.errors.ParserError) as e:
                logger.warning(f"Could not load existing combined results for {sp}: {e}")
    return results


# ----- Progress Tracking -----

def is_species_complete(
    species: str,
    phenos: PhenotypeDict,
    mash_dir: pathlib.Path,
    pyseer_dir: pathlib.Path,
    distance_assoc_dir: pathlib.Path,
    unitig_dir: pathlib.Path
) -> bool:
    """
    Check if a species has been fully processed by verifying all required output files exist.

    Args:
        species: Species name to check
        phenos: Dictionary of phenotype information per species
        mash_dir: Directory for Mash output files
        pyseer_dir: Directory for pyseer GWAS results
        distance_assoc_dir: Directory for distance-based association results
        unitig_dir: Directory for unitig files

    Returns:
        True if species is fully processed, False otherwise
    """
    # Check mash file
    mash_file = mash_dir / species / 'mash.tsv'
    if not mash_file.exists():
        return False

    # Check unitig file
    unitig_file = unitig_dir / species / 'uc.pyseer'
    if not unitig_file.exists():
        return False

    # Check distance association combined file
    dist_combined_file = distance_assoc_dir / species / f'{species}.combined.dist_assoc.tsv'
    if not dist_combined_file.exists():
        return False

    # Check association files for all phenotypes
    for pheno_info in phenos.get(species, []):
        var, typ, _ = pheno_info
        # Check raw Pyseer files for binary/continuous phenotypes
        if typ in ('binary', 'continuous'):
            raw_pyseer_file = pyseer_dir / species / f'{species}__{var}.pyseer.tsv'
            if not raw_pyseer_file.exists():
                return False

            # Check FDR-corrected Pyseer files
            gwas_file = pyseer_dir / species / f'{species}__{var}.pyseer.fdr.tsv'
            if not gwas_file.exists():
                return False

    # Check pairwise files if any categorical phenotypes exist
    has_categorical = any(typ == 'categorical' for _, typ, _ in phenos.get(species, []))
    if has_categorical:
        # Check combined pairwise association file
        pairwise_assoc_file = distance_assoc_dir / species / f'{species}.pairwise_permanova.tsv'
        if not pairwise_assoc_file.exists():
            return False

        # Check combined pairwise Pyseer file
        pairwise_pyseer_file = pyseer_dir / species / f'{species}.pairwise.pyseer.tsv'
        if not pairwise_pyseer_file.exists():
            return False

        # Check pairwise directory exists
        pairwise_dir = pyseer_dir / species / 'pairwise'
        if not pairwise_dir.exists():
            return False

        # Check for individual pairwise files (at least one should exist)
        pairwise_files = list(pairwise_dir.glob(f'{species}__*.pyseer.tsv'))
        if not pairwise_files:
            return False

    return True


def are_phenotype_files_complete(species: str, phenos_dir: pathlib.Path) -> bool:
    """
    Check if all phenotype files for a species exist based on the list.tsv manifest.

    Args:
        species: Species name to check
        phenos_dir: Directory containing phenotype files and list.tsv manifests

    Returns:
        True if all phenotype files exist as specified in the manifest, False otherwise
    """
    list_file = phenos_dir / f"{species}.list.tsv"
    if not list_file.exists():
        return False

    try:
        # Read the manifest
        df = pd.read_csv(list_file, sep='\t')

        # Check each phenotype file exists
        for _, row in df.iterrows():
            pheno_file = pathlib.Path(row['pheno_tsv'])
            if not pheno_file.exists():
                return False

        return True
    except (OSError, IOError, pd.errors.ParserError, KeyError):
        # If we can't read the manifest or it's malformed, assume incomplete
        return False


def load_phenotype_manifest(phenos_dir: pathlib.Path) -> PhenotypeDict:
    """
    Load phenotype information from existing list.tsv manifest files.

    Args:
        phenos_dir: Directory containing phenotype files and list.tsv manifests

    Returns:
        Dictionary mapping species names to lists of PhenotypeInfo tuples
    """
    phenos = {}

    logger.info(f"Loading phenotype manifests from: {phenos_dir}")

    # Find all list.tsv files
    list_files = list(phenos_dir.glob("*.list.tsv"))
    logger.info(f"Found {len(list_files)} list.tsv files")

    for list_file in list_files:
        # Remove .list.tsv extension properly
        species = list_file.name.replace('.list.tsv', '')
        try:
            df = pd.read_csv(list_file, sep='\t')
            species_phenos = []

            for _, row in df.iterrows():
                # Validate phenotype type
                pheno_type = row['type']
                if pheno_type not in VALID_PHENOTYPE_TYPES:
                    logger.warning(
                        f"Invalid phenotype type '{pheno_type}' for {row['variable']} in {species}, skipping. "
                        f"Valid types: {', '.join(sorted(VALID_PHENOTYPE_TYPES))}"
                    )
                    continue
                species_phenos.append(PhenotypeInfo(row['variable'], pheno_type, row['pheno_tsv']))

            phenos[species] = species_phenos
            logger.debug(f"Loaded {len(species_phenos)} phenotypes for {species}")
        except (OSError, IOError) as e:
            logger.warning(f"Could not read phenotype manifest for {species}: {e}")
        except pd.errors.ParserError as e:
            logger.warning(f"Invalid TSV format in phenotype manifest for {species}: {e}")
        except KeyError as e:
            logger.warning(f"Missing required column in phenotype manifest for {species}: {e}")

    logger.info(f"Total species with phenotypes: {len(phenos)}")
    return phenos


def has_passing_phenotypes(species: str, phenos_dir: pathlib.Path) -> bool:
    """
    Check if a species has any phenotypes that passed filters by reading the phenotype summary file.

    Args:
        species: Species name to check
        phenos_dir: Directory containing phenotype summary files

    Returns:
        True if at least one phenotype passed filters, False otherwise
    """
    summary_file = phenos_dir / f"{species}.pheno_summary.tsv"
    if not summary_file.exists():
        return False

    try:
        df = pd.read_csv(summary_file, sep='\t')
        if df.empty:
            return False
        # Check if any phenotype has passes_filters == True
        if 'passes_filters' in df.columns:
            # Convert to boolean if it's stored as string (e.g., "True"/"False" or "1"/"0")
            passes_col = df['passes_filters']
            # Handle various formats: boolean, string "True"/"False", int 1/0
            if passes_col.dtype == 'object':
                # Try to convert string representations to boolean
                # Handle "True", "False", "1", "0", etc.
                passes_col = passes_col.astype(str).str.lower().isin(['true', '1', 'yes'])
            elif passes_col.dtype in ['int64', 'int32', 'float64', 'float32']:
                # If numeric, convert 1 to True, 0 to False
                passes_col = passes_col.astype(bool)
            # Now check if any are True (passes_col should already be boolean)
            has_passing = bool(passes_col.any())
            logger.debug(f"Species {species}: has_passing_phenotypes={has_passing} (from {len(df)} phenotype rows)")
            return has_passing
        # If column doesn't exist, assume no phenotypes passed (old format)
        return False
    except (OSError, IOError) as e:
        logger.warning(f"Could not read phenotype summary for {species}: {e}")
        return False
    except pd.errors.ParserError as e:
        logger.warning(f"Invalid TSV format in phenotype summary for {species}: {e}")
        return False


# ----- Worker Helper Functions -----

def _run_mash_for_species(species: str, samples: List[str], config: AnalysisConfig) -> str:
    """
    Run Mash distance calculation for a species.

    Args:
        species: Species name
        samples: List of sample names
        config: Analysis configuration

    Returns:
        Path to Mash distance matrix TSV
    """
    threads_per_worker = max(1, config.args.threads_per_worker)
    paths = [config.s2p[s] for s in samples if s in config.s2p]
    species_output_dir = config.mash_dir / species
    species_output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Processing {len(paths)} FASTA files for species {species}")
    mash_tsv = mash_within_species(
        paths, str(species_output_dir),
        k=config.args.mash_k, s=config.args.mash_s, threads=threads_per_worker
    )

    return mash_tsv


def _run_unitigs_for_species(species: str, samples: List[str], config: AnalysisConfig) -> str:
    """
    Run unitig calling for a species.

    Args:
        species: Species name
        samples: List of sample names
        config: Analysis configuration

    Returns:
        Path to unitig pyseer file
    """
    threads_per_worker = max(1, config.args.threads_per_worker)
    paths = [config.s2p[s] for s in samples if s in config.s2p]
    species_output_dir = config.mash_dir / species

    # Create refs file
    ref_txt = species_output_dir / 'refs.txt'
    if not ref_txt.exists():
        with open(ref_txt, 'w') as fh:
            fh.write('\n'.join(paths))

    uc_pyseer = ensure_unitigs(
        str(ref_txt), str(config.unitig_dir / species),
        kmer=config.args.kmer, threads=threads_per_worker
    )
    return uc_pyseer


def _run_distance_tests(species: str, config: AnalysisConfig, mash_tsv: str) -> Tuple[List[pd.DataFrame], pd.DataFrame]:
    """
    Run distance-based association tests for all phenotypes of a species.

    Args:
        species: Species name
        config: Analysis configuration
        mash_tsv: Path to Mash distance matrix

    Returns:
        Tuple of (list of result DataFrames, full results DataFrame)
    """
    phenotypes = config.phenos.get(species, [])
    if not phenotypes:
        logger.warning(f"No phenotypes found for species {species}")
        return [], pd.DataFrame()

    # Create PhenotypeJob objects
    phenotype_jobs = []
    for pheno_info in phenotypes:
        var, typ, pheno_tsv = pheno_info
        if not os.path.exists(pheno_tsv):
            logger.error(f"Phenotype file does not exist: {pheno_tsv}")
            continue
        phenotype_jobs.append(PhenotypeJob(species=species, variable=var, typ=typ, pheno_tsv=pheno_tsv))

    if not phenotype_jobs:
        return [], pd.DataFrame()

    logger.info(f"Running association tests for {len(phenotype_jobs)} phenotypes")

    # Build permutation ladder
    if config.args.tests == 'exact':
        perm_ladder = []
        current = config.args.perms
        while current <= config.args.max_perms:
            perm_ladder.append(current)
            if current >= config.args.max_perms:
                break
            next_val = min(current * 10, config.args.max_perms)
            if next_val == current:
                break
            current = next_val
        perm_ladder = tuple(perm_ladder)
    else:
        perm_ladder = (FAST_MODE_PERMS,)

    logger.info(f"Using permutation ladder: {perm_ladder}, mode: {config.args.perm_mode}")

    # Run association tests
    try:
        results_df = run_assoc(
            mash_tsv=str(mash_tsv),
            phenos=phenotype_jobs,
            n_workers=1,
            fast_p_thresh=FAST_P_THRESHOLD,
            fast_max_axes=config.args.max_axes,
            perm_ladder=perm_ladder,
            early_stop_p=EARLY_STOP_P,
            escalate_p1=ESCALATE_P1,
            escalate_p2=ESCALATE_P2,
            escalate_p3=ESCALATE_P3,
            perm_mode=config.args.perm_mode
        )

        if not results_df.empty:
            logger.info(f"Association testing completed for {species}: {len(results_df)} results")

            # Create species-specific directory
            species_dist_dir = config.distance_assoc_dir / species
            species_dist_dir.mkdir(parents=True, exist_ok=True)

            # Save combined results file
            combined_file = species_dist_dir / f'{species}.combined.dist_assoc.tsv'
            results_df.to_csv(combined_file, sep='\t', index=False)
            logger.info(f"Distance association results saved to: {combined_file}")

            # Save individual per-variable files as well for easier inspection
            grouped_results = results_df.groupby('metadata')
            for var, var_results in grouped_results:
                var_file = species_dist_dir / f'{species}__{var}.dist_assoc.tsv'
                var_results.to_csv(var_file, sep='\t', index=False)

            # Convert to list of row DataFrames for backward compatibility
            rows = [pd.DataFrame([row.to_dict()]) for _, row in results_df.iterrows()]
            return rows, results_df
        else:
            logger.warning(f"No results from association testing for {species}")
            return [], pd.DataFrame()

    except (OSError, IOError, pd.errors.ParserError) as e:
        logger.error(f"Association testing failed for {species}: {e}", exc_info=True)
        return [], pd.DataFrame()
    except Exception as e:
        logger.error(f"Unexpected error in association testing for {species}: {e}", exc_info=True)
        raise


def _run_single_pairwise_pyseer(
    species: str,
    variable: str,
    comparison_name: str,
    pheno_file: pathlib.Path,
    output_file: pathlib.Path,
    unitig_file: str,
    config: AnalysisConfig
) -> Dict[str, str]:
    """
    Run a single pairwise Pyseer analysis.

    Args:
        species: Species name
        variable: Variable name
        comparison_name: Name of the comparison (e.g., "group1_vs_group2")
        pheno_file: Path to phenotype file
        output_file: Path for output file
        unitig_file: Path to unitig file
        config: Analysis configuration

    Returns:
        Tracking dictionary for this run
    """
    # Check preconditions
    can_run, precondition_reason = _check_pyseer_preconditions(str(pheno_file), min_samples=PYSEER_MIN_SAMPLES_TOTAL)
    if not can_run:
        logger.warning(f"Pairwise Pyseer skipped for {species}__{variable}__{comparison_name}: {precondition_reason}")
        return _track_pyseer_run(
            species, f"{variable}__{comparison_name}", 'pairwise',
            str(pheno_file), unitig_file, str(output_file),
            'skipped', precondition_reason
        )

    # Validate inputs
    is_valid, validation_errors = _validate_pyseer_inputs(str(pheno_file), unitig_file)
    if not is_valid:
        error_msg = "; ".join(validation_errors)
        logger.error(f"Pairwise Pyseer input validation failed for {species}__{variable}__{comparison_name}: {error_msg}")
        return _track_pyseer_run(
            species, f"{variable}__{comparison_name}", 'pairwise',
            str(pheno_file), unitig_file, str(output_file),
            'failed', error_msg
        )

    # Track start
    _track_pyseer_run(
        species, f"{variable}__{comparison_name}", 'pairwise',
        str(pheno_file), unitig_file, str(output_file),
        'started'
    )

    # Run pyseer with stderr logging
    threads_per_worker = max(1, config.args.threads_per_worker)
    stderr_log = output_file.parent / f"{output_file.stem}.stderr.log"

    try:
        with open(output_file, 'w') as stdout_fh, open(stderr_log, 'w') as stderr_fh:
            cmd = [
                'pyseer',
                '--phenotypes', str(pheno_file),
                '--kmers', unitig_file,
                '--uncompressed',
                '--min-af', str(config.args.maf),
                '--cpu', str(threads_per_worker),
                '--no-distances'
            ]
            subprocess.check_call(cmd, stdout=stdout_fh, stderr=stderr_fh)

        if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
            logger.info(f"Pairwise Pyseer completed successfully for {species}__{variable}__{comparison_name}")
            return _track_pyseer_run(
                species, f"{variable}__{comparison_name}", 'pairwise',
                str(pheno_file), unitig_file, str(output_file),
                'completed'
            )
        else:
            error_msg = "Pyseer output is empty"
            logger.warning(f"Pairwise Pyseer output is empty for {species}__{variable}__{comparison_name}")
            return _track_pyseer_run(
                species, f"{variable}__{comparison_name}", 'pairwise',
                str(pheno_file), unitig_file, str(output_file),
                'failed', error_msg
            )

    except subprocess.CalledProcessError as e:
        error_msg = f"Pyseer failed with return code {e.returncode}"
        if stderr_log.exists():
            error_msg += f" (see {stderr_log} for details)"
        logger.error(f"Pairwise Pyseer failed for {species}__{variable}__{comparison_name}: {error_msg}")
        return _track_pyseer_run(
            species, f"{variable}__{comparison_name}", 'pairwise',
            str(pheno_file), unitig_file, str(output_file),
            'failed', error_msg
        )
    except (OSError, IOError) as e:
        error_msg = f"I/O error: {e}"
        logger.error(f"I/O error running pairwise pyseer for {species}__{variable}__{comparison_name}: {e}")
        return _track_pyseer_run(
            species, f"{variable}__{comparison_name}", 'pairwise',
            str(pheno_file), unitig_file, str(output_file),
            'failed', error_msg
        )
    except Exception as e:
        error_msg = f"Unexpected error: {e}"
        logger.error(f"Unexpected error running pairwise pyseer for {species}__{variable}__{comparison_name}: {e}", exc_info=True)
        return _track_pyseer_run(
            species, f"{variable}__{comparison_name}", 'pairwise',
            str(pheno_file), unitig_file, str(output_file),
            'failed', error_msg
        )


def _create_pairwise_comparison(
    species: str,
    variable: str,
    phenotype_df: pd.DataFrame,
    group1: str,
    group2: str,
    comparison_type: Literal['one_vs_rest', 'group_vs_group'],
    config: AnalysisConfig,
    pairwise_pyseer_dir: pathlib.Path,
    unitig_file: str
) -> Tuple[Dict[str, Any], List[Dict[str, str]], pathlib.Path]:
    """
    Create and analyze a single pairwise comparison.

    Args:
        species: Species name
        variable: Variable name
        phenotype_df: DataFrame with phenotype data
        group1: First group identifier
        group2: Second group identifier ('rest' for one-vs-rest)
        comparison_type: Type of comparison
        config: Analysis configuration
        pairwise_pyseer_dir: Directory for pairwise pyseer outputs
        unitig_file: Path to unitig file

    Returns:
        Tuple of (permanova_result, pyseer_tracking_list, output_file_path or None)
    """
    pyseer_tracking = []

    # Create binary phenotype
    if comparison_type == 'one_vs_rest':
        binary_pheno_df = phenotype_df.copy()
        binary_pheno_df['phenotype'] = (binary_pheno_df['phenotype'] == group1).astype(int)
        comparison_name = f"{group1}_vs_rest"
    else:  # group_vs_group
        binary_pheno_df = phenotype_df[phenotype_df['phenotype'].isin([group1, group2])].copy()
        if binary_pheno_df.empty:
            return None, pyseer_tracking, None
        binary_pheno_df['phenotype'] = (binary_pheno_df['phenotype'] == group1).astype(int)
        comparison_name = f"{group1}_vs_{group2}"

    n_group1 = int((binary_pheno_df['phenotype'] == 1).sum())
    n_group2 = int((binary_pheno_df['phenotype'] == 0).sum())

    # Check minimum sample requirements
    if n_group1 < config.args.pair_min_n or n_group2 < config.args.pair_min_n:
        return None, pyseer_tracking, None

    # Save phenotype file
    pheno_file = pairwise_pyseer_dir / f"{species}__{variable}__{comparison_name}.binary.tsv"
    binary_pheno_df[['sample', 'phenotype']].to_csv(pheno_file, sep='\t', index=False)

    # Run permutation test
    permanova_result = _assoc_permutation_test(
        str(pheno_file), 'binary',
        perms=config.args.perms, mode=config.args.perm_mode
    )
    permanova_result['species'] = species
    permanova_result['metadata'] = variable
    permanova_result['comparison'] = comparison_name
    permanova_result['group1'] = group1
    permanova_result['group2'] = group2 if group2 != 'rest' else 'rest'
    permanova_result['n_group1'] = n_group1
    permanova_result['n_group2'] = n_group2

    # Run Pyseer if significant and sufficient samples
    if pd.notna(permanova_result.get('pvalue')) and permanova_result.get('pvalue') <= SIGNIFICANCE_ALPHA:
        if n_group1 >= config.args.pair_pyseer_min_n and n_group2 >= config.args.pair_pyseer_min_n:
            output_file = pairwise_pyseer_dir / f"{species}__{variable}__{comparison_name}.pyseer.tsv"
            tracking = _run_single_pairwise_pyseer(
                species, variable, comparison_name,
                pheno_file, output_file, unitig_file, config
            )
            pyseer_tracking.append(tracking)

            if tracking['status'] == 'completed':
                return permanova_result, pyseer_tracking, output_file
            else:
                return permanova_result, pyseer_tracking, None
        else:
            skip_reason = f"Insufficient samples for Pyseer (n1={n_group1}, n2={n_group2}, min_required={config.args.pair_pyseer_min_n})"
            logger.info(f"Pairwise Pyseer skipped for {species}__{variable}__{comparison_name}: {skip_reason}")
            pyseer_tracking.append(_track_pyseer_run(
                species, f"{variable}__{comparison_name}", 'pairwise',
                str(pheno_file), unitig_file,
                str(pairwise_pyseer_dir / f"{species}__{variable}__{comparison_name}.pyseer.tsv"),
                'skipped', skip_reason
            ))

    return permanova_result, pyseer_tracking, None


def _filter_significant_categorical(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter for significant categorical results that warrant pairwise testing.

    Args:
        results_df: Results DataFrame from distance tests

    Returns:
        Filtered DataFrame with significant categorical results
    """
    if results_df.empty:
        return pd.DataFrame()

    return results_df[
        (results_df['type'] == 'categorical') &
        (results_df['exact_test'] == 'PERMANOVA') &
        (results_df['exact_pvalue'] <= SIGNIFICANCE_ALPHA)
    ].copy()


def _run_pairwise_tests(
    species: str,
    config: AnalysisConfig,
    mash_tsv: str,
    results_df: pd.DataFrame,
    unitig_file: str,
) -> List[Dict[str, str]]:
    """
    Run pairwise PERMANOVA and optional Pyseer for significant categorical results.

    Args:
        species: Species name
        config: Analysis configuration
        mash_tsv: Path to Mash distance matrix
        results_df: Results DataFrame from distance tests
        unitig_file: Path to unitig file

    Returns:
        List of tracking dictionaries for pyseer runs
    """
    pyseer_tracking = []

    try:
        # Filter for significant categorical results
        significant_categorical = _filter_significant_categorical(results_df)

        if significant_categorical.empty:
            return pyseer_tracking

        # Initialize worker for permutation tests
        _assoc_init_worker(str(mash_tsv))

        # Setup directories
        pairwise_pyseer_dir = config.pyseer_dir / species / 'pairwise'
        pairwise_pyseer_dir.mkdir(parents=True, exist_ok=True)
        pairwise_dist_dir = config.distance_assoc_dir / species
        pairwise_dist_dir.mkdir(parents=True, exist_ok=True)

        pairwise_outputs = []
        pairwise_results = []

        # Process each significant categorical variable
        for _, result_row in significant_categorical.iterrows():
            variable = result_row['metadata']

            # Find matching phenotype info
            matching_phenotypes = [p for p in config.phenos.get(species, []) if p.variable == variable]
            if not matching_phenotypes:
                continue

            pheno_tsv = matching_phenotypes[0].pheno_tsv_path

            # Load phenotype data
            try:
                phenotype_df = pd.read_csv(pheno_tsv, sep='\t').dropna(subset=['phenotype'])
                phenotype_df['phenotype'] = phenotype_df['phenotype'].astype(str)
            except (OSError, IOError, pd.errors.ParserError) as e:
                logger.error(f"Could not load phenotype file {pheno_tsv}: {e}")
                continue

            groups = sorted(phenotype_df['phenotype'].unique())

            # One-vs-rest comparisons
            for group in groups:
                permanova_result, tracking, output_file = _create_pairwise_comparison(
                    species, variable, phenotype_df,
                    group, 'rest', 'one_vs_rest',
                    config, pairwise_pyseer_dir, unitig_file
                )

                if permanova_result:
                    pairwise_results.append(permanova_result)
                pyseer_tracking.extend(tracking)
                if output_file:
                    pairwise_outputs.append(output_file)

            # Group-vs-group comparisons
            for group1, group2 in itertools.combinations(groups, 2):
                permanova_result, tracking, output_file = _create_pairwise_comparison(
                    species, variable, phenotype_df,
                    group1, group2, 'group_vs_group',
                    config, pairwise_pyseer_dir, unitig_file
                )

                if permanova_result:
                    pairwise_results.append(permanova_result)
                pyseer_tracking.extend(tracking)
                if output_file:
                    pairwise_outputs.append(output_file)

        # Save pairwise PERMANOVA results
        if pairwise_results:
            pairwise_df = pd.DataFrame(pairwise_results)
            pairwise_file = pairwise_dist_dir / f"{species}.pairwise_permanova.tsv"
            pairwise_df.to_csv(pairwise_file, sep='\t', index=False)
            logger.info(f"Saved pairwise PERMANOVA results: {pairwise_file}")

        # Combine pairwise Pyseer outputs
        if pairwise_outputs:
            parts = []
            for output_path in pairwise_outputs:
                try:
                    df_part = pd.read_csv(output_path, sep='\t')
                    df_part['__source__'] = output_path.name
                    parts.append(df_part)
                except (OSError, IOError, pd.errors.ParserError):
                    pass

            if parts:
                combined = config.pyseer_dir / species / f"{species}.pairwise.pyseer.tsv"
                pd.concat(parts, ignore_index=True).to_csv(combined, sep='\t', index=False)
                logger.info(f"Saved pairwise pyseer results: {combined}")

    except (OSError, IOError, pd.errors.ParserError) as e:
        logger.error(f"Pairwise testing failed for {species}: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"Unexpected error in pairwise testing for {species}: {e}", exc_info=True)

    return pyseer_tracking


def _run_pyseer_gwas(species: str, config: AnalysisConfig, unitig_file: str) -> List[Dict[str, str]]:
    """
    Run Pyseer GWAS for binary/continuous phenotypes.

    Args:
        species: Species name
        config: Analysis configuration
        unitig_file: Path to unitig file

    Returns:
        List of tracking dictionaries for Pyseer runs
    """
    threads_per_worker = max(1, config.args.threads_per_worker)
    phenotypes = config.phenos.get(species, [])
    pyseer_tracking = []

    # Create species-specific pyseer directory
    species_pyseer_dir = config.pyseer_dir / species
    species_pyseer_dir.mkdir(parents=True, exist_ok=True)

    for pheno_info in phenotypes:
        var, typ, pheno_tsv = pheno_info
        if typ not in ('binary', 'continuous'):
            continue

        logger.info(f"Running pyseer GWAS for {species}__{var} ({typ})")
        output_file = species_pyseer_dir / f'{species}__{var}.pyseer.tsv'
        stderr_log = species_pyseer_dir / f'{species}__{var}.pyseer.stderr.log'

        # Check preconditions
        can_run, precondition_reason = _check_pyseer_preconditions(pheno_tsv, min_samples=PYSEER_MIN_SAMPLES_TOTAL)
        if not can_run:
            logger.warning(f"Pyseer skipped for {species}__{var}: {precondition_reason}")
            pyseer_tracking.append(_track_pyseer_run(
                species, var, 'main_phenotype', pheno_tsv, unitig_file, str(output_file),
                'skipped', precondition_reason
            ))
            continue

        # Validate inputs
        is_valid, validation_errors = _validate_pyseer_inputs(pheno_tsv, unitig_file)
        if not is_valid:
            error_msg = "; ".join(validation_errors)
            logger.error(f"Pyseer input validation failed for {species}__{var}: {error_msg}")
            pyseer_tracking.append(_track_pyseer_run(
                species, var, 'main_phenotype', pheno_tsv, unitig_file, str(output_file),
                'failed', error_msg
            ))
            continue

        # Track start
        pyseer_tracking.append(_track_pyseer_run(
            species, var, 'main_phenotype', pheno_tsv, unitig_file, str(output_file),
            'started'
        ))

        # Run pyseer with stderr logging
        try:
            with open(output_file, 'w') as stdout_fh, open(stderr_log, 'w') as stderr_fh:
                cmd = [
                    'pyseer',
                    '--phenotypes', pheno_tsv,
                    '--kmers', unitig_file,
                    '--uncompressed',
                    '--min-af', str(config.args.maf),
                    '--cpu', str(threads_per_worker),
                    '--no-distances'
                ]
                subprocess.check_call(cmd, stdout=stdout_fh, stderr=stderr_fh)

            if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                logger.info(f"Pyseer completed successfully for {species}__{var}")
                add_bh(str(output_file), str(species_pyseer_dir / f'{species}__{var}.pyseer.fdr.tsv'))
                pyseer_tracking.append(_track_pyseer_run(
                    species, var, 'main_phenotype', pheno_tsv, unitig_file, str(output_file),
                    'completed'
                ))
            else:
                error_msg = "Pyseer output is empty"
                logger.warning(f"Pyseer output is empty for {species}__{var}")
                pyseer_tracking.append(_track_pyseer_run(
                    species, var, 'main_phenotype', pheno_tsv, unitig_file, str(output_file),
                    'failed', error_msg
                ))

        except subprocess.CalledProcessError as e:
            error_msg = f"Pyseer failed with return code {e.returncode}"
            if stderr_log.exists():
                error_msg += f" (see {stderr_log} for details)"
            logger.error(f"Pyseer failed for {species}__{var}: {error_msg}")
            pyseer_tracking.append(_track_pyseer_run(
                species, var, 'main_phenotype', pheno_tsv, unitig_file, str(output_file),
                'failed', error_msg
            ))
        except (OSError, IOError) as e:
            error_msg = f"I/O error: {e}"
            logger.error(f"I/O error running pyseer for {species}__{var}: {e}", exc_info=True)
            pyseer_tracking.append(_track_pyseer_run(
                species, var, 'main_phenotype', pheno_tsv, unitig_file, str(output_file),
                'failed', error_msg
            ))
        except Exception as e:
            error_msg = f"Unexpected error: {e}"
            logger.error(f"Unexpected error running pyseer for {species}__{var}: {e}", exc_info=True)
            pyseer_tracking.append(_track_pyseer_run(
                species, var, 'main_phenotype', pheno_tsv, unitig_file, str(output_file),
                'failed', error_msg
            ))

    return pyseer_tracking


# ----- Worker -----

def _worker(
    species: str,
    samples: List[str],
    config: AnalysisConfig
) -> Tuple[pd.DataFrame, List[Dict[str, str]]]:
    """
    Worker function for parallel processing of species-specific analyses.

    Args:
        species: Species name
        samples: List of sample names for this species
        config: Analysis configuration

    Returns:
        Tuple of (results DataFrame, pyseer tracking list)
    """
    logger.info(f"Starting processing for species: {species}")

    # Check if species has any valid phenotypes
    species_phenos = config.phenos.get(species, [])
    if not species_phenos:
        logger.warning(f"No phenotypes found for species {species} - skipping mash/unitig")
        return pd.DataFrame(columns=['species', 'metadata', 'test', 'stat', 'pvalue', 'n_samples']), []

    # Check if any phenotypes actually passed filters by reading the summary file
    if not has_passing_phenotypes(species, config.phenos_dir):
        logger.warning(f"No passing phenotypes found for species {species} - skipping mash/unitig")
        return pd.DataFrame(columns=['species', 'metadata', 'test', 'stat', 'pvalue', 'n_samples']), []

    # Initialize Pyseer tracking for this worker
    pyseer_tracking = []

    try:
        # Step 1: Mash distance calculation
        mash_tsv = _run_mash_for_species(species, samples, config)

        # Step 2: Unitig calling
        unitig_file = _run_unitigs_for_species(species, samples, config)

        # Step 3: Distance-based association tests
        rows, results_df = _run_distance_tests(species, config, mash_tsv)

        # Step 4: Pyseer GWAS for binary/continuous phenotypes
        gwas_tracking = _run_pyseer_gwas(species, config, unitig_file)
        pyseer_tracking.extend(gwas_tracking)

        # Step 5: Pairwise testing for categorical phenotypes with significant global association
        pairwise_tracking = _run_pairwise_tests(
            species=species,
            config=config,
            mash_tsv=mash_tsv,
            results_df=results_df,
            unitig_file=unitig_file
        )
        pyseer_tracking.extend(pairwise_tracking)

        # Processing completed successfully
        if rows:
            return pd.concat(rows, ignore_index=True), pyseer_tracking
        else:
            return pd.DataFrame(columns=['species', 'metadata', 'test', 'stat', 'pvalue', 'n_samples']), pyseer_tracking

    except (OSError, IOError, pd.errors.ParserError) as e:
        logger.error(f"Processing failed for species {species}: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Unexpected error processing species {species}: {e}", exc_info=True)
        raise


# ----- Main Function Components -----

def _setup_directories(output_dir: pathlib.Path) -> Dict[str, pathlib.Path]:
    """
    Create and return all required output directories.

    Args:
        output_dir: Base output directory

    Returns:
        Dictionary of directory paths
    """
    directories = {
        'out': output_dir,
        'work': output_dir / 'tmp',
        'mash': output_dir / 'mash_by_species',
        'pyseer': output_dir / 'pyseer_by_species',
        'distance_assoc': output_dir / 'distance_assoc_by_species',
        'unitig': output_dir / 'unitigs_by_species',
        'assoc': output_dir / 'assoc'
    }

    for dir_path in directories.values():
        ensure_dir(dir_path)

    return directories


def _validate_species_names(ani: pd.DataFrame) -> None:
    """
    Validate species names for security (prevent path traversal).

    Args:
        ani: ANI mapping DataFrame

    Raises:
        SystemExit: If invalid species names are found
    """
    invalid_species = []
    for species_name in ani['species'].unique():
        try:
            safe_sample_name(str(species_name))
        except ValueError as e:
            invalid_species.append(f"{species_name}: {e}")

    if invalid_species:
        logger.fatal(f"Invalid species names in ANI mapping (first 5): {invalid_species[:5]}")
        sys.exit(1)


def _load_and_validate_data(args: argparse.Namespace) -> Tuple[Dict[str, str], pd.DataFrame]:
    """
    Load and validate input data (FASTAs and ANI mapping).

    Args:
        args: Command line arguments

    Returns:
        Tuple of (sample_to_path dict, ANI DataFrame)
    """
    # Enumerate FASTAs (basename == sample id)
    sample_to_path = list_fastas(args.genomes)

    # CheckM filter (keeps all if file missing/empty)
    if args.checkm:
        pre_n = len(sample_to_path)
        sample_to_path = filter_by_checkm(sample_to_path, args.checkm, args.comp_min, args.cont_max)
        post_n = len(sample_to_path)
        if post_n == 0:
            logger.fatal(f"CheckM filter removed all samples (comp_min={args.comp_min}, cont_max={args.cont_max})")
            sys.exit(1)
        if post_n < pre_n:
            logger.info(f"CheckM filter kept {post_n}/{pre_n} samples")

    # Load ANI mapping with validation
    try:
        ani = pd.read_csv(args.ani_map, sep='\t', names=['species', 'sample'])
    except (OSError, IOError) as e:
        logger.fatal(f"Cannot read ANI mapping file {args.ani_map}: {e}")
        sys.exit(1)
    except pd.errors.ParserError as e:
        logger.fatal(f"Invalid TSV format in ANI mapping {args.ani_map}: {e}")
        sys.exit(1)

    if ani.empty:
        logger.fatal(f"ANI mapping file is empty: {args.ani_map}")
        sys.exit(1)

    # Validate sample names for security (prevent path traversal)
    invalid_samples = []
    for sample in ani['sample']:
        try:
            safe_sample_name(str(sample))
        except ValueError as e:
            invalid_samples.append(f"{sample}: {e}")

    if invalid_samples:
        logger.fatal(f"Invalid sample names in ANI mapping (first 5): {invalid_samples[:5]}")
        sys.exit(1)

    # Validate species names
    _validate_species_names(ani)

    ani = ani.drop_duplicates()

    return sample_to_path, ani


def _prepare_phenotypes(
    args: argparse.Namespace,
    ani: pd.DataFrame,
    sample_to_path: Dict[str, str],
    work_dir: pathlib.Path,
    eligible_species: set
) -> Tuple[pathlib.Path, PhenotypeDict]:
    """
    Prepare phenotype files with resume logic.

    Args:
        args: Command line arguments
        ani: ANI mapping DataFrame
        sample_to_path: Sample to path mapping
        work_dir: Working directory
        eligible_species: Set of eligible species

    Returns:
        Tuple of (phenotypes directory, phenotypes dictionary)
    """
    phenos_dir = work_dir / 'phenos'
    phenos_dir.mkdir(parents=True, exist_ok=True)

    species_list = sorted(ani['species'].unique())

    # Create temporary ANI map limited to eligible species
    temp_ani_map_all = work_dir / 'temp_ani_map_eligible.tsv'
    ani_eligible = ani[ani['species'].isin(eligible_species)]
    ani_eligible.to_csv(temp_ani_map_all, sep='\t', index=False, header=False)

    if args.resume:
        # Check which species have complete phenotype files
        complete_pheno_species = set()
        incomplete_pheno_species = set()

        for species in species_list:
            if are_phenotype_files_complete(species, phenos_dir):
                complete_pheno_species.add(species)
            else:
                incomplete_pheno_species.add(species)

        if complete_pheno_species:
            logger.info(f"Found complete phenotype files for {len(complete_pheno_species)} species")

        if incomplete_pheno_species:
            logger.info(f"Need to regenerate phenotype files for {len(incomplete_pheno_species)} species")
            # Only regenerate for incomplete species
            temp_ani_map = work_dir / 'temp_ani_map.tsv'
            incomplete_ani = ani[ani['species'].isin(incomplete_pheno_species)]
            incomplete_ani.to_csv(temp_ani_map, sep='\t', index=False, header=False)

            # Generate phenotype files for incomplete species only (but still restricted to eligible species)
            filter_metadata_per_species(
                metadata_fp=args.metadata, ani_map_fp=str(temp_ani_map), out_dir=str(phenos_dir),
                min_n=args.min_n, max_missing_frac=args.max_missing_frac,
                min_level_n=args.min_level_n, min_unique_cont=args.min_unique_cont,
                custom_missing_values=args.missing_values
            )

        # Load all phenotype information (existing + newly generated)
        phenos = load_phenotype_manifest(phenos_dir)

        # Optional filtering by phenotype substring
        if args.phenotype:
            phenotype_filter = args.phenotype.lower()
            for sp in list(phenos.keys()):
                phenos[sp] = [p for p in phenos[sp] if phenotype_filter in p.variable.lower()]
    else:
        # No resume - regenerate all phenotype files (restricted to eligible species)
        logger.info("Generating all phenotype files")

        phenos = filter_metadata_per_species(
            metadata_fp=args.metadata, ani_map_fp=str(temp_ani_map_all), out_dir=str(phenos_dir),
            min_n=args.min_n, max_missing_frac=args.max_missing_frac,
            min_level_n=args.min_level_n, min_unique_cont=args.min_unique_cont,
            custom_missing_values=args.missing_values,
            phenotype_filter=args.phenotype
        )

    return phenos_dir, phenos


def _determine_species_to_process(
    species_to_samples: Dict[str, List[str]],
    phenos: PhenotypeDict,
    phenos_dir: pathlib.Path,
    args: argparse.Namespace,
    directories: Dict[str, pathlib.Path]
) -> Dict[str, List[str]]:
    """
    Determine which species need to be processed.

    Args:
        species_to_samples: Mapping of species to samples
        phenos: Phenotype dictionary
        phenos_dir: Phenotypes directory
        args: Command line arguments
        directories: Output directories

    Returns:
        Dictionary of species to process
    """
    # Get remaining species based on file existence
    remaining_species = _get_remaining_species(
        species_to_samples, args, phenos,
        directories['mash'], directories['pyseer'],
        directories['distance_assoc'], directories['unitig']
    )

    # Filter out species with no valid phenotypes
    species_with_valid_phenos = {}
    skipped_species = []

    for sp, samples in remaining_species.items():
        # Check multiple conditions:
        # 1. Species must be in phenos dict
        # 2. Species must have at least one phenotype entry
        # 3. At least one phenotype must have passed filters (check summary file)
        if sp not in phenos:
            skipped_species.append(f"{sp} (not in phenos)")
            continue
        if len(phenos[sp]) == 0:
            skipped_species.append(f"{sp} (empty phenotype list)")
            continue
        if not has_passing_phenotypes(sp, phenos_dir):
            skipped_species.append(f"{sp} (no passing phenotypes)")
            continue
        # All checks passed
        species_with_valid_phenos[sp] = samples

    if skipped_species:
        logger.info(f"Skipping {len(skipped_species)} species with no passing phenotypes:")
        for skipped in skipped_species[:10]:  # Log first 10
            logger.info(f"  - {skipped}")
        if len(skipped_species) > 10:
            logger.info(f"  ... and {len(skipped_species) - 10} more")

    return species_with_valid_phenos


def _run_parallel_processing(
    remaining_species: Dict[str, List[str]],
    config: AnalysisConfig
) -> Tuple[List[pd.DataFrame], List[Dict[str, str]]]:
    """
    Run parallel processing for all species.

    Args:
        remaining_species: Species to process
        config: Analysis configuration

    Returns:
        Tuple of (results list, pyseer tracking list)
    """
    results = []
    pyseer_tracking = []

    with ProcessPoolExecutor(max_workers=config.args.workers) as executor:
        futures = []
        for species, samples in remaining_species.items():
            future = executor.submit(_worker, species, samples, config)
            futures.append(future)

        for future in as_completed(futures):
            try:
                df, worker_tracking = future.result()
                if not df.empty:
                    results.append(df)
                pyseer_tracking.extend(worker_tracking)
            except Exception as e:
                logger.error(f"Worker failed: {e}", exc_info=True)

    return results, pyseer_tracking


def _write_results(
    results: List[pd.DataFrame],
    pyseer_tracking: List[Dict[str, str]],
    output_dir: pathlib.Path,
    pyseer_dir: pathlib.Path
) -> None:
    """
    Write final combined results and summaries.

    Args:
        results: List of result DataFrames
        pyseer_tracking: List of pyseer tracking dicts
        output_dir: Output directory
        pyseer_dir: Pyseer directory
    """
    assoc_dir = output_dir / 'assoc'

    # Write distance association results
    if results:
        master = assoc_dir / 'distance_associations_all_species.tsv'
        combined_df = pd.concat(results, ignore_index=True)
        combined_df.to_csv(master, sep='\t', index=False)

        # Apply FDR correction
        fdr_master = assoc_dir / 'distance_associations_all_species.fdr.tsv'
        add_bh(str(master), str(fdr_master))
        logger.info(f"Final distance association results written to {master}")
        logger.info(f"FDR-corrected results written to {fdr_master}")
    else:
        logger.info("No distance association results to write")

    # Write Pyseer tracking report
    if pyseer_tracking:
        tracking_report_file = output_dir / 'pyseer_tracking_report.tsv'
        _write_pyseer_tracking_report(pyseer_tracking, tracking_report_file)
    else:
        logger.warning("No Pyseer tracking data collected")

    # Summarize pyseer results
    logger.info("Summarizing pyseer results...")
    summarize_pyseer(
        indir=str(pyseer_dir),
        out=str(assoc_dir / 'pyseer_summary.tsv'),
        alpha=SIGNIFICANCE_ALPHA,
        pattern="**/*.pyseer.tsv"
    )


def main() -> None:
    """
    Main CLI entry point for panpiper-kit.

    Performs ANI-split Mash lineage tests and per-species unitig GWAS with optional CheckM filtering
    and parallel processing across species.

    This pipeline processes genomic data through the following steps:
    1. Load and filter samples based on CheckM quality metrics (optional)
    2. Generate phenotype files per species from metadata
    3. Compute Mash distances within each species
    4. Generate unitigs for GWAS analysis
    5. Run distance-based association tests (PERMANOVA/Mantel or fast PCoA-based)
    6. Run pyseer GWAS for binary/continuous phenotypes
    7. Apply FDR correction to results

    Features:
    - Resume capability: Skip previously completed species on restart
    - File verification: Check actual file existence for completeness
    - Parallel processing: Process multiple species simultaneously
    - Flexible phenotype types: Binary, categorical, and continuous phenotypes
    """
    ap = argparse.ArgumentParser(
        description='ANI-split Mash lineage tests + per-species unitig GWAS (parallel)'
    )
    ap.add_argument('--genomes', required=True, help='directory of FASTA files')
    ap.add_argument('--metadata', required=True, help='TSV with SampleID column and phenotype data')
    ap.add_argument('--ani-map', required=True, help='TSV with cluster/species<TAB>bin_identifier (from FASTA basename)')
    ap.add_argument('--out', required=True, help='output directory')
    ap.add_argument('--checkm', default=None, help='CheckM/quality TSV (columns like sample/bin/genome, completeness/comp, contamination/contam)')
    ap.add_argument('--comp-min', type=float, default=80.0, help='minimum completeness to keep (default 80.0)')
    ap.add_argument('--cont-max', type=float, default=10.0, help='maximum contamination to keep (default 10.0)')
    ap.add_argument('--threads', type=int, default=32, help='total CPU budget')
    ap.add_argument('--workers', type=int, default=4, help='number of species processed in parallel')
    ap.add_argument('--perms', type=int, default=999, help='starting number of permutations for PERMANOVA/Mantel (can auto-escalate to max-perms)')
    ap.add_argument('--max-perms', type=int, default=999999, help='maximum permutations (can go beyond 1M with custom implementation, default: 999999)')
    ap.add_argument('--perm-mode', choices=['auto', 'standard', 'large', 'analytical'], default='auto',
                    help='Permutation strategy: auto=adaptive (uses analytical for n>500, large for >10k perms), '
                         'standard=scikit-bio only, large=custom 1M+ implementation, analytical=F-distribution approximation')
    ap.add_argument('--tests', choices=['exact', 'fast'], default='fast', help='exact=PERMANOVA/Mantel; fast=PC-based ANOVA/OLS (no permutations)')
    ap.add_argument('--max-axes', type=int, default=10, help='PC axes used in fast mode')
    ap.add_argument('--mash-k', type=int, default=18, help='Mash k-mer size')
    ap.add_argument('--mash-s', type=int, default=10000, help='Mash sketch size')
    ap.add_argument('--kmer', type=int, default=31, help='unitig-caller k-mer size')
    ap.add_argument('--maf', type=float, default=0.05, help='pyseer min allele freq for unitigs')
    ap.add_argument('--pair-min-n', type=int, default=20, help='minimum samples per group for pairwise tests')
    ap.add_argument('--pair-pyseer-min-n', type=int, default=50, help='minimum samples per group to run pyseer in pairwise tests')
    ap.add_argument('--min-n', type=int, default=DEFAULT_MIN_SAMPLES, help='minimum usable samples per species')
    ap.add_argument('--max-missing-frac', type=float, default=DEFAULT_MAX_MISSING_FRAC, help='max fraction of missing values allowed for a phenotype')
    ap.add_argument('--min-level-n', type=int, default=DEFAULT_MIN_LEVEL_N, help='min samples required per category level')
    ap.add_argument('--min-unique-cont', type=int, default=DEFAULT_MIN_UNIQUE_CONT, help='min unique values required for continuous phenotypes')
    ap.add_argument('--resume', action='store_true', default=False,
                    help='Resume from previous run (skip completed species)')
    ap.add_argument('--missing-values', nargs='*', default=None,
                    help='Additional missing value indicators to treat as NaN (space-separated)')
    ap.add_argument('--phenotype', default=None,
                    help='Optional substring to select a single phenotype to test (case-insensitive). If omitted, test all.')
    args = ap.parse_args()

    # Derive per-worker threads
    args.threads_per_worker = max(1, args.threads // max(1, args.workers))
    os.environ['OMP_NUM_THREADS'] = str(args.threads_per_worker)
    os.environ['OPENBLAS_NUM_THREADS'] = str(args.threads_per_worker)
    os.environ['MKL_NUM_THREADS'] = str(args.threads_per_worker)
    os.environ['NUMEXPR_NUM_THREADS'] = str(args.threads_per_worker)

    # Setup directories
    directories = _setup_directories(pathlib.Path(args.out))

    # Log run information
    logger.info(f"Output directory: {directories['out']}")
    if args.resume:
        logger.info("Resume mode: Will skip completed species based on file existence")
    else:
        logger.info("Processing all species from scratch")

    # Load and validate data
    sample_to_path, ani = _load_and_validate_data(args)

    # Build species -> sample list early to restrict phenotype generation to eligible species
    species_to_samples_all = _build_species_sample_map(ani, sample_to_path, args.min_n)
    eligible_species = set(species_to_samples_all.keys())
    logger.info(f"Eligible species for processing (N>={args.min_n}): {len(eligible_species)}")

    # Prepare phenotypes
    phenos_dir, phenos = _prepare_phenotypes(
        args, ani, sample_to_path, directories['work'], eligible_species
    )

    # Determine which species to process
    remaining_species = _determine_species_to_process(
        species_to_samples_all, phenos, phenos_dir, args, directories
    )

    # Generate comprehensive summary
    _generate_analysis_summary(
        phenos, species_to_samples_all, remaining_species, args,
        directories['pyseer'], directories['distance_assoc']
    )

    # Generate expected Pyseer runs list
    expected_pyseer_runs = _generate_expected_pyseer_runs(phenos, directories['pyseer'])
    logger.info(f"Generated {len(expected_pyseer_runs)} expected Pyseer runs")

    # Write expected runs to file for reference
    if expected_pyseer_runs:
        expected_runs_file = directories['out'] / 'expected_pyseer_runs.tsv'
        pd.DataFrame(expected_pyseer_runs).to_csv(expected_runs_file, sep='\t', index=False)
        logger.info(f"Expected Pyseer runs written to: {expected_runs_file}")

    # Create analysis configuration
    config = AnalysisConfig(
        args=args,
        phenos=phenos,
        mash_dir=directories['mash'],
        pyseer_dir=directories['pyseer'],
        distance_assoc_dir=directories['distance_assoc'],
        unitig_dir=directories['unitig'],
        phenos_dir=phenos_dir,
        s2p=sample_to_path
    )

    # Run parallel processing
    results, pyseer_tracking = _run_parallel_processing(remaining_species, config)

    # Collect results from previously completed species if resuming
    if args.resume:
        # Find completed species based on file existence
        actually_complete = set()
        for sp in species_to_samples_all.keys():
            if is_species_complete(sp, phenos, directories['mash'], directories['pyseer'],
                                 directories['distance_assoc'], directories['unitig']):
                actually_complete.add(sp)

        existing_results = _collect_existing_results(actually_complete, phenos, directories['distance_assoc'])
        results.extend(existing_results)

    # Write final results
    _write_results(results, pyseer_tracking, directories['out'], directories['pyseer'])

    logger.info("Pipeline completed successfully!")


if __name__ == '__main__':
    main()

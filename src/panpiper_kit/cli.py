import argparse
import logging
import pathlib
import pandas as pd
import os
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple, Any

from .files import list_fastas, ensure_dir, run
from .filter import filter_metadata_per_species, filter_by_checkm
from .mash import mash_within_species
from .assoc import run_assoc, PhenotypeJob, _init_worker as _assoc_init_worker, _permutation_test as _assoc_permutation_test
from .gwas import ensure_unitigs
from .fdr import add_bh
from .summarize_pyseer import summarize_pyseer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
DEFAULT_MIN_SAMPLES = 30
DEFAULT_MAX_MISSING_FRAC = 0.2
DEFAULT_MIN_LEVEL_N = 5
DEFAULT_MIN_UNIQUE_CONT = 3

# ----- helper functions -----

def _generate_expected_pyseer_runs(phenos: Dict[str, List[Tuple[str, str, str]]],
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
        for var, typ, pheno_tsv in phenotypes:
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


def _check_pyseer_preconditions(pheno_tsv: str, min_samples: int = 2) -> Tuple[bool, str]:
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
            if min_count < 2:  # Pyseer needs at least 2 samples per group
                return False, f"Insufficient samples per group (min: {min_count})"
        
        return True, ""
        
    except Exception as e:
        return False, f"Error reading phenotype file: {e}"


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
            elif len(df) < 2:
                errors.append(f"Phenotype file has insufficient samples (<2): {pheno_tsv}")
        except Exception as e:
            errors.append(f"Phenotype file is not readable: {pheno_tsv} - {e}")
    
    # Check unitig file
    if not os.path.exists(unitig_file):
        errors.append(f"Unitig file does not exist: {unitig_file}")
    else:
        try:
            if os.path.getsize(unitig_file) == 0:
                errors.append(f"Unitig file is empty: {unitig_file}")
        except Exception as e:
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


def _generate_analysis_summary(phenos: Dict[str, List[Tuple[str, str, str]]],
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
    all_analyses = []
    
    for species in sorted(sp_to_samples.keys()):
        species_phenos = phenos.get(species, [])
        n_samples = len(sp_to_samples[species])
        is_remaining = species in remaining_species
        
        logger.info(f"SPECIES: {species} (N={n_samples}) {'[REMAINING]' if is_remaining else '[COMPLETED]'}")
        logger.info("-" * 60)
        
        if not species_phenos:
            logger.info("  No phenotypes found for this species")
            continue
            
        for var, typ, pheno_tsv in species_phenos:
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
                        max_group = value_counts.max()
                        logger.info(f"  {var} ({typ}): N={n_valid} | Groups: {dict(value_counts)} | Min group: {min_group}")
                        
                        # Check Pyseer requirements
                        if min_group < 2:
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
                
            except Exception as e:
                logger.info(f"  {var} ({typ}): ERROR reading file - {e}")
                
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
                          phenos: Dict[str, List[Tuple[str, str, str]]],
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
    if args.resume and not args.force:
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
        if args.force:
            logger.info(f"Force mode: will re-run all {len(sp_to_samples)} species")
        return sp_to_samples


def _collect_existing_results(actually_complete: set, phenos: Dict[str, List[Tuple[str, str, str]]],
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
            except Exception as e:
                logger.warning(f"Could not load existing combined results for {sp}: {e}")
    return results

# ----- progress tracking -----

def is_species_complete(
    species: str,
    phenos: Dict[str, List[Tuple[str, str, str]]],
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
    for var, typ, _ in phenos.get(species, []):
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
    except Exception:
        # If we can't read the manifest, assume incomplete
        return False


def load_phenotype_manifest(phenos_dir: pathlib.Path) -> Dict[str, List[Tuple[str, str, str]]]:
    """
    Load phenotype information from existing list.tsv manifest files.
    
    Args:
        phenos_dir: Directory containing phenotype files and list.tsv manifests
        
    Returns:
        Dictionary mapping species names to lists of (variable, type, filepath) tuples
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
                species_phenos.append((row['variable'], row['type'], row['pheno_tsv']))
            
            phenos[species] = species_phenos
            logger.debug(f"Loaded {len(species_phenos)} phenotypes for {species}")
        except Exception as e:
            logger.warning(f"Could not load phenotype manifest for {species}: {e}")
    
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
            return df['passes_filters'].any()
        # If column doesn't exist, assume no phenotypes passed (old format)
        return False
    except Exception as e:
        logger.warning(f"Could not read phenotype summary for {species}: {e}")
        return False



# ----- worker helper functions -----

def _run_mash_for_species(species: str, sams: List[str], s2p: Dict[str, str],
                          args: argparse.Namespace, mash_dir: pathlib.Path) -> str:
    """
    Run Mash distance calculation for a species.

    Returns:
        Path to Mash distance matrix TSV
    """
    t = max(1, args.threads_per_worker)
    paths = [s2p[s] for s in sams if s in s2p]
    sp_out = mash_dir / species
    sp_out.mkdir(parents=True, exist_ok=True)

    logger.info(f"Processing {len(paths)} FASTA files for species {species}")
    mash_tsv = mash_within_species(paths, str(sp_out), k=args.mash_k, s=args.mash_s, threads=t)

    return mash_tsv


def _run_unitigs_for_species(species: str, sams: List[str], s2p: Dict[str, str],
                             args: argparse.Namespace, mash_dir: pathlib.Path,
                             unitig_dir: pathlib.Path) -> str:
    """
    Run unitig calling for a species.

    Returns:
        Path to unitig pyseer file
    """
    t = max(1, args.threads_per_worker)
    paths = [s2p[s] for s in sams if s in s2p]
    sp_out = mash_dir / species

    # Create refs file
    ref_txt = sp_out / 'refs.txt'
    if not ref_txt.exists():
        with open(ref_txt, 'w') as fh:
            fh.write('\n'.join(paths))

    uc_pyseer = ensure_unitigs(str(ref_txt), str(unitig_dir / species), kmer=args.kmer, threads=t)
    return uc_pyseer


def _run_distance_tests(species: str, phenos: Dict[str, List[Tuple[str, str, str]]],
                       args: argparse.Namespace, mash_tsv: str,
                       distance_assoc_dir: pathlib.Path) -> Tuple[List[pd.DataFrame], pd.DataFrame]:
    """
    Run distance-based association tests for all phenotypes of a species.

    Returns:
        Tuple of (list of result DataFrames, full results DataFrame)
    """
    phenotypes = phenos.get(species, [])
    if not phenotypes:
        logger.warning(f"No phenotypes found for species {species}")
        return [], pd.DataFrame()

    # Create PhenotypeJob objects
    phenotype_jobs = []
    for var, typ, pheno_tsv in phenotypes:
        if not os.path.exists(pheno_tsv):
            logger.error(f"Phenotype file does not exist: {pheno_tsv}")
            continue
        phenotype_jobs.append(PhenotypeJob(species=species, variable=var, typ=typ, pheno_tsv=pheno_tsv))

    if not phenotype_jobs:
        return [], pd.DataFrame()

    logger.info(f"Running association tests for {len(phenotype_jobs)} phenotypes")

    # Build permutation ladder
    if args.tests == 'exact':
        perm_ladder = []
        current = args.perms
        while current <= args.max_perms:
            perm_ladder.append(current)
            if current >= args.max_perms:
                break
            next_val = min(current * 10, args.max_perms)
            if next_val == current:
                break
            current = next_val
        perm_ladder = tuple(perm_ladder)
    else:
        perm_ladder = (199,)

    logger.info(f"Using permutation ladder: {perm_ladder}, mode: {args.perm_mode}")

    # Run association tests
    try:
        results_df = run_assoc(
            mash_tsv=str(mash_tsv),
            phenos=phenotype_jobs,
            n_workers=1,
            fast_p_thresh=0.10,
            fast_max_axes=args.max_axes,
            perm_ladder=perm_ladder,
            early_stop_p=0.20,
            escalate_p1=0.10,
            escalate_p2=0.05,
            escalate_p3=0.01,
            perm_mode=args.perm_mode
        )

        if not results_df.empty:
            logger.info(f"Association testing completed for {species}: {len(results_df)} results")

            # Create species-specific directory
            sp_dist_dir = distance_assoc_dir / species
            sp_dist_dir.mkdir(parents=True, exist_ok=True)

            # Save combined results file
            combined_file = sp_dist_dir / f'{species}.combined.dist_assoc.tsv'
            results_df.to_csv(combined_file, sep='\t', index=False)
            logger.info(f"Distance association results saved to: {combined_file}")

            # Save individual per-variable files as well for easier inspection
            for var, typ, _ in phenotypes:
                var_results = results_df[results_df['metadata'] == var]
                if not var_results.empty:
                    var_file = sp_dist_dir / f'{species}__{var}.dist_assoc.tsv'
                    var_results.to_csv(var_file, sep='\t', index=False)

            # Convert to list of row DataFrames for backward compatibility
            rows = [pd.DataFrame([row.to_dict()]) for _, row in results_df.iterrows()]
            return rows, results_df
        else:
            logger.warning(f"No results from association testing for {species}")
            return [], pd.DataFrame()

    except Exception as e:
        logger.error(f"Association testing failed for {species}: {e}", exc_info=True)
        return [], pd.DataFrame()


def _run_pyseer_gwas(species: str, phenos: Dict[str, List[Tuple[str, str, str]]],
                    args: argparse.Namespace, uc_pyseer: str,
                    pyseer_dir: pathlib.Path) -> List[Dict[str, str]]:
    """
    Run Pyseer GWAS for binary/continuous phenotypes.

    Returns:
        List of tracking dictionaries for Pyseer runs
    """
    t = max(1, args.threads_per_worker)
    phenotypes = phenos.get(species, [])
    worker_pyseer_tracking = []

    # Create species-specific pyseer directory
    sp_pyseer_dir = pyseer_dir / species
    sp_pyseer_dir.mkdir(parents=True, exist_ok=True)

    for var, typ, pheno_tsv in phenotypes:
        if typ not in ('binary', 'continuous'):
            continue

        logger.info(f"Running pyseer GWAS for {species}__{var} ({typ})")
        out_fp = sp_pyseer_dir / f'{species}__{var}.pyseer.tsv'

        # Check preconditions
        can_run, precondition_reason = _check_pyseer_preconditions(pheno_tsv, min_samples=2)
        if not can_run:
            logger.warning(f"Pyseer skipped for {species}__{var}: {precondition_reason}")
            worker_pyseer_tracking.append(_track_pyseer_run(
                species, var, 'main_phenotype', pheno_tsv, uc_pyseer, str(out_fp),
                'skipped', precondition_reason
            ))
            continue

        # Validate inputs
        is_valid, validation_errors = _validate_pyseer_inputs(pheno_tsv, uc_pyseer)
        if not is_valid:
            error_msg = "; ".join(validation_errors)
            logger.error(f"Pyseer input validation failed for {species}__{var}: {error_msg}")
            worker_pyseer_tracking.append(_track_pyseer_run(
                species, var, 'main_phenotype', pheno_tsv, uc_pyseer, str(out_fp),
                'failed', error_msg
            ))
            continue

        # Track start
        worker_pyseer_tracking.append(_track_pyseer_run(
            species, var, 'main_phenotype', pheno_tsv, uc_pyseer, str(out_fp),
            'started'
        ))

        # Run pyseer
        import subprocess
        try:
            with open(out_fp, 'w') as fh:
                cmd = ['pyseer',
                       '--phenotypes', pheno_tsv,
                       '--kmers', uc_pyseer, '--uncompressed',
                       '--min-af', str(args.maf), '--cpu', str(t), '--no-distances']
                subprocess.check_call(cmd, stdout=fh, stderr=subprocess.DEVNULL)

            if os.path.exists(out_fp) and os.path.getsize(out_fp) > 0:
                logger.info(f"Pyseer completed successfully for {species}__{var}")
                add_bh(str(out_fp), str(sp_pyseer_dir / f'{species}__{var}.pyseer.fdr.tsv'))
                worker_pyseer_tracking.append(_track_pyseer_run(
                    species, var, 'main_phenotype', pheno_tsv, uc_pyseer, str(out_fp),
                    'completed'
                ))
            else:
                error_msg = "Pyseer output is empty"
                logger.warning(f"Pyseer output is empty for {species}__{var}")
                worker_pyseer_tracking.append(_track_pyseer_run(
                    species, var, 'main_phenotype', pheno_tsv, uc_pyseer, str(out_fp),
                    'failed', error_msg
                ))

        except subprocess.CalledProcessError as e:
            error_msg = f"Pyseer failed with return code {e.returncode}"
            logger.error(f"Pyseer failed for {species}__{var}: {error_msg}")
            worker_pyseer_tracking.append(_track_pyseer_run(
                species, var, 'main_phenotype', pheno_tsv, uc_pyseer, str(out_fp),
                'failed', error_msg
            ))
        except Exception as e:
            error_msg = f"Unexpected error: {e}"
            logger.error(f"Unexpected error running pyseer for {species}__{var}: {e}", exc_info=True)
            worker_pyseer_tracking.append(_track_pyseer_run(
                species, var, 'main_phenotype', pheno_tsv, uc_pyseer, str(out_fp),
                'failed', error_msg
            ))

    return worker_pyseer_tracking


# ----- worker -----

def _worker(
    species: str,
    sams: List[str],
    s2p: Dict[str, str],
    args: argparse.Namespace,
    phenos: Dict[str, List[Tuple[str, str, str]]],
    mash_dir: pathlib.Path,
    pyseer_dir: pathlib.Path,
    distance_assoc_dir: pathlib.Path,
    unitig_dir: pathlib.Path,
    phenos_dir: pathlib.Path
) -> Tuple[pd.DataFrame, List[Dict[str, str]]]:
    """
    Worker function for parallel processing of species-specific analyses.

    Now refactored into smaller helper functions for better maintainability.
    """
    logger.info(f"Starting processing for species: {species}")

    # Check if species has any valid phenotypes
    species_phenos = phenos.get(species, [])
    if not species_phenos:
        logger.warning(f"No phenotypes found for species {species} - skipping mash/unitig")
        return pd.DataFrame(columns=['species','metadata','test','stat','pvalue','n_samples']), []
    
    # Check if any phenotypes actually passed filters by reading the summary file
    if not has_passing_phenotypes(species, phenos_dir):
        logger.warning(f"No passing phenotypes found for species {species} - skipping mash/unitig")
        return pd.DataFrame(columns=['species','metadata','test','stat','pvalue','n_samples']), []

    # Initialize Pyseer tracking for this worker
    worker_pyseer_tracking = []

    try:
        # Step 1: Mash distance calculation
        mash_tsv = _run_mash_for_species(species, sams, s2p, args, mash_dir)

        # Step 2: Unitig calling
        uc_pyseer = _run_unitigs_for_species(species, sams, s2p, args, mash_dir, unitig_dir)

        # Step 3: Distance-based association tests
        rows, results_df = _run_distance_tests(species, phenos, args, mash_tsv, distance_assoc_dir)

        # Step 4: Pyseer GWAS for binary/continuous phenotypes
        pyseer_tracking = _run_pyseer_gwas(species, phenos, args, uc_pyseer, pyseer_dir)
        worker_pyseer_tracking.extend(pyseer_tracking)

        # Step 5: Pairwise testing for categorical phenotypes with significant global association
        try:
            df_for_pairs = results_df if not results_df.empty else pd.DataFrame()
            if not df_for_pairs.empty:
                sig_cat = df_for_pairs[(df_for_pairs['type'] == 'categorical') &
                                       (df_for_pairs['exact_test'] == 'PERMANOVA') &
                                       (df_for_pairs['exact_pvalue'] <= 0.05)].copy()
            else:
                sig_cat = pd.DataFrame()

            if not sig_cat.empty:
                # Initialize DM in this process so we can reuse the exact test function
                _assoc_init_worker(str(mash_tsv))
                # Create species-specific pairwise directories
                pair_pyseer_dir = pyseer_dir / species / 'pairwise'
                pair_pyseer_dir.mkdir(parents=True, exist_ok=True)
                pair_dist_dir = distance_assoc_dir / species
                pair_dist_dir.mkdir(parents=True, exist_ok=True)
                pairwise_outputs = []
                pairwise_results = []  # Store pairwise PERMANOVA results
                phenotypes = phenos.get(species, [])

                for _, r in sig_cat.iterrows():
                    var = r['metadata']
                    src = [x for x in phenotypes if x[0] == var]
                    if not src:
                        continue
                    pheno_tsv = src[0][2]
                    ph = pd.read_csv(pheno_tsv, sep='\t').dropna(subset=['phenotype'])
                    ph['phenotype'] = ph['phenotype'].astype(str)
                    groups = sorted(ph['phenotype'].unique())
                    # One-vs-rest
                    for g in groups:
                        ph_bin = ph.copy()
                        ph_bin['phenotype'] = (ph_bin['phenotype'] == g).astype(int)
                        n_g = int((ph_bin['phenotype'] == 1).sum())
                        n_rest = int((ph_bin['phenotype'] == 0).sum())
                        if n_g < args.pair_min_n or n_rest < args.pair_min_n:
                            continue
                        tmp = pair_pyseer_dir / f"{species}__{var}__{g}_vs_rest.binary.tsv"
                        ph_bin[['sample','phenotype']].to_csv(tmp, sep='\t', index=False)
                        res = _assoc_permutation_test(str(tmp), 'binary', perms=args.perms, mode=args.perm_mode)
                        # Store pairwise PERMANOVA result
                        res['species'] = species
                        res['metadata'] = var
                        res['comparison'] = f"{g}_vs_rest"
                        res['group1'] = g
                        res['group2'] = "rest"
                        res['n_group1'] = n_g
                        res['n_group2'] = n_rest
                        pairwise_results.append(res)
                        if pd.notna(res.get('pvalue')) and res.get('pvalue') <= 0.05:
                            # Run pyseer if sufficiently large groups
                            if n_g >= args.pair_pyseer_min_n and n_rest >= args.pair_pyseer_min_n:
                                import subprocess
                                t = max(1, args.threads_per_worker)
                                out_fp = pair_pyseer_dir / f"{species}__{var}__{g}_vs_rest.pyseer.tsv"
                                
                                # Check preconditions for pairwise Pyseer
                                can_run, precondition_reason = _check_pyseer_preconditions(str(tmp), min_samples=2)
                                if not can_run:
                                    logger.warning(f"Pairwise Pyseer skipped for {species}__{var}__{g}_vs_rest: {precondition_reason}")
                                    worker_pyseer_tracking.append(_track_pyseer_run(
                                        species, f"{var}__{g}_vs_rest", 'pairwise', str(tmp), uc_pyseer, str(out_fp),
                                        'skipped', precondition_reason
                                    ))
                                else:
                                    # Validate inputs
                                    is_valid, validation_errors = _validate_pyseer_inputs(str(tmp), uc_pyseer)
                                    if not is_valid:
                                        error_msg = "; ".join(validation_errors)
                                        logger.error(f"Pairwise Pyseer input validation failed for {species}__{var}__{g}_vs_rest: {error_msg}")
                                        worker_pyseer_tracking.append(_track_pyseer_run(
                                            species, f"{var}__{g}_vs_rest", 'pairwise', str(tmp), uc_pyseer, str(out_fp),
                                            'failed', error_msg
                                        ))
                                    else:
                                        # Track start
                                        worker_pyseer_tracking.append(_track_pyseer_run(
                                            species, f"{var}__{g}_vs_rest", 'pairwise', str(tmp), uc_pyseer, str(out_fp),
                                            'started'
                                        ))
                                        
                                        try:
                                            with open(out_fp, 'w') as fh:
                                                cmd = ['pyseer','--phenotypes', str(tmp), '--kmers', uc_pyseer, '--uncompressed',
                                                       '--min-af', str(args.maf), '--cpu', str(t), '--no-distances']
                                                subprocess.check_call(cmd, stdout=fh, stderr=subprocess.DEVNULL)
                                            
                                            # Check if output was created and has content
                                            if os.path.exists(out_fp) and os.path.getsize(out_fp) > 0:
                                                logger.info(f"Pairwise Pyseer completed successfully for {species}__{var}__{g}_vs_rest")
                                                worker_pyseer_tracking.append(_track_pyseer_run(
                                                    species, f"{var}__{g}_vs_rest", 'pairwise', str(tmp), uc_pyseer, str(out_fp),
                                                    'completed'
                                                ))
                                                pairwise_outputs.append(out_fp)
                                            else:
                                                error_msg = "Pyseer output is empty"
                                                logger.warning(f"Pairwise Pyseer output is empty for {species}__{var}__{g}_vs_rest")
                                                worker_pyseer_tracking.append(_track_pyseer_run(
                                                    species, f"{var}__{g}_vs_rest", 'pairwise', str(tmp), uc_pyseer, str(out_fp),
                                                    'failed', error_msg
                                                ))
                                                
                                        except subprocess.CalledProcessError as e:
                                            error_msg = f"Pyseer failed with return code {e.returncode}"
                                            logger.error(f"Pairwise Pyseer failed for {species}__{var}__{g}_vs_rest: {error_msg}")
                                            worker_pyseer_tracking.append(_track_pyseer_run(
                                                species, f"{var}__{g}_vs_rest", 'pairwise', str(tmp), uc_pyseer, str(out_fp),
                                                'failed', error_msg
                                            ))
                                        except Exception as e:
                                            error_msg = f"Unexpected error: {e}"
                                            logger.error(f"Unexpected error running pairwise pyseer for {species}__{var}__{g}_vs_rest: {e}")
                                            worker_pyseer_tracking.append(_track_pyseer_run(
                                                species, f"{var}__{g}_vs_rest", 'pairwise', str(tmp), uc_pyseer, str(out_fp),
                                                'failed', error_msg
                                            ))
                            else:
                                # Track skipped due to sample size
                                skip_reason = f"Insufficient samples for Pyseer (n_g={n_g}, n_rest={n_rest}, min_required={args.pair_pyseer_min_n})"
                                logger.info(f"Pairwise Pyseer skipped for {species}__{var}__{g}_vs_rest: {skip_reason}")
                                worker_pyseer_tracking.append(_track_pyseer_run(
                                    species, f"{var}__{g}_vs_rest", 'pairwise', str(tmp), uc_pyseer, str(pair_pyseer_dir / f"{species}__{var}__{g}_vs_rest.pyseer.tsv"),
                                    'skipped', skip_reason
                                ))
                        # Group-vs-group
                for g1, g2 in itertools.combinations(groups, 2):
                    sub = ph[ph['phenotype'].isin([g1, g2])].copy()
                    if sub.empty:
                        continue
                    sub['phenotype'] = (sub['phenotype'] == g1).astype(int)
                    n1 = int((sub['phenotype'] == 1).sum())
                    n2 = int((sub['phenotype'] == 0).sum())
                    if n1 < args.pair_min_n or n2 < args.pair_min_n:
                        continue
                    tmp = pair_pyseer_dir / f"{species}__{var}__{g1}_vs_{g2}.binary.tsv"
                    sub[['sample','phenotype']].to_csv(tmp, sep='\t', index=False)
                    res = _assoc_permutation_test(str(tmp), 'binary', perms=args.perms, mode=args.perm_mode)
                    # Store pairwise PERMANOVA result
                    res['species'] = species
                    res['metadata'] = var
                    res['comparison'] = f"{g1}_vs_{g2}"
                    res['group1'] = g1
                    res['group2'] = g2
                    res['n_group1'] = n1
                    res['n_group2'] = n2
                    pairwise_results.append(res)
                    if pd.notna(res.get('pvalue')) and res.get('pvalue') <= 0.05:
                        if n1 >= args.pair_pyseer_min_n and n2 >= args.pair_pyseer_min_n:
                            import subprocess
                            t = max(1, args.threads_per_worker)
                            out_fp = pair_pyseer_dir / f"{species}__{var}__{g1}_vs_{g2}.pyseer.tsv"
                            
                            # Check preconditions for pairwise Pyseer
                            can_run, precondition_reason = _check_pyseer_preconditions(str(tmp), min_samples=2)
                            if not can_run:
                                logger.warning(f"Pairwise Pyseer skipped for {species}__{var}__{g1}_vs_{g2}: {precondition_reason}")
                                worker_pyseer_tracking.append(_track_pyseer_run(
                                    species, f"{var}__{g1}_vs_{g2}", 'pairwise', str(tmp), uc_pyseer, str(out_fp),
                                    'skipped', precondition_reason
                                ))
                            else:
                                # Validate inputs
                                is_valid, validation_errors = _validate_pyseer_inputs(str(tmp), uc_pyseer)
                                if not is_valid:
                                    error_msg = "; ".join(validation_errors)
                                    logger.error(f"Pairwise Pyseer input validation failed for {species}__{var}__{g1}_vs_{g2}: {error_msg}")
                                    worker_pyseer_tracking.append(_track_pyseer_run(
                                        species, f"{var}__{g1}_vs_{g2}", 'pairwise', str(tmp), uc_pyseer, str(out_fp),
                                        'failed', error_msg
                                    ))
                                else:
                                    # Track start
                                    worker_pyseer_tracking.append(_track_pyseer_run(
                                        species, f"{var}__{g1}_vs_{g2}", 'pairwise', str(tmp), uc_pyseer, str(out_fp),
                                        'started'
                                    ))
                                    
                                    try:
                                        with open(out_fp, 'w') as fh:
                                            cmd = ['pyseer','--phenotypes', str(tmp), '--kmers', uc_pyseer, '--uncompressed',
                                                   '--min-af', str(args.maf), '--cpu', str(t), '--no-distances']
                                            subprocess.check_call(cmd, stdout=fh, stderr=subprocess.DEVNULL)
                                        
                                        # Check if output was created and has content
                                        if os.path.exists(out_fp) and os.path.getsize(out_fp) > 0:
                                            logger.info(f"Pairwise Pyseer completed successfully for {species}__{var}__{g1}_vs_{g2}")
                                            worker_pyseer_tracking.append(_track_pyseer_run(
                                                species, f"{var}__{g1}_vs_{g2}", 'pairwise', str(tmp), uc_pyseer, str(out_fp),
                                                'completed'
                                            ))
                                            pairwise_outputs.append(out_fp)
                                        else:
                                            error_msg = "Pyseer output is empty"
                                            logger.warning(f"Pairwise Pyseer output is empty for {species}__{var}__{g1}_vs_{g2}")
                                            worker_pyseer_tracking.append(_track_pyseer_run(
                                                species, f"{var}__{g1}_vs_{g2}", 'pairwise', str(tmp), uc_pyseer, str(out_fp),
                                                'failed', error_msg
                                            ))
                                            
                                    except subprocess.CalledProcessError as e:
                                        error_msg = f"Pyseer failed with return code {e.returncode}"
                                        logger.error(f"Pairwise Pyseer failed for {species}__{var}__{g1}_vs_{g2}: {error_msg}")
                                        worker_pyseer_tracking.append(_track_pyseer_run(
                                            species, f"{var}__{g1}_vs_{g2}", 'pairwise', str(tmp), uc_pyseer, str(out_fp),
                                            'failed', error_msg
                                        ))
                                    except Exception as e:
                                        error_msg = f"Unexpected error: {e}"
                                        logger.error(f"Unexpected error running pairwise pyseer for {species}__{var}__{g1}_vs_{g2}: {e}")
                                        worker_pyseer_tracking.append(_track_pyseer_run(
                                            species, f"{var}__{g1}_vs_{g2}", 'pairwise', str(tmp), uc_pyseer, str(out_fp),
                                            'failed', error_msg
                                        ))
                        else:
                            # Track skipped due to sample size
                            skip_reason = f"Insufficient samples for Pyseer (n1={n1}, n2={n2}, min_required={args.pair_pyseer_min_n})"
                            logger.info(f"Pairwise Pyseer skipped for {species}__{var}__{g1}_vs_{g2}: {skip_reason}")
                            worker_pyseer_tracking.append(_track_pyseer_run(
                                species, f"{var}__{g1}_vs_{g2}", 'pairwise', str(tmp), uc_pyseer, str(pair_pyseer_dir / f"{species}__{var}__{g1}_vs_{g2}.pyseer.tsv"),
                                'skipped', skip_reason
                            ))
                # Save pairwise PERMANOVA results
                if pairwise_results:
                    pairwise_df = pd.DataFrame(pairwise_results)
                    pairwise_file = pair_dist_dir / f"{species}.pairwise_permanova.tsv"
                    pairwise_df.to_csv(pairwise_file, sep='\t', index=False)
                    logger.info(f"Saved pairwise PERMANOVA results: {pairwise_file}")

                # Combine per-species pairwise pyseer outputs
                if pairwise_outputs:
                    parts = []
                    for fp in pairwise_outputs:
                        try:
                            dfp = pd.read_csv(fp, sep='\t')
                            dfp['__source__'] = fp.name
                            parts.append(dfp)
                        except Exception:
                            pass
                    if parts:
                        combined = pyseer_dir / species / f"{species}.pairwise.pyseer.tsv"
                        pd.concat(parts, ignore_index=True).to_csv(combined, sep='\t', index=False)
                        logger.info(f"Saved pairwise pyseer results: {combined}")
        except Exception as e:
            logger.error(f"Pairwise testing failed for {species}: {e}", exc_info=True)
        
        # Processing completed successfully
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=['species','metadata','test','stat','pvalue','n_samples']), worker_pyseer_tracking
    
    except Exception as e:
        # Processing failed
        logger.error(f"Processing failed for species {species}: {e}")
        raise

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
    - Progress tracking: Log species processing status to progress.log
    - Force re-run: Option to ignore existing files and re-run all species
    - File verification: Check actual file existence, not just progress log
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
    ap.add_argument('--cont-max', type=float, default=10.0,  help='maximum contamination to keep (default 10.0)')
    ap.add_argument('--threads', type=int, default=32, help='total CPU budget')
    ap.add_argument('--workers', type=int, default=4, help='number of species processed in parallel')
    ap.add_argument('--perms', type=int, default=999, help='starting number of permutations for PERMANOVA/Mantel (can auto-escalate to max-perms)')
    ap.add_argument('--max-perms', type=int, default=999999, help='maximum permutations (can go beyond 1M with custom implementation, default: 999999)')
    ap.add_argument('--perm-mode', choices=['auto', 'standard', 'large', 'analytical'], default='auto',
                    help='Permutation strategy: auto=adaptive (uses analytical for n>500, large for >10k perms), '
                         'standard=scikit-bio only, large=custom 1M+ implementation, analytical=F-distribution approximation')
    ap.add_argument('--tests', choices=['exact','fast'], default='fast', help='exact=PERMANOVA/Mantel; fast=PC-based ANOVA/OLS (no permutations)')
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
    ap.add_argument('--force', action='store_true', default=True,
                    help='Force re-run all species (ignore existing files)')
    ap.add_argument('--missing-values', nargs='*', default=None,
                    help='Additional missing value indicators to treat as NaN (space-separated)')
    ap.add_argument('--phenotype', default=None,
                    help='Optional substring to select a single phenotype to test (case-insensitive). If omitted, test all.')
    args = ap.parse_args()
    
    # derive per-worker threads
    args.threads_per_worker = max(1, args.threads // max(1, args.workers))
    os.environ['OMP_NUM_THREADS'] = str(args.threads_per_worker)
    os.environ['OPENBLAS_NUM_THREADS'] = str(args.threads_per_worker)
    os.environ['MKL_NUM_THREADS'] = str(args.threads_per_worker)
    os.environ['NUMEXPR_NUM_THREADS'] = str(args.threads_per_worker)

    OUT = pathlib.Path(args.out); ensure_dir(OUT)
    work = OUT/'tmp'; ensure_dir(work)
    mash_dir = OUT/'mash_by_species'; ensure_dir(mash_dir)
    pyseer_dir = OUT/'pyseer_by_species'; ensure_dir(pyseer_dir)
    distance_assoc_dir = OUT/'distance_assoc_by_species'; ensure_dir(distance_assoc_dir)
    unitig_dir = OUT/'unitigs_by_species'; ensure_dir(unitig_dir)
    ensure_dir(OUT/'assoc')
    
    # Log run information
    logger.info(f"Output directory: {OUT}")
    if args.resume and not args.force:
        logger.info("Resume mode: Will skip completed species based on file existence")
    elif args.force:
        logger.info("Force mode: Will re-run all species")

    # enumerate FASTAs (basename == sample id)
    s2p = list_fastas(args.genomes)

    # CheckM filter (keeps all if file missing/empty)
    if args.checkm:
        pre_n = len(s2p)
        s2p = filter_by_checkm(s2p, args.checkm, args.comp_min, args.cont_max)
        post_n = len(s2p)
        if post_n == 0:
            logger.fatal(f"CheckM filter removed all samples (comp_min={args.comp_min}, cont_max={args.cont_max})")
            return
        if post_n < pre_n:
            logger.info(f"CheckM filter kept {post_n}/{pre_n} samples")

    # Load ANI mapping once
    ani = pd.read_csv(args.ani_map, sep='\t', names=['species','sample']).drop_duplicates()
    species_list = sorted(ani['species'].unique())
    
    # Handle phenotype file generation with resume logic
    phenos_dir = work / 'phenos'
    phenos_dir.mkdir(parents=True, exist_ok=True)
    
    if args.resume and not args.force:
        # Check if phenotype files are complete for all species
        
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
            # Create a temporary ANI map with only incomplete species
            temp_ani_map = work / 'temp_ani_map.tsv'
            incomplete_ani = ani[ani['species'].isin(incomplete_pheno_species)]
            incomplete_ani.to_csv(temp_ani_map, sep='\t', index=False, header=False)
            
            # Generate phenotype files for incomplete species only
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
            pf = args.phenotype.lower()
            for sp in list(phenos.keys()):
                phenos[sp] = [(v, t, p) for (v, t, p) in phenos[sp] if pf in str(v).lower()]
        
    else:
        # Force mode or no resume - regenerate all phenotype files
        if args.force:
            logger.info("Force mode: Regenerating all phenotype files")
        
        phenos = filter_metadata_per_species(
            metadata_fp=args.metadata, ani_map_fp=args.ani_map, out_dir=str(phenos_dir),
            min_n=args.min_n, max_missing_frac=args.max_missing_frac,
            min_level_n=args.min_level_n, min_unique_cont=args.min_unique_cont,
            custom_missing_values=args.missing_values,
            phenotype_filter=args.phenotype
        )

    # Build species -> sample list and determine which species to process
    sp_to_samples = _build_species_sample_map(ani, s2p, args.min_n)
    remaining_species = _get_remaining_species(sp_to_samples, args, phenos, mash_dir, pyseer_dir, distance_assoc_dir, unitig_dir)
    
    # Filter out species with no valid phenotypes
    species_with_valid_phenos = {
        sp: sams for sp, sams in remaining_species.items()
        if sp in phenos and len(phenos[sp]) > 0 and has_passing_phenotypes(sp, phenos_dir)
    }
    if len(species_with_valid_phenos) < len(remaining_species):
        skipped = len(remaining_species) - len(species_with_valid_phenos)
        logger.info(f"Skipping {skipped} species with no passing phenotypes")
    remaining_species = species_with_valid_phenos

    # Generate comprehensive summary of all input files and their status
    _generate_analysis_summary(phenos, sp_to_samples, remaining_species, args, pyseer_dir, distance_assoc_dir)

    # Generate expected Pyseer runs list
    expected_pyseer_runs = _generate_expected_pyseer_runs(phenos, pyseer_dir)
    logger.info(f"Generated {len(expected_pyseer_runs)} expected Pyseer runs")

    # Write expected runs to file for reference
    expected_runs_file = OUT / 'expected_pyseer_runs.tsv'
    if expected_pyseer_runs:
        pd.DataFrame(expected_pyseer_runs).to_csv(expected_runs_file, sep='\t', index=False)
        logger.info(f"Expected Pyseer runs written to: {expected_runs_file}")

    results = []
    pyseer_tracking = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = []
        for sp, sams in remaining_species.items():
            futs.append(ex.submit(_worker, sp, sams, s2p, args, phenos, mash_dir, pyseer_dir, distance_assoc_dir, unitig_dir, phenos_dir))
        for f in as_completed(futs):
            df, worker_tracking = f.result()
            if not df.empty:
                results.append(df)
            pyseer_tracking.extend(worker_tracking)

    # Collect results from previously completed species if resuming
    if args.resume and not args.force:
        # Find completed species based on file existence
        actually_complete = set()
        for sp in sp_to_samples.keys():
            if is_species_complete(sp, phenos, mash_dir, pyseer_dir, distance_assoc_dir, unitig_dir):
                actually_complete.add(sp)

        existing_results = _collect_existing_results(actually_complete, phenos, distance_assoc_dir)
        results.extend(existing_results)

    if results:
        master = OUT/'assoc'/'distance_associations_all_species.tsv'
        combined_df = pd.concat(results, ignore_index=True)
        combined_df.to_csv(master, sep='\t', index=False)

        # Apply FDR correction
        fdr_master = OUT/'assoc'/'distance_associations_all_species.fdr.tsv'
        add_bh(str(master), str(fdr_master))
        logger.info(f"Final distance association results written to {master}")
        logger.info(f"FDR-corrected results written to {fdr_master}")
    else:
        logger.info("No results to write")

    # Write Pyseer tracking report
    if pyseer_tracking:
        tracking_report_file = OUT / 'pyseer_tracking_report.tsv'
        _write_pyseer_tracking_report(pyseer_tracking, tracking_report_file)
    else:
        logger.warning("No Pyseer tracking data collected")

    # Always summarize pyseer results
    logger.info("Summarizing pyseer results...")
    summarize_pyseer(
        indir=str(pyseer_dir),
        out=str(OUT/'assoc'/'pyseer_summary.tsv'),
        alpha=0.05,
        pattern="**/*.pyseer.tsv"
    )

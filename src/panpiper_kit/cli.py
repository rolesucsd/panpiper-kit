import argparse
import pathlib
import pandas as pd
import os
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple, Any

from .files import list_fastas, ensure_dir, run
from .filter import filter_metadata_per_species, filter_by_checkm
from .mash import mash_within_species
from .assoc import distance_assoc_one, fast_distance_tests
from .gwas import ensure_unitigs
from .fdr import add_bh
from .external import check_perl_dependencies

# ----- progress tracking -----

def is_species_complete(
    species: str, 
    phenos: Dict[str, List[Tuple[str, str, str]]], 
    mash_dir: pathlib.Path, 
    assoc_dir: pathlib.Path, 
    unitig_dir: pathlib.Path
) -> bool:
    """
    Check if a species has been fully processed by verifying all required output files exist.
    
    Args:
        species: Species name to check
        phenos: Dictionary of phenotype information per species
        mash_dir: Directory for Mash output files
        assoc_dir: Directory for association test results
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
    
    # Check association files for all phenotypes
    for var, typ, _ in phenos.get(species, []):
        # Check distance association file
        dist_file = assoc_dir / f'{species}__{var}.dist_assoc.tsv'
        if not dist_file.exists():
            return False
        
        # Check GWAS files for binary/continuous phenotypes
        if typ in ('binary', 'continuous'):
            gwas_file = assoc_dir / f'{species}__{var}.pyseer.fdr.tsv'
            if not gwas_file.exists():
                return False
    
    return True


def log_progress(species: str, status: str, progress_file: pathlib.Path) -> None:
    """
    Log species processing progress to a file.
    
    Args:
        species: Species name
        status: Processing status ('started', 'completed', 'failed')
        progress_file: Path to progress log file
    """
    import datetime
    timestamp = datetime.datetime.now().isoformat()
    with open(progress_file, 'a') as f:
        f.write(f"{timestamp}\t{species}\t{status}\n")


def load_completed_species(progress_file: pathlib.Path) -> set:
    """
    Load list of completed species from progress file.
    
    Args:
        progress_file: Path to progress log file
        
    Returns:
        Set of species names that have been completed
    """
    completed = set()
    if not progress_file.exists():
        return completed
    
    with open(progress_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3 and parts[2] == 'completed':
                completed.add(parts[1])
    
    return completed

# ----- worker -----

def _worker(
    species: str, 
    sams: List[str], 
    s2p: Dict[str, str], 
    args: Any, 
    phenos: Dict[str, List[Tuple[str, str, str]]], 
    mash_dir: pathlib.Path, 
    assoc_dir: pathlib.Path, 
    unitig_dir: pathlib.Path,
    progress_file: pathlib.Path
) -> pd.DataFrame:
    """
    Worker function for parallel processing of species-specific analyses.
    
    Performs Mash distance calculation, unitig calling, and association tests
    for a single species in parallel.
    
    Args:
        species: Species name to process
        sams: List of sample names for this species
        s2p: Dictionary mapping sample names to file paths
        args: Command line arguments object
        phenos: Dictionary of phenotype information per species
        mash_dir: Directory for Mash output files
        assoc_dir: Directory for association test results
        unitig_dir: Directory for unitig files
        progress_file: Path to progress log file
        
    Returns:
        DataFrame containing association test results for this species
    """
    # Log start of processing
    log_progress(species, 'started', progress_file)
    
    try:
        # assign this worker its own thread budget
        t = max(1, args.threads_per_worker)
        # build paths
        paths = [s2p[s] for s in sams if s in s2p]
        sp_out = mash_dir/species
        sp_out.mkdir(parents=True, exist_ok=True)
        
        # Mash within species
        mash_tsv = mash_within_species(paths, str(sp_out), k=args.mash_k, s=args.mash_s, threads=t)
        # refs for unitigs (once)
        ref_txt = sp_out/'refs.txt'
        if not ref_txt.exists():
            with open(ref_txt,'w') as fh: fh.write('\n'.join(paths))
        uc_pyseer = ensure_unitigs(str(ref_txt), str(unitig_dir/species), kmer=args.kmer, threads=t)

        rows = []
        for (var, typ, pheno_tsv) in phenos.get(species, []):
            # distance test
            if args.tests == 'exact':
                df = distance_assoc_one(str(mash_tsv), pheno_tsv, typ, args.perms)
            else:
                df = fast_distance_tests(str(mash_tsv), pheno_tsv, typ, max_axes=args.max_axes)
            df['species'] = species
            df['metadata'] = var
            out_d = assoc_dir/f'{species}__{var}.dist_assoc.tsv'
            df.to_csv(out_d, sep='\t', index=False)
            rows.append(df)

            # GWAS for binary/continuous
            if typ in ('binary','continuous'):
                out_fp = assoc_dir/f'{species}__{var}.pyseer.tsv'
                cmd = (
                    f'pyseer --phenotypes "{pheno_tsv}" '
                    f'--kmers "{uc_pyseer}" --uncompressed '
                    f'--min-af {args.maf} --cpu {t} --no-distances > "{out_fp}"'
                )
                run(['bash','-lc', cmd])
                add_bh(str(out_fp), str(assoc_dir/f'{species}__{var}.pyseer.fdr.tsv'))
        
        # Log successful completion
        log_progress(species, 'completed', progress_file)
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=['species','metadata','test','stat','pvalue','n_samples'])
    
    except Exception as e:
        # Log failure
        log_progress(species, f'failed: {str(e)}', progress_file)
        raise

def main() -> None:
    """
    Main CLI entry point for panpiper-kit.

    Performs ANI-split Mash lineage tests and per-species unitig GWAS with optional CheckM filtering
    and parallel processing across species.
    
    Features:
    - Resume capability: Skip previously completed species on restart
    - Progress tracking: Log species processing status to progress.log
    - Force re-run: Option to ignore existing files and re-run all species
    - File verification: Check actual file existence, not just progress log
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
    ap.add_argument('--perms', type=int, default=999, help='number of permutations for PERMANOVA/Mantel')
    ap.add_argument('--tests', choices=['exact','fast'], default='exact', help='exact=PERMANOVA/Mantel; fast=PC-based ANOVA/OLS (no permutations)')
    ap.add_argument('--max-axes', type=int, default=10, help='PC axes used in fast mode')
    ap.add_argument('--mash-k', type=int, default=18, help='Mash k-mer size')
    ap.add_argument('--mash-s', type=int, default=10000, help='Mash sketch size')
    ap.add_argument('--kmer', type=int, default=31, help='unitig-caller k-mer size')
    ap.add_argument('--maf', type=float, default=0.05, help='pyseer min allele freq for unitigs')
    ap.add_argument('--min-n', type=int, default=30, help='minimum usable samples per species')
    ap.add_argument('--max-missing-frac', type=float, default=0.2, help='max fraction of missing values allowed for a phenotype')
    ap.add_argument('--min-level-n', type=int, default=3, help='min samples required per category level')
    ap.add_argument('--min-unique-cont', type=int, default=3, help='min unique values required for continuous phenotypes')
    ap.add_argument('--resume', action='store_true', default=True, 
                    help='Resume from previous run (skip completed species)')
    ap.add_argument('--force', action='store_true', default=False,
                    help='Force re-run all species (ignore existing files)')
    args = ap.parse_args()

    # Check for common dependency issues
    check_perl_dependencies()
    
    # derive per-worker threads
    args.threads_per_worker = max(1, args.threads // max(1, args.workers))
    os.environ['OMP_NUM_THREADS'] = str(args.threads_per_worker)
    os.environ['OPENBLAS_NUM_THREADS'] = str(args.threads_per_worker)
    os.environ['MKL_NUM_THREADS'] = str(args.threads_per_worker)
    os.environ['NUMEXPR_NUM_THREADS'] = str(args.threads_per_worker)

    OUT = pathlib.Path(args.out); ensure_dir(OUT)
    work = OUT/'tmp'; ensure_dir(work)
    mash_dir = OUT/'mash_by_species'; ensure_dir(mash_dir)
    assoc_dir = OUT/'assoc_by_species'; ensure_dir(assoc_dir)
    unitig_dir = OUT/'unitigs_by_species'; ensure_dir(unitig_dir)
    ensure_dir(OUT/'assoc')
    
    # Progress tracking
    progress_file = OUT / 'progress.log'
    
    # Print run information
    print(f"[info] Output directory: {OUT}")
    print(f"[info] Progress log: {progress_file}")
    if args.resume and not args.force:
        print(f"[info] Resume mode: Will skip completed species")
    elif args.force:
        print(f"[info] Force mode: Will re-run all species")

    # enumerate FASTAs (basename == sample id)
    s2p = list_fastas(args.genomes)

    # CheckM filter (keeps all if file missing/empty)
    if args.checkm:
        pre_n = len(s2p)
        s2p = filter_by_checkm(s2p, args.checkm, args.comp_min, args.cont_max)
        post_n = len(s2p)
        if post_n == 0:
            print(f"[fatal] CheckM filter removed all samples (comp_min={args.comp_min}, cont_max={args.cont_max}).", flush=True)
            return
        if post_n < pre_n:
            print(f"[note] CheckM filter kept {post_n}/{pre_n} samples.", flush=True)

    # filter metadata per species (uses only remaining samples downstream)
    phenos = filter_metadata_per_species(
        metadata_fp=args.metadata, ani_map_fp=args.ani_map, out_dir=str(work/'phenos'),
        min_n=args.min_n, max_missing_frac=args.max_missing_frac,
        min_level_n=args.min_level_n, min_unique_cont=args.min_unique_cont
    )

    # build species -> sample list and run in parallel
    ani = pd.read_csv(args.ani_map, sep='\t', names=['species','sample']).drop_duplicates()
    species_list = sorted(ani['species'].unique())
    sp_to_samples = {sp: [s for s in ani.loc[ani['species']==sp, 'sample'] if s in s2p] for sp in species_list}

    # Handle resume logic
    if args.resume and not args.force:
        completed_species = load_completed_species(progress_file)
        print(f"[info] Found {len(completed_species)} previously completed species in progress log")
        
        # Check which species are actually complete (file-based verification)
        actually_complete = set()
        for sp in completed_species:
            if is_species_complete(sp, phenos, mash_dir, assoc_dir, unitig_dir):
                actually_complete.add(sp)
            else:
                print(f"[info] Species {sp} marked as complete but files missing, will re-run")
        
        # Filter out completed species
        remaining_species = {sp: sams for sp, sams in sp_to_samples.items() 
                           if sp not in actually_complete and len(sams) >= args.min_n}
        print(f"[info] Will process {len(remaining_species)} remaining species")
    else:
        remaining_species = {sp: sams for sp, sams in sp_to_samples.items() 
                           if len(sams) >= args.min_n}
        if args.force:
            print(f"[info] Force mode: will re-run all {len(remaining_species)} species")

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = []
        for sp, sams in remaining_species.items():
            futs.append(ex.submit(_worker, sp, sams, s2p, args, phenos, mash_dir, assoc_dir, unitig_dir, progress_file))
        for f in as_completed(futs):
            df = f.result()
            if not df.empty:
                results.append(df)

    # Collect results from previously completed species if resuming
    if args.resume and not args.force:
        completed_species = load_completed_species(progress_file)
        actually_complete = {sp for sp in completed_species 
                           if is_species_complete(sp, phenos, mash_dir, assoc_dir, unitig_dir)}
        
        # Load existing results from completed species
        for sp in actually_complete:
            # Find existing association files for this species
            for var, typ, _ in phenos.get(sp, []):
                dist_file = assoc_dir / f'{sp}__{var}.dist_assoc.tsv'
                if dist_file.exists():
                    try:
                        df = pd.read_csv(dist_file, sep='\t')
                        results.append(df)
                    except Exception as e:
                        print(f"[warning] Could not load existing results for {sp}__{var}: {e}")

    if results:
        master = OUT/'assoc'/'mash_lineage_assoc_by_species.tsv'
        pd.concat(results, ignore_index=True).to_csv(master, sep='\t', index=False)
        add_bh(str(master), str(OUT/'assoc'/'mash_lineage_assoc_by_species.fdr.tsv'))
        print(f"[info] Final results written to {master}")
    else:
        print("[info] No results to write")

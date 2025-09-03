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
    
    # Find all list.tsv files
    for list_file in phenos_dir.glob("*.list.tsv"):
        species = list_file.stem  # Remove .list.tsv extension
        
        try:
            df = pd.read_csv(list_file, sep='\t')
            species_phenos = []
            
            for _, row in df.iterrows():
                species_phenos.append((row['variable'], row['type'], row['pheno_tsv']))
            
            phenos[species] = species_phenos
        except Exception as e:
            print(f"[warning] Could not load phenotype manifest for {species}: {e}")
    
    return phenos

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
    print(f"[DEBUG] ===== Starting processing for species: {species} =====")
    print(f"[DEBUG] Available samples in s2p: {len(s2p)}")
    print(f"[DEBUG] Available phenotypes: {len(phenos)}")
    if species in phenos:
        print(f"[DEBUG] Phenotypes for {species}: {len(phenos[species])}")
        for i, (var, typ, path) in enumerate(phenos[species]):
            print(f"  {i+1}. {var} ({typ}) -> {path}")
    else:
        print(f"[DEBUG] WARNING: No phenotypes found for species {species}")
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
        
        print(f"[DEBUG] Processing species: {species}")
        print(f"[DEBUG] Found {len(paths)} FASTA files for this species")
        print(f"[DEBUG] Sample paths: {paths[:3]}...")  # Show first 3 paths
        
        # Mash within species
        print(f"[DEBUG] Computing Mash distances (k={args.mash_k}, s={args.mash_s})")
        mash_tsv = mash_within_species(paths, str(sp_out), k=args.mash_k, s=args.mash_s, threads=t)
        print(f"[DEBUG] Mash distances computed: {mash_tsv}")
        
        # refs for unitigs (once)
        ref_txt = sp_out/'refs.txt'
        if not ref_txt.exists():
            print(f"[DEBUG] Creating refs.txt with {len(paths)} paths")
            with open(ref_txt,'w') as fh: fh.write('\n'.join(paths))
        else:
            print(f"[DEBUG] Using existing refs.txt")
        
        print(f"[DEBUG] Generating unitigs (kmer={args.kmer})")
        uc_pyseer = ensure_unitigs(str(ref_txt), str(unitig_dir/species), kmer=args.kmer, threads=t)
        print(f"[DEBUG] Unitigs generated: {uc_pyseer}")

        rows = []
        phenotypes = phenos.get(species, [])
        print(f"[DEBUG] Found {len(phenotypes)} phenotypes for {species}")
        
        for i, (var, typ, pheno_tsv) in enumerate(phenotypes):
            print(f"[DEBUG] Processing phenotype {i+1}/{len(phenotypes)}: {var} ({typ})")
            print(f"[DEBUG] Phenotype file: {pheno_tsv}")
            
            # Check if phenotype file exists
            if not os.path.exists(pheno_tsv):
                print(f"[DEBUG] ERROR: Phenotype file does not exist: {pheno_tsv}")
                continue
            
            # Check phenotype file content
            try:
                pheno_df = pd.read_csv(pheno_tsv, sep='\t')
                print(f"[DEBUG] Phenotype file shape: {pheno_df.shape}")
                print(f"[DEBUG] Phenotype columns: {list(pheno_df.columns)}")
                if 'phenotype' in pheno_df.columns:
                    print(f"[DEBUG] Phenotype value counts: {pheno_df['phenotype'].value_counts().head()}")
            except Exception as e:
                print(f"[DEBUG] ERROR: Could not read phenotype file: {e}")
                continue
            
            # distance test
            print(f"[DEBUG] Running distance association test ({args.tests} mode)")
            if args.tests == 'exact':
                df = distance_assoc_one(str(mash_tsv), pheno_tsv, typ, args.perms)
            else:
                df = fast_distance_tests(str(mash_tsv), pheno_tsv, typ, max_axes=args.max_axes)
            
            print(f"[DEBUG] Distance test result: {df.iloc[0].to_dict()}")
            df['species'] = species
            df['metadata'] = var
            out_d = assoc_dir/f'{species}__{var}.dist_assoc.tsv'
            df.to_csv(out_d, sep='\t', index=False)
            print(f"[DEBUG] Distance association saved to: {out_d}")
            rows.append(df)

            # GWAS for binary/continuous
            if typ in ('binary','continuous'):
                print(f"[DEBUG] Starting pyseer GWAS for {species}__{var} ({typ})")
                print(f"[DEBUG] Input files:")
                print(f"  phenotypes: {pheno_tsv}")
                print(f"  kmers: {uc_pyseer}")
                print(f"  maf: {args.maf}, threads: {t}")
                
                out_fp = assoc_dir/f'{species}__{var}.pyseer.tsv'
                print(f"[DEBUG] Output file: {out_fp}")
                
                # Check if input files exist
                if not os.path.exists(pheno_tsv):
                    print(f"[DEBUG] ERROR: Phenotype file does not exist: {pheno_tsv}")
                    continue
                if not os.path.exists(uc_pyseer):
                    print(f"[DEBUG] ERROR: Unitig file does not exist: {uc_pyseer}")
                    continue
                
                # Check file sizes
                pheno_size = os.path.getsize(pheno_tsv)
                unitig_size = os.path.getsize(uc_pyseer)
                print(f"[DEBUG] File sizes: phenotype={pheno_size} bytes, unitigs={unitig_size} bytes")
                
                # (no shell; capture stdout directly)
                import subprocess
                try:
                    with open(out_fp, 'w') as fh:
                        cmd = ['pyseer',
                               '--phenotypes', pheno_tsv,
                               '--kmers', uc_pyseer, '--uncompressed',
                               '--min-af', str(args.maf), '--cpu', str(t), '--no-distances']
                        print(f"[DEBUG] Running command: {' '.join(cmd)}")
                        subprocess.check_call(cmd, stdout=fh)
                    
                    # Check if output was created and has content
                    if os.path.exists(out_fp):
                        output_size = os.path.getsize(out_fp)
                        print(f"[DEBUG] Pyseer completed successfully, output size: {output_size} bytes")
                        if output_size > 0:
                            # Show first few lines of output
                            with open(out_fp, 'r') as fh:
                                lines = fh.readlines()[:5]
                                print(f"[DEBUG] Output preview (first {len(lines)} lines):")
                                for i, line in enumerate(lines):
                                    print(f"  {i+1}: {line.strip()}")
                        else:
                            print(f"[DEBUG] WARNING: Output file is empty!")
                    else:
                        print(f"[DEBUG] ERROR: Output file was not created!")
                        
                except subprocess.CalledProcessError as e:
                    print(f"[DEBUG] ERROR: Pyseer failed with return code {e.returncode}")
                    print(f"[DEBUG] Command that failed: {' '.join(cmd)}")
                    continue
                except Exception as e:
                    print(f"[DEBUG] ERROR: Unexpected error running pyseer: {e}")
                    continue
                
                print(f"[DEBUG] Adding FDR correction")
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
    ap.add_argument('--missing-values', nargs='*', default=None,
                    help='Additional missing value indicators to treat as NaN (space-separated)')
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
            print(f"[info] Found complete phenotype files for {len(complete_pheno_species)} species")
        
        if incomplete_pheno_species:
            print(f"[info] Need to regenerate phenotype files for {len(incomplete_pheno_species)} species")
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
        
    else:
        # Force mode or no resume - regenerate all phenotype files
        if args.force:
            print(f"[info] Force mode: Regenerating all phenotype files")
        
        phenos = filter_metadata_per_species(
            metadata_fp=args.metadata, ani_map_fp=args.ani_map, out_dir=str(phenos_dir),
            min_n=args.min_n, max_missing_frac=args.max_missing_frac,
            min_level_n=args.min_level_n, min_unique_cont=args.min_unique_cont,
            custom_missing_values=args.missing_values
        )

    # build species -> sample list and run in parallel
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

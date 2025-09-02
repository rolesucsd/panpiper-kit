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

# ----- worker -----

def _worker(
    species: str, 
    sams: List[str], 
    s2p: Dict[str, str], 
    args: Any, 
    phenos: Dict[str, List[Tuple[str, str, str]]], 
    mash_dir: pathlib.Path, 
    assoc_dir: pathlib.Path, 
    unitig_dir: pathlib.Path
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
        
    Returns:
        DataFrame containing association test results for this species
    """
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
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=['species','metadata','test','stat','pvalue','n_samples'])

def main() -> None:
    """
    Main CLI entry point for panpiper-kit.

    Performs ANI-split Mash lineage tests and per-species unitig GWAS with optional CheckM filtering
    and parallel processing across species.
    """
    ap = argparse.ArgumentParser(
        description='ANI-split Mash lineage tests + per-species unitig GWAS (parallel)'
    )
    ap.add_argument('--genomes', required=True, help='directory of FASTA files')
    ap.add_argument('--metadata', required=True)
    ap.add_argument('--ani-map', required=True, help='TSV with cluster/species<TAB>sample (basename of FASTA file)')
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
    assoc_dir = OUT/'assoc_by_species'; ensure_dir(assoc_dir)
    unitig_dir = OUT/'unitigs_by_species'; ensure_dir(unitig_dir)
    ensure_dir(OUT/'assoc')

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

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = []
        for sp, sams in sp_to_samples.items():
            if len(sams) < args.min_n:
                continue
            futs.append(ex.submit(_worker, sp, sams, s2p, args, phenos, mash_dir, assoc_dir, unitig_dir))
        for f in as_completed(futs):
            df = f.result()
            if not df.empty:
                results.append(df)

    if results:
        master = OUT/'assoc'/'mash_lineage_assoc_by_species.tsv'
        pd.concat(results, ignore_index=True).to_csv(master, sep='\t', index=False)
        add_bh(str(master), str(OUT/'assoc'/'mash_lineage_assoc_by_species.fdr.tsv'))

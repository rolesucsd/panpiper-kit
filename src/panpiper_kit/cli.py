import argparse, pathlib, pandas as pd
from .files import list_fastas, ensure_dir, run
from .filter_pheno import filter_metadata_per_species
from .mash import mash_within_species
from .assoc import distance_assoc_one
from .gwas import ensure_unitigs
from .fdr import add_bh

def main():
    ap = argparse.ArgumentParser(description='ANI-split Mash lineage tests + per-species unitig GWAS')
    ap.add_argument('--genomes', required=True)
    ap.add_argument('--metadata', required=True)
    ap.add_argument('--ani-map', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--threads', type=int, default=16)
    ap.add_argument('--perms', type=int, default=999)
    ap.add_argument('--mash-k', type=int, default=18)
    ap.add_argument('--mash-s', type=int, default=5000)
    ap.add_argument('--kmer', type=int, default=31)
    ap.add_argument('--min-n', type=int, default=6)
    ap.add_argument('--max-missing-frac', type=float, default=0.2)
    ap.add_argument('--min-level-n', type=int, default=3)
    ap.add_argument('--min-unique-cont', type=int, default=6)
    args = ap.parse_args()

    OUT = pathlib.Path(args.out); ensure_dir(OUT)
    work = OUT/'work'; ensure_dir(work)
    mash_dir = OUT/'mash_by_species'; ensure_dir(mash_dir)
    assoc_dir = OUT/'assoc_by_species'; ensure_dir(assoc_dir)
    unitig_dir = OUT/'unitigs_by_species'; ensure_dir(unitig_dir)
    ensure_dir(OUT/'assoc')

    s2p = list_fastas(args.genomes)

    phenos = filter_metadata_per_species(
        metadata_fp=args.metadata, ani_map_fp=args.ani_map, out_dir=str(work/'phenos'),
        min_n=args.min_n, max_missing_frac=args.max_missing_frac,
        min_level_n=args.min_level_n, min_unique_cont=args.min_unique_cont
    )

    ani = pd.read_csv(args.ani_map, sep='\t', names=['sample','species']).drop_duplicates()

    all_rows = []
    for species in sorted(ani['species'].unique()):
        sams = list(ani.loc[ani['species']==species,'sample'])
        paths = [s2p[s] for s in sams if s in s2p]
        if len(paths) < args.min_n: 
            continue
        sp_out = mash_dir/species
        mash_tsv = mash_within_species(paths, str(sp_out), k=args.mash_k, s=args.mash_s, threads=args.threads)
        ref_txt = sp_out/'refs.txt'
        if not ref_txt.exists():
            with open(ref_txt,'w') as fh: fh.write('\n'.join(paths))
        uc_pyseer = ensure_unitigs(str(ref_txt), str(unitig_dir/species), kmer=args.kmer, threads=args.threads)

        for (var, typ, pheno_tsv) in phenos.get(species, []):
            df = distance_assoc_one(str(mash_tsv), pheno_tsv, typ, args.perms)
            df.to_csv(assoc_dir/f'{species}__{var}.dist_assoc.tsv', sep='\t', index=False)
            all_rows.append(df)
            if typ in ('binary','continuous'):
                out_fp = assoc_dir/f'{species}__{var}.pyseer.tsv'
                cmd = f'pyseer --phenotypes "{pheno_tsv}" --kmers "{uc_pyseer}" --uncompressed --min-af 0.01 --cpu {args.threads} --no-distances > "{out_fp}"'
                run(['bash','-lc', cmd])
                add_bh(str(out_fp), str(assoc_dir/f'{species}__{var}.pyseer.fdr.tsv'))

    if all_rows:
        master = OUT/'assoc'/'mash_lineage_assoc_by_species.tsv'
        import pandas as pd
        pd.concat(all_rows, ignore_index=True).to_csv(master, sep='\t', index=False)
        add_bh(str(master), str(OUT/'assoc'/'mash_lineage_assoc_by_species.fdr.tsv'))

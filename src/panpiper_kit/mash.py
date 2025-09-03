import os
import pathlib
from typing import List

import pandas as pd
import numpy as np
from .files import run, ensure_dir


def _square_from_pairs(pairs_tsv: str, out_tsv: str) -> None:
    """
    Convert Mash pairwise distance output to square distance matrix.
    
    Args:
        pairs_tsv: Input file with pairwise Mash distances
        out_tsv: Output file for square distance matrix
    """
    df = pd.read_csv(pairs_tsv, sep='\t', header=None, names=['q','r','dist','p','shared'])
    mat = df.pivot_table(index='q', columns='r', values='dist')
    idx = sorted(set(mat.index)|set(mat.columns))
    mat = mat.reindex(index=idx, columns=idx).fillna(0.0)
    for i in idx: mat.loc[i,i] = 0.0
    mat.index=[os.path.basename(i) for i in mat.index]
    mat.columns=[os.path.basename(i) for i in mat.columns]
    mat.to_csv(out_tsv, sep='\t')

def mash_within_species(fasta_paths: List[str], out_dir: str, k: int, s: int, threads: int) -> str:
    """
    Calculate Mash distances within a species using provided FASTA files.
    
    Args:
        fasta_paths: List of paths to FASTA files for this species
        out_dir: Output directory for Mash files
        k: K-mer size for Mash sketching
        s: Sketch size for Mash
        threads: Number of threads to use
        
    Returns:
        Path to the generated Mash distance matrix TSV file
    """
    ensure_dir(out_dir)
    ref_list = os.path.join(out_dir, 'refs.txt')
    with open(ref_list,'w') as fh:
        fh.write('\n'.join(fasta_paths))
    msh = os.path.join(out_dir, 'genomes.msh')
    run(['mash','sketch','-k',str(k),'-s',str(s),'-p',str(threads),'-l',ref_list,'-o',msh[:-4]],
        log=os.path.join(out_dir,'mash_sketch.log'))
    out = os.path.join(out_dir,'mash.tsv')
    # try square_mash
    try:
        run(['bash','--noprofile','--norc','-c',
         f'mash dist -p {threads} {msh} {msh} | square_mash > {out}'])
    except Exception:
        pairs = os.path.join(out_dir,'mash_pairs.tsv')
        run(['mash','dist','-p',str(threads),msh,msh], log=pairs)
        
        _square_from_pairs(pairs, out)
    D = pd.read_csv(out, sep='\t', index_col=0)
    D = (D + D.T)/2; import numpy as _np; _np.fill_diagonal(D.values, 0.0)
    D.to_csv(out, sep='\t')
    return out

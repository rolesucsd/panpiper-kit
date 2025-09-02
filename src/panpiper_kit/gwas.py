import os
import pandas as pd
from .files import run, ensure_dir


def ensure_unitigs(refs_txt: str, out_dir: str, kmer: int, threads: int) -> str:
    """
    Ensure unitig files are generated for GWAS analysis.
    
    Runs unitig-caller to generate unitig presence/absence matrix in pyseer format
    if the output file doesn't already exist.
    
    Args:
        refs_txt: Path to text file containing reference genome paths
        out_dir: Output directory for unitig files
        kmer: K-mer size for unitig calling
        threads: Number of threads to use
        
    Returns:
        Path to the generated pyseer-format unitig file
    """
    ensure_dir(out_dir)
    out_base = os.path.join(out_dir, 'uc')
    pyseer_path = out_base + '.pyseer'
    if not os.path.exists(pyseer_path):
        run(['unitig-caller','--call','--refs',refs_txt,'--kmer',str(kmer),
             '--threads',str(threads),'--pyseer','--out',out_base],
            log=os.path.join(out_dir,'unitig_caller.log'))
    return pyseer_path

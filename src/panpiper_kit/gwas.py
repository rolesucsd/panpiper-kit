import os, pandas as pd
from .files import run, ensure_dir

def ensure_unitigs(refs_txt: str, out_dir: str, kmer: int, threads: int) -> str:
    ensure_dir(out_dir)
    out_base = os.path.join(out_dir, 'uc')
    pyseer_path = out_base + '.pyseer'
    if not os.path.exists(pyseer_path):
        run(['unitig-caller','--call','--refs',refs_txt,'--kmer',str(kmer),
             '--threads',str(threads),'--pyseer','--out',out_base],
            log=os.path.join(out_dir,'unitig_caller.log'))
    return pyseer_path

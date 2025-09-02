import os, re, pathlib, subprocess, shutil
from typing import List, Dict

FA_RE = re.compile(r'\.(fa|fna|fasta)(\.gz)?$', re.I)

def list_fastas(genomes_dir: str) -> Dict[str, str]:
    pdir = pathlib.Path(genomes_dir)
    if not pdir.exists():
        raise RuntimeError(f"Genomes dir not found: {genomes_dir}")
    m = {}
    for p in pdir.iterdir():
        if p.is_file() and FA_RE.search(p.name):
            b = FA_RE.sub('', p.name)
            if b in m:
                raise RuntimeError(f"Duplicate sample basename '{b}' from {m[b]} and {p}")
            m[b] = str(p.resolve())
    if not m:
        raise RuntimeError("No FASTA files found.")
    return dict(sorted(m.items()))

def ensure_dir(p: str):
    pathlib.Path(p).mkdir(parents=True, exist_ok=True)

def run(cmd: List[str], log: str|None=None):
    if log:
        with open(log, 'w') as fh:
            subprocess.check_call(cmd, stdout=fh, stderr=subprocess.STDOUT)
    else:
        subprocess.check_call(cmd)

import os
import re
import pathlib
import subprocess
import shutil
from typing import List, Dict, Optional

FA_RE = re.compile(r'\.(fa|fna|fasta)(\.gz)?$', re.I)


def list_fastas(genomes_dir: str) -> Dict[str, str]:
    """
    List all FASTA files in a directory and map basenames to full paths.
    
    Args:
        genomes_dir: Directory containing FASTA files
        
    Returns:
        Dictionary mapping sample basenames to full file paths
        
    Raises:
        RuntimeError: If directory doesn't exist, no FASTA files found, or duplicate basenames
    """
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

def ensure_dir(p: str) -> None:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        p: Directory path to create
    """
    pathlib.Path(p).mkdir(parents=True, exist_ok=True)


def run(cmd: List[str], log: Optional[str] = None) -> None:
    """
    Run a command with optional logging to file.
    
    Args:
        cmd: Command to run as list of strings
        log: Optional log file path to write stdout/stderr
        
    Raises:
        subprocess.CalledProcessError: If command fails
    """
    if log:
        with open(log, 'w') as fh:
            subprocess.check_call(cmd, stdout=fh, stderr=subprocess.STDOUT)
    else:
        subprocess.check_call(cmd)

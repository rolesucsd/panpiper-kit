import shutil
from typing import Optional


def require(cmd: str) -> str:
    """
    Check if a required executable is available in PATH.
    
    Args:
        cmd: Command/executable name to check for
        
    Returns:
        Full path to the executable if found
        
    Raises:
        RuntimeError: If the executable is not found in PATH
    """
    path = shutil.which(cmd)
    if path is None:
        raise RuntimeError(f"Required executable '{cmd}' not found in PATH")
    return path
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


def check_perl_dependencies() -> None:
    """
    Check for common Perl dependency issues and provide helpful error messages.
    
    This function can be called to diagnose Perl-related issues with external tools.
    """
    import subprocess
    import sys
    
    try:
        # Test if perl is available
        subprocess.run(['perl', '--version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Warning: Perl not found. Some tools (mash, unitig-caller) may require Perl.")
        return
    
    # Check for local::lib module
    try:
        subprocess.run(['perl', '-Mlocal::lib', '-e', '1'], capture_output=True, check=True)
    except subprocess.CalledProcessError:
        print("Warning: Perl local::lib module not found.")
        print("To fix this, try one of the following:")
        print("1. Install via conda: conda install -c conda-forge perl-local-lib")
        print("2. Install via cpan: cpan local::lib")
        print("3. Or ignore this warning if tools work without it")


# check but don't import globally to avoid side effects; use require() at runtime.

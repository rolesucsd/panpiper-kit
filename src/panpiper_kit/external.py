import shutil

def require(cmd: str) -> str:
    path = shutil.which(cmd)
    if path is None:
        raise RuntimeError(f"Required executable '{cmd}' not found in PATH")
    return path

# check but don't import globally to avoid side effects; use require() at runtime.

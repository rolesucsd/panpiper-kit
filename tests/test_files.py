import os
import sys
import pathlib
import tempfile
import pandas as pd
from unittest.mock import patch, MagicMock

# Ensure project src/ is on sys.path for imports when running tests locally
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from panpiper_kit.files import list_fastas, ensure_dir, run


def test_list_fastas():
    """Test list_fastas function with various file extensions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test FASTA files
        fasta_files = [
            'sample1.fasta',
            'sample2.fa',
            'sample3.fna',
            'sample4.fa.gz',
            'not_fasta.txt',
            'sample5.fasta.gz'
        ]
        
        for fname in fasta_files:
            (pathlib.Path(tmpdir) / fname).touch()
        
        result = list_fastas(tmpdir)
        
        # Should find all FASTA files (including .gz)
        # Use resolve() to get absolute paths for comparison
        expected = {
            'sample1': str((pathlib.Path(tmpdir) / 'sample1.fasta').resolve()),
            'sample2': str((pathlib.Path(tmpdir) / 'sample2.fa').resolve()),
            'sample3': str((pathlib.Path(tmpdir) / 'sample3.fna').resolve()),
            'sample4': str((pathlib.Path(tmpdir) / 'sample4.fa.gz').resolve()),
            'sample5': str((pathlib.Path(tmpdir) / 'sample5.fasta.gz').resolve())
        }
        
        assert result == expected
        assert 'not_fasta.txt' not in result


def test_list_fastas_empty_directory():
    """Test list_fastas with empty directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            result = list_fastas(tmpdir)
            assert False, "Should have raised RuntimeError"
        except RuntimeError as e:
            assert "No FASTA files found" in str(e)


def test_list_fastas_nonexistent_directory():
    """Test list_fastas with non-existent directory."""
    try:
        result = list_fastas('/nonexistent/directory')
        assert False, "Should have raised RuntimeError"
    except RuntimeError as e:
        assert "Genomes dir not found" in str(e)


def test_ensure_dir():
    """Test ensure_dir creates directories."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_dir = pathlib.Path(tmpdir) / 'test' / 'nested' / 'dir'
        ensure_dir(str(test_dir))
        assert test_dir.exists()
        assert test_dir.is_dir()


def test_ensure_dir_existing():
    """Test ensure_dir with existing directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_dir = pathlib.Path(tmpdir) / 'existing'
        test_dir.mkdir()
        ensure_dir(str(test_dir))  # Should not raise error
        assert test_dir.exists()


@patch('subprocess.check_call')
def test_run_success(mock_check_call):
    """Test run function with successful command."""
    mock_check_call.return_value = None
    
    cmd = ['echo', 'test']
    run(cmd)
    
    mock_check_call.assert_called_once_with(cmd)


@patch('subprocess.check_call')
def test_run_with_log(mock_check_call):
    """Test run function with log file."""
    mock_check_call.return_value = None
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as log_file:
        log_path = log_file.name
    
    try:
        cmd = ['echo', 'test']
        run(cmd, log=log_path)
        
        mock_check_call.assert_called_once()
        # Check that log file was created
        assert pathlib.Path(log_path).exists()
    finally:
        os.unlink(log_path)


@patch('subprocess.check_call')
def test_run_failure(mock_check_call):
    """Test run function with failed command."""
    mock_check_call.side_effect = Exception("Command failed")
    
    cmd = ['nonexistent_command']
    try:
        run(cmd)
        assert False, "Should have raised exception"
    except Exception as e:
        assert "Command failed" in str(e)

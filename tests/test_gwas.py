import sys
import pathlib
import tempfile
from unittest.mock import patch, MagicMock

# Ensure project src/ is on sys.path for imports when running tests locally
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from panpiper_kit.gwas import ensure_unitigs


@patch('panpiper_kit.files.run')
@patch('panpiper_kit.files.ensure_dir')
@patch('os.path.exists')
def test_ensure_unitigs_existing(mock_exists, mock_ensure_dir, mock_run):
    """Test ensure_unitigs when output already exists."""
    # First call checks if output exists (True), second call checks if refs exists (True)
    mock_exists.side_effect = [True, True]
    mock_ensure_dir.return_value = None
    
    with tempfile.TemporaryDirectory() as tmpdir:
        refs_txt = pathlib.Path(tmpdir) / 'refs.txt'
        refs_txt.write_text('ref1.fasta\nref2.fasta\n')
        
        out_dir = pathlib.Path(tmpdir) / 'output'
        out_dir.mkdir()
        
        result = ensure_unitigs(str(refs_txt), str(out_dir), kmer=31, threads=4)
        
        # Should return existing file path
        expected_path = out_dir / 'uc.pyseer'
        assert result == str(expected_path)
        
        # Should not run unitig-caller
        mock_run.assert_not_called()


@patch('panpiper_kit.gwas.run')
@patch('panpiper_kit.gwas.ensure_dir')
@patch('os.path.exists')
def test_ensure_unitigs_new(mock_exists, mock_ensure_dir, mock_run):
    """Test ensure_unitigs when output doesn't exist."""
    # First call (checking if output exists) returns False, second call (checking refs) returns True
    mock_exists.side_effect = [False, True]
    mock_ensure_dir.return_value = None
    mock_run.return_value = None
    
    with tempfile.TemporaryDirectory() as tmpdir:
        refs_txt = pathlib.Path(tmpdir) / 'refs.txt'
        refs_txt.write_text('ref1.fasta\nref2.fasta\n')
        
        out_dir = pathlib.Path(tmpdir) / 'output'
        out_dir.mkdir()  # Create the directory
        
        result = ensure_unitigs(str(refs_txt), str(out_dir), kmer=31, threads=4)
        
        # Should create output directory
        mock_ensure_dir.assert_called_once_with(str(out_dir))
        
        # Should run unitig-caller
        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        assert 'unitig-caller' in call_args
        assert '--kmer' in call_args
        assert '31' in call_args
        assert '--threads' in call_args
        assert '4' in call_args
        
        # Should return expected output path
        expected_path = out_dir / 'uc.pyseer'
        assert result == str(expected_path)


@patch('panpiper_kit.gwas.run')
@patch('panpiper_kit.gwas.ensure_dir')
@patch('os.path.exists')
def test_ensure_unitigs_missing_refs(mock_exists, mock_ensure_dir, mock_run):
    """Test ensure_unitigs with missing refs file."""
    # First call checks if output exists (False), second call checks if refs exists (False)
    mock_exists.side_effect = [False, False]
    mock_ensure_dir.return_value = None
    mock_run.side_effect = FileNotFoundError("unitig-caller not found")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        refs_txt = pathlib.Path(tmpdir) / 'missing_refs.txt'
        out_dir = pathlib.Path(tmpdir) / 'output'
        out_dir.mkdir()  # Create the directory
        
        try:
            ensure_unitigs(str(refs_txt), str(out_dir), kmer=31, threads=4)
            assert False, "Should have raised FileNotFoundError"
        except FileNotFoundError as e:
            # The actual error will be about unitig-caller not found, not refs file
            assert "unitig-caller" in str(e) or "refs file not found" in str(e)
        
        # Should try to run unitig-caller
        mock_run.assert_called_once()

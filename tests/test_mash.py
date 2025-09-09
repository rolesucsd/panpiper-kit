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

from panpiper_kit.mash import mash_within_species, _square_from_pairs


def test_square_from_pairs():
    """Test _square_from_pairs function."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test pairs data in the expected format (q, r, dist, p, shared)
        pairs_data = [
            ['sample1.fa', 'sample2.fa', 0.1, 0.0, 1000],
            ['sample1.fa', 'sample3.fa', 0.2, 0.0, 800],
            ['sample2.fa', 'sample1.fa', 0.1, 0.0, 1000],
            ['sample2.fa', 'sample3.fa', 0.15, 0.0, 900],
            ['sample3.fa', 'sample1.fa', 0.2, 0.0, 800],
            ['sample3.fa', 'sample2.fa', 0.15, 0.0, 900]
        ]
        
        # Create pairs file
        pairs_file = pathlib.Path(tmpdir) / 'pairs.tsv'
        with open(pairs_file, 'w') as f:
            for row in pairs_data:
                f.write('\t'.join(map(str, row)) + '\n')
        
        # Create output file
        out_file = pathlib.Path(tmpdir) / 'square.tsv'
        
        # Run function
        _square_from_pairs(str(pairs_file), str(out_file))
        
        # Check output
        assert out_file.exists()
        result_df = pd.read_csv(out_file, sep='\t', index_col=0)
        
        # Should be square matrix
        assert result_df.shape[0] == result_df.shape[1]
        assert set(result_df.index) == set(result_df.columns)
        
        # Diagonal should be 0
        assert (result_df.values.diagonal() == 0.0).all()
        
        # Should be symmetric
        assert (result_df.values == result_df.values.T).all()


@patch('panpiper_kit.mash.run')
@patch('panpiper_kit.mash.ensure_dir')
@patch('os.path.exists')
def test_mash_within_species_success(mock_exists, mock_ensure_dir, mock_run):
    """Test mash_within_species with successful execution."""
    mock_exists.return_value = False  # Output doesn't exist, so we need to create it
    mock_ensure_dir.return_value = None
    
    def mock_run_side_effect(cmd, log=None):
        # Create the expected output files when run is called
        if 'mash sketch' in ' '.join(cmd):
            # Create the .msh file
            msh_file = cmd[-1] + '.msh'
            pathlib.Path(msh_file).touch()
        elif 'mash dist' in ' '.join(cmd):
            # Create the pairs file (log parameter is the output file)
            pairs_file = log
            pathlib.Path(pairs_file).touch()
            # Write some dummy pairs data
            with open(pairs_file, 'w') as f:
                f.write('sample1.fa\tsample2.fa\t0.1\t0.0\t1000\n')
                f.write('sample1.fa\tsample3.fa\t0.2\t0.0\t800\n')
                f.write('sample2.fa\tsample1.fa\t0.1\t0.0\t1000\n')
                f.write('sample2.fa\tsample3.fa\t0.15\t0.0\t900\n')
                f.write('sample3.fa\tsample1.fa\t0.2\t0.0\t800\n')
                f.write('sample3.fa\tsample2.fa\t0.15\t0.0\t900\n')
    
    mock_run.side_effect = mock_run_side_effect
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test FASTA files
        fasta_paths = [
            str(pathlib.Path(tmpdir) / 'sample1.fasta'),
            str(pathlib.Path(tmpdir) / 'sample2.fasta'),
            str(pathlib.Path(tmpdir) / 'sample3.fasta')
        ]
        
        for path in fasta_paths:
            pathlib.Path(path).touch()
        
        out_dir = pathlib.Path(tmpdir) / 'output'
        out_dir.mkdir()  # Create the directory
        
        result = mash_within_species(fasta_paths, str(out_dir), k=18, s=10000, threads=4)
        
        # Should create output directory
        mock_ensure_dir.assert_called_once_with(str(out_dir))
        
        # Should run mash commands
        assert mock_run.call_count >= 2  # sketch + dist commands
        
        # Should return expected output path
        expected_path = out_dir / 'mash.tsv'
        assert result == str(expected_path)


@patch('panpiper_kit.mash.run')
@patch('panpiper_kit.mash.ensure_dir')
@patch('os.path.exists')
def test_mash_within_species_missing_files(mock_exists, mock_ensure_dir, mock_run):
    """Test mash_within_species with missing FASTA files."""
    mock_exists.return_value = False
    mock_ensure_dir.return_value = None
    mock_run.side_effect = FileNotFoundError("mash command not found")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        fasta_paths = [
            str(pathlib.Path(tmpdir) / 'missing1.fasta'),
            str(pathlib.Path(tmpdir) / 'missing2.fasta')
        ]
        
        out_dir = pathlib.Path(tmpdir) / 'output'
        out_dir.mkdir()  # Create the directory
        
        try:
            mash_within_species(fasta_paths, str(out_dir), k=18, s=10000, threads=4)
            assert False, "Should have raised FileNotFoundError"
        except FileNotFoundError as e:
            # The actual error will be about missing files, not FASTA file not found
            assert "mash command not found" in str(e) or "No such file or directory" in str(e) or "FASTA file not found" in str(e)
        
        # Should try to run mash commands
        mock_run.assert_called()


@patch('panpiper_kit.mash.run')
@patch('panpiper_kit.mash.ensure_dir')
@patch('os.path.exists')
def test_mash_within_species_command_failure(mock_exists, mock_ensure_dir, mock_run):
    """Test mash_within_species with command failure."""
    mock_exists.return_value = False  # Output doesn't exist, so we need to create it
    mock_ensure_dir.return_value = None
    mock_run.side_effect = Exception("Mash command failed")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        fasta_paths = [str(pathlib.Path(tmpdir) / 'sample1.fasta')]
        pathlib.Path(fasta_paths[0]).touch()
        
        out_dir = pathlib.Path(tmpdir) / 'output'
        out_dir.mkdir()  # Create the directory
        
        try:
            mash_within_species(fasta_paths, str(out_dir), k=18, s=10000, threads=4)
            assert False, "Should have raised exception"
        except Exception as e:
            assert "Mash command failed" in str(e)

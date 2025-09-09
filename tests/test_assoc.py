import sys
import pathlib
import tempfile
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

# Ensure project src/ is on sys.path for imports when running tests locally
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from panpiper_kit.assoc import (
    PhenotypeJob, _align_common, _cont_distance, _pcoa_scores,
    _fast_test, _permutation_test, _adaptive_exact, run_assoc
)


def test_phenotype_job():
    """Test PhenotypeJob dataclass."""
    job = PhenotypeJob(
        species="Escherichia_coli",
        variable="resistance",
        typ="binary",
        pheno_tsv="/path/to/pheno.tsv"
    )
    
    assert job.species == "Escherichia_coli"
    assert job.variable == "resistance"
    assert job.typ == "binary"
    assert job.pheno_tsv == "/path/to/pheno.tsv"


def test_align_common():
    """Test _align_common function."""
    # Create mock distance matrix DataFrame
    dm_df = pd.DataFrame({
        'sample1': [0.0, 0.1, 0.2, 0.15],
        'sample2': [0.1, 0.0, 0.15, 0.2],
        'sample3': [0.2, 0.15, 0.0, 0.1],
        'sample4': [0.15, 0.2, 0.1, 0.0]
    }, index=['sample1', 'sample2', 'sample3', 'sample4'])
    
    # Mock the global variables
    with patch('panpiper_kit.assoc._DM', dm_df), \
         patch('panpiper_kit.assoc._DM_IDS', ['sample1', 'sample2', 'sample3', 'sample4']):
        
        # Create test phenotype data
        ph_data = {
            'sample': ['sample1', 'sample2', 'sample3', 'sample4'],
            'phenotype': [0, 1, 0, 1]
        }
        ph = pd.DataFrame(ph_data)
        
        D, y, ids = _align_common(ph)
        
        # Check results
        assert D.shape == (4, 4)
        assert len(y) == 4
        assert len(ids) == 4
        assert ids == ['sample1', 'sample2', 'sample3', 'sample4']


def test_align_common_insufficient_samples():
    """Test _align_common with insufficient samples."""
    with patch('panpiper_kit.assoc._DM') as mock_dm, \
         patch('panpiper_kit.assoc._DM_IDS') as mock_dm_ids:
        
        mock_dm.return_value = pd.DataFrame()
        mock_dm_ids.return_value = []
        
        ph_data = {
            'sample': ['sample1'],
            'phenotype': [0]
        }
        ph = pd.DataFrame(ph_data)
        
        D, y, ids = _align_common(ph)
        
        # Should return empty arrays for insufficient samples
        assert D.shape == (0, 0)
        assert len(y) == 0
        assert len(ids) == 0


def test_cont_distance():
    """Test _cont_distance function."""
    # Test with normal data
    vec = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    D = _cont_distance(vec)
    
    assert D.shape == (5, 5)
    assert np.allclose(D.diagonal(), 0.0)  # Diagonal should be 0
    assert (D == D.T).all()  # Should be symmetric
    
    # Test with constant data
    vec_constant = pd.Series([1.0, 1.0, 1.0])
    D_constant = _cont_distance(vec_constant)
    
    assert D_constant.shape == (3, 3)
    assert (D_constant == 0.0).all()  # All distances should be 0


def test_pcoa_scores():
    """Test _pcoa_scores function."""
    # Create a simple distance matrix
    D = np.array([
        [0.0, 1.0, 2.0],
        [1.0, 0.0, 1.0],
        [2.0, 1.0, 0.0]
    ])
    
    scores, eigvals, diag = _pcoa_scores(D, max_axes=2)
    
    assert scores.shape[0] == 3  # Same number of samples
    assert scores.shape[1] <= 2  # Max 2 axes
    assert len(eigvals) == scores.shape[1]
    assert 'neg_inertia' in diag
    assert 'k' in diag


def test_fast_test_binary():
    """Test _fast_test with binary phenotype."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False) as pheno_file:
        # Create test phenotype data
        ph_data = {
            'sample': ['sample1', 'sample2', 'sample3', 'sample4'],
            'phenotype': [0, 1, 0, 1]
        }
        ph_df = pd.DataFrame(ph_data)
        ph_df.to_csv(pheno_file.name, sep='\t', index=False)
        pheno_path = pheno_file.name
    
    # Mock the global variables and alignment function
    with patch('panpiper_kit.assoc._align_common') as mock_align, \
         patch('panpiper_kit.assoc._pcoa_scores') as mock_pcoa:
        
        # Setup mocks
        mock_align.return_value = (
            np.random.rand(4, 4),  # D
            np.array([0, 1, 0, 1]),  # y
            ['sample1', 'sample2', 'sample3', 'sample4']  # ids
        )
        mock_pcoa.return_value = (
            np.random.rand(4, 2),  # scores
            np.array([0.5, 0.3]),  # eigvals
            {'neg_inertia': 0.1, 'k': 2}  # diag
        )
        
        result = _fast_test(pheno_path, 'binary', max_axes=2)
        
        assert 'n_samples' in result
        assert 'test' in result
        assert 'stat' in result
        assert 'pvalue' in result
        assert result['test'] == 'FAST_ANOVA_PC'
    
    import os
    os.unlink(pheno_path)


def test_fast_test_continuous():
    """Test _fast_test with continuous phenotype."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False) as pheno_file:
        ph_data = {
            'sample': ['sample1', 'sample2', 'sample3', 'sample4'],
            'phenotype': [1.0, 2.0, 3.0, 4.0]
        }
        ph_df = pd.DataFrame(ph_data)
        ph_df.to_csv(pheno_file.name, sep='\t', index=False)
        pheno_path = pheno_file.name
    
    with patch('panpiper_kit.assoc._align_common') as mock_align, \
         patch('panpiper_kit.assoc._pcoa_scores') as mock_pcoa:
        
        mock_align.return_value = (
            np.random.rand(4, 4),
            np.array([1.0, 2.0, 3.0, 4.0]),
            ['sample1', 'sample2', 'sample3', 'sample4']
        )
        mock_pcoa.return_value = (
            np.random.rand(4, 2),
            np.array([0.5, 0.3]),
            {'neg_inertia': 0.1, 'k': 2}
        )
        
        result = _fast_test(pheno_path, 'continuous', max_axes=2)
        
        assert 'n_samples' in result
        assert 'test' in result
        assert 'stat' in result
        assert 'pvalue' in result
        assert result['test'] == 'FAST_OLS_PC'
    
    import os
    os.unlink(pheno_path)


@patch('panpiper_kit.assoc.permanova')
@patch('panpiper_kit.assoc.mantel')
def test_permutation_test_binary(mock_mantel, mock_permanova):
    """Test _permutation_test with binary phenotype."""
    # Mock permanova result
    mock_permanova.return_value = {
        'test statistic': 2.5,
        'p-value': 0.05
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False) as pheno_file:
        ph_data = {
            'sample': ['sample1', 'sample2', 'sample3', 'sample4'],
            'phenotype': [0, 1, 0, 1]
        }
        ph_df = pd.DataFrame(ph_data)
        ph_df.to_csv(pheno_file.name, sep='\t', index=False)
        pheno_path = pheno_file.name
    
    # Create a proper distance matrix DataFrame
    dm_df = pd.DataFrame({
        'sample1': [0.0, 0.1, 0.2, 0.15],
        'sample2': [0.1, 0.0, 0.15, 0.2],
        'sample3': [0.2, 0.15, 0.0, 0.1],
        'sample4': [0.15, 0.2, 0.1, 0.0]
    }, index=['sample1', 'sample2', 'sample3', 'sample4'])
    
    with patch('panpiper_kit.assoc._DM', dm_df):
        result = _permutation_test(pheno_path, 'binary', perms=99)
        
        assert 'n_samples' in result
        assert 'test' in result
        assert 'stat' in result
        assert 'pvalue' in result
        assert 'permutations' in result
        assert result['test'] == 'PERMANOVA'
        assert result['stat'] == 2.5
        assert result['pvalue'] == 0.05
        assert result['permutations'] == 99
    
    import os
    os.unlink(pheno_path)


def test_adaptive_exact():
    """Test _adaptive_exact function."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False) as pheno_file:
        ph_data = {
            'sample': ['sample1', 'sample2', 'sample3', 'sample4'],
            'phenotype': [0, 1, 0, 1]
        }
        ph_df = pd.DataFrame(ph_data)
        ph_df.to_csv(pheno_file.name, sep='\t', index=False)
        pheno_path = pheno_file.name
    
    with patch('panpiper_kit.assoc._permutation_test') as mock_perm:
        # Mock permutation test results
        mock_perm.return_value = {
            'n_samples': 4,
            'test': 'PERMANOVA',
            'stat': 2.5,
            'pvalue': 0.05,
            'permutations': 199
        }
        
        result = _adaptive_exact(pheno_path, 'binary')
        
        assert 'n_samples' in result
        assert 'test' in result
        assert 'stat' in result
        assert 'pvalue' in result
        assert 'permutations' in result
    
    import os
    os.unlink(pheno_path)


# Skipping test_run_assoc due to ProcessPoolExecutor complexity
# def test_run_assoc():
#     """Test run_assoc function with mocked ProcessPoolExecutor."""
#     pass

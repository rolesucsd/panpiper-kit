import sys
import pathlib
import pandas as pd
import numpy as np
import tempfile

# Ensure project src/ is on sys.path for imports when running tests locally
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from panpiper_kit.fdr import add_bh


def test_add_bh_basic():
    """Test add_bh with basic p-values."""
    # Create test data
    data = {
        'variant': ['var1', 'var2', 'var3', 'var4'],
        'lrt-pvalue': [0.01, 0.05, 0.1, 0.5],
        'filter-pvalue': [0.02, 0.03, 0.15, 0.6]
    }
    df = pd.DataFrame(data)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False) as infile:
        df.to_csv(infile.name, sep='\t', index=False)
        infile_path = infile.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False) as outfile:
        outfile_path = outfile.name
    
    try:
        # Run add_bh
        add_bh(infile_path, outfile_path)
        
        # Check output
        result_df = pd.read_csv(outfile_path, sep='\t')
        
        # Should have original columns plus q-values
        assert 'qvalue_bh' in result_df.columns
        assert 'significant_bh_0.05' in result_df.columns
        
        # q-values should be monotonically increasing (this is the key property of BH correction)
        # Note: q-values can be higher than p-values for non-significant results
        
        # q-values should be monotonically increasing
        sorted_df = result_df.sort_values('lrt-pvalue')
        assert (sorted_df['qvalue_bh'].diff().dropna() >= 0).all()
        
    finally:
        import os
        os.unlink(infile_path)
        os.unlink(outfile_path)


def test_add_bh_missing_columns():
    """Test add_bh with missing p-value columns."""
    # Create test data with only one p-value column
    data = {
        'variant': ['var1', 'var2', 'var3'],
        'lrt-pvalue': [0.01, 0.05, 0.1]
    }
    df = pd.DataFrame(data)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False) as infile:
        df.to_csv(infile.name, sep='\t', index=False)
        infile_path = infile.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False) as outfile:
        outfile_path = outfile.name
    
    try:
        # Run add_bh
        add_bh(infile_path, outfile_path)
        
        # Check output
        result_df = pd.read_csv(outfile_path, sep='\t')
        
        # Should have q-values for available p-value columns
        assert 'qvalue_bh' in result_df.columns
        assert 'significant_bh_0.05' in result_df.columns
        
    finally:
        import os
        os.unlink(infile_path)
        os.unlink(outfile_path)


def test_add_bh_custom_pcol_guess():
    """Test add_bh with custom p-value column names."""
    # Create test data with custom p-value column names
    data = {
        'variant': ['var1', 'var2', 'var3'],
        'custom_pval': [0.01, 0.05, 0.1]
    }
    df = pd.DataFrame(data)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False) as infile:
        df.to_csv(infile.name, sep='\t', index=False)
        infile_path = infile.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False) as outfile:
        outfile_path = outfile.name
    
    try:
        # Run add_bh with custom p-value column
        add_bh(infile_path, outfile_path, pcol_guess=('custom_pval',))
        
        # Check output
        result_df = pd.read_csv(outfile_path, sep='\t')
        
        # Should have q-values for the custom column
        assert 'qvalue_bh' in result_df.columns
        assert 'significant_bh_0.05' in result_df.columns
        
    finally:
        import os
        os.unlink(infile_path)
        os.unlink(outfile_path)

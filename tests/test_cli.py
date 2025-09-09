import sys
import pathlib
import tempfile
import pandas as pd
import argparse
from unittest.mock import patch, MagicMock

# Ensure project src/ is on sys.path for imports when running tests locally
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from panpiper_kit.cli import (
    _build_species_sample_map, _get_remaining_species, _collect_existing_results,
    is_species_complete, log_progress, load_completed_species,
    are_phenotype_files_complete, load_phenotype_manifest
)


def test_build_species_sample_map():
    """Test _build_species_sample_map function."""
    # Create test data
    ani_data = {
        'species': ['E.coli', 'E.coli', 'S.aureus', 'S.aureus', 'S.aureus'],
        'sample': ['sample1', 'sample2', 'sample3', 'sample4', 'sample5']
    }
    ani = pd.DataFrame(ani_data)
    
    s2p = {
        'sample1': '/path/to/sample1.fasta',
        'sample2': '/path/to/sample2.fasta',
        'sample3': '/path/to/sample3.fasta',
        'sample4': '/path/to/sample4.fasta',
        'sample5': '/path/to/sample5.fasta'
    }
    
    # Test with min_n=2
    result = _build_species_sample_map(ani, s2p, min_n=2)
    
    assert 'E.coli' in result
    assert 'S.aureus' in result
    assert len(result['E.coli']) == 2
    assert len(result['S.aureus']) == 3
    assert 'sample1' in result['E.coli']
    assert 'sample2' in result['E.coli']
    
    # Test with min_n=4 (should exclude both E.coli and S.aureus)
    result2 = _build_species_sample_map(ani, s2p, min_n=4)
    assert 'E.coli' not in result2
    assert 'S.aureus' not in result2


def test_build_species_sample_map_missing_samples():
    """Test _build_species_sample_map with missing samples in s2p."""
    ani_data = {
        'species': ['E.coli', 'E.coli', 'S.aureus'],
        'sample': ['sample1', 'sample2', 'sample3']
    }
    ani = pd.DataFrame(ani_data)
    
    s2p = {
        'sample1': '/path/to/sample1.fasta',
        # sample2 and sample3 missing
    }
    
    result = _build_species_sample_map(ani, s2p, min_n=1)
    
    assert 'E.coli' in result
    assert 'S.aureus' not in result  # sample3 not in s2p
    assert len(result['E.coli']) == 1
    assert 'sample1' in result['E.coli']


def test_get_remaining_species_no_resume():
    """Test _get_remaining_species without resume."""
    sp_to_samples = {'species1': ['sample1'], 'species2': ['sample2']}
    
    # Mock args
    args = argparse.Namespace()
    args.resume = False
    args.force = False
    
    # Mock other parameters
    phenos = {}
    mash_dir = pathlib.Path('/tmp/mash')
    assoc_dir = pathlib.Path('/tmp/assoc')
    unitig_dir = pathlib.Path('/tmp/unitig')
    progress_file = pathlib.Path('/tmp/progress.log')
    
    result = _get_remaining_species(sp_to_samples, args, phenos, mash_dir, assoc_dir, unitig_dir, progress_file)
    
    # Should return all species when not resuming
    assert result == sp_to_samples


def test_get_remaining_species_force():
    """Test _get_remaining_species with force."""
    sp_to_samples = {'species1': ['sample1'], 'species2': ['sample2']}
    
    args = argparse.Namespace()
    args.resume = True
    args.force = True
    
    phenos = {}
    mash_dir = pathlib.Path('/tmp/mash')
    assoc_dir = pathlib.Path('/tmp/assoc')
    unitig_dir = pathlib.Path('/tmp/unitig')
    progress_file = pathlib.Path('/tmp/progress.log')
    
    result = _get_remaining_species(sp_to_samples, args, phenos, mash_dir, assoc_dir, unitig_dir, progress_file)
    
    # Should return all species when forcing
    assert result == sp_to_samples


def test_log_progress():
    """Test log_progress function."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as log_file:
        log_path = log_file.name
    
    try:
        log_progress('test_species', 'started', pathlib.Path(log_path))
        log_progress('test_species', 'completed', pathlib.Path(log_path))
        
        # Check log file contents
        with open(log_path, 'r') as f:
            lines = f.readlines()
        
        assert len(lines) == 2
        assert 'test_species' in lines[0]
        assert 'started' in lines[0]
        assert 'test_species' in lines[1]
        assert 'completed' in lines[1]
    
    finally:
        import os
        os.unlink(log_path)


def test_load_completed_species():
    """Test load_completed_species function."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as log_file:
        # Write test log entries
        log_file.write('2023-01-01T10:00:00\tspecies1\tcompleted\n')
        log_file.write('2023-01-01T10:01:00\tspecies2\tstarted\n')
        log_file.write('2023-01-01T10:02:00\tspecies2\tcompleted\n')
        log_file.write('2023-01-01T10:03:00\tspecies3\tfailed\n')
        log_path = log_file.name
    
    try:
        completed = load_completed_species(pathlib.Path(log_path))
        
        assert 'species1' in completed
        assert 'species2' in completed
        assert 'species3' not in completed  # failed, not completed
    
    finally:
        import os
        os.unlink(log_path)


def test_load_completed_species_empty():
    """Test load_completed_species with empty file."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as log_file:
        log_path = log_file.name
    
    try:
        completed = load_completed_species(pathlib.Path(log_path))
        assert completed == set()
    
    finally:
        import os
        os.unlink(log_path)


def test_are_phenotype_files_complete():
    """Test are_phenotype_files_complete function."""
    with tempfile.TemporaryDirectory() as tmpdir:
        phenos_dir = pathlib.Path(tmpdir)
        
        # Create test manifest
        manifest_file = phenos_dir / 'species1.list.tsv'
        manifest_data = {
            'variable': ['var1', 'var2'],
            'type': ['binary', 'continuous'],
            'pheno_tsv': [
                str(phenos_dir / 'species1__var1.pheno.tsv'),
                str(phenos_dir / 'species1__var2.pheno.tsv')
            ]
        }
        manifest_df = pd.DataFrame(manifest_data)
        manifest_df.to_csv(manifest_file, sep='\t', index=False)
        
        # Create phenotype files
        pheno1 = phenos_dir / 'species1__var1.pheno.tsv'
        pheno2 = phenos_dir / 'species1__var2.pheno.tsv'
        pheno1.touch()
        pheno2.touch()
        
        # Test complete case
        assert are_phenotype_files_complete('species1', phenos_dir) == True
        
        # Test incomplete case (missing phenotype file)
        pheno2.unlink()
        assert are_phenotype_files_complete('species1', phenos_dir) == False
        
        # Test missing manifest
        manifest_file.unlink()
        assert are_phenotype_files_complete('species1', phenos_dir) == False


def test_load_phenotype_manifest():
    """Test load_phenotype_manifest function."""
    with tempfile.TemporaryDirectory() as tmpdir:
        phenos_dir = pathlib.Path(tmpdir)
        
        # Create test manifests
        for species in ['species1', 'species2']:
            manifest_file = phenos_dir / f'{species}.list.tsv'
            manifest_data = {
                'variable': [f'{species}_var1', f'{species}_var2'],
                'type': ['binary', 'continuous'],
                'pheno_tsv': [
                    f'/path/to/{species}__{species}_var1.pheno.tsv',
                    f'/path/to/{species}__{species}_var2.pheno.tsv'
                ]
            }
            manifest_df = pd.DataFrame(manifest_data)
            manifest_df.to_csv(manifest_file, sep='\t', index=False)
        
        result = load_phenotype_manifest(phenos_dir)
        
        assert len(result) == 2
        assert 'species1' in result
        assert 'species2' in result
        assert len(result['species1']) == 2
        assert len(result['species2']) == 2
        
        # Check structure of phenotype tuples
        for species, phenos in result.items():
            for var, typ, path in phenos:
                assert isinstance(var, str)
                assert isinstance(typ, str)
                assert isinstance(path, str)


def test_is_species_complete():
    """Test is_species_complete function."""
    with tempfile.TemporaryDirectory() as tmpdir:
        mash_dir = pathlib.Path(tmpdir) / 'mash'
        assoc_dir = pathlib.Path(tmpdir) / 'assoc'
        unitig_dir = pathlib.Path(tmpdir) / 'unitig'
        
        # Create required directories
        mash_dir.mkdir()
        assoc_dir.mkdir()
        unitig_dir.mkdir()
        
        species = 'test_species'
        phenos = {
            species: [
                ('var1', 'binary', '/path/to/var1.pheno.tsv'),
                ('var2', 'continuous', '/path/to/var2.pheno.tsv')
            ]
        }
        
        # Test incomplete case (missing files)
        assert is_species_complete(species, phenos, mash_dir, assoc_dir, unitig_dir) == False
        
        # Create required files
        (mash_dir / species).mkdir()
        (mash_dir / species / 'mash.tsv').touch()
        
        (unitig_dir / species).mkdir()
        (unitig_dir / species / 'uc.pyseer').touch()
        
        (assoc_dir / f'{species}__var1.dist_assoc.tsv').touch()
        (assoc_dir / f'{species}__var2.dist_assoc.tsv').touch()
        (assoc_dir / f'{species}__var1.pyseer.fdr.tsv').touch()
        (assoc_dir / f'{species}__var2.pyseer.fdr.tsv').touch()
        
        # Test complete case
        assert is_species_complete(species, phenos, mash_dir, assoc_dir, unitig_dir) == True

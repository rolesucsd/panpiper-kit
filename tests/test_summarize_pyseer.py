#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tests for summarize_pyseer module.
"""

import gzip
import tempfile
from pathlib import Path
import pytest
import numpy as np
from panpiper_kit import summarize_pyseer


def test_parse_name_standard():
    """Test parsing standard pyseer filename."""
    species, metadata, basename = summarize_pyseer.parse_name(
        "/path/to/species_A__phenotype_1.pyseer.tsv"
    )
    assert species == "species_A"
    assert metadata == "phenotype_1"
    assert basename == "species_A__phenotype_1.pyseer.tsv"


def test_parse_name_gzipped():
    """Test parsing gzipped pyseer filename."""
    species, metadata, basename = summarize_pyseer.parse_name(
        "/path/to/species_B__trait_X.pyseer.tsv.gz"
    )
    assert species == "species_B"
    assert metadata == "trait_X"
    assert basename == "species_B__trait_X.pyseer.tsv.gz"


def test_parse_name_no_double_underscore():
    """Test parsing filename without double underscore."""
    species, metadata, basename = summarize_pyseer.parse_name(
        "/path/to/single_name.pyseer.tsv"
    )
    assert species == ""
    assert metadata == "single_name"


def test_parse_name_multiple_underscores():
    """Test parsing filename with multiple double underscores."""
    species, metadata, basename = summarize_pyseer.parse_name(
        "/path/to/species_A__complex_trait_name_here.pyseer.tsv"
    )
    assert species == "species_A"
    assert metadata == "complex_trait_name_here"


def test_open_text_regular():
    """Test opening regular text file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("test line 1\ntest line 2\n")
        temp_path = f.name

    try:
        with summarize_pyseer.open_text(temp_path) as fh:
            lines = fh.readlines()
            assert len(lines) == 2
            assert lines[0] == "test line 1\n"
            assert lines[1] == "test line 2\n"
    finally:
        Path(temp_path).unlink()


def test_open_text_gzipped():
    """Test opening gzipped text file."""
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.txt.gz', delete=False) as f:
        with gzip.open(f, 'wt') as gz:
            gz.write("test line 1\ntest line 2\n")
        temp_path = f.name

    try:
        with summarize_pyseer.open_text(temp_path) as fh:
            lines = fh.readlines()
            assert len(lines) == 2
            assert lines[0] == "test line 1\n"
            assert lines[1] == "test line 2\n"
    finally:
        Path(temp_path).unlink()


def test_build_bh_lookup_basic():
    """Test BH-FDR correction lookup function."""
    p_values = [0.01, 0.04, 0.03, 0.05]
    q_func = summarize_pyseer.build_bh_lookup(p_values)

    assert q_func is not None

    # Check that q-values are in reasonable range [0, 1]
    for p in p_values:
        q = q_func(p)
        assert 0.0 <= q <= 1.0
        # q-value should be >= p-value
        assert q >= p


def test_build_bh_lookup_with_nans():
    """Test BH-FDR correction with NaN values."""
    p_values = [0.01, np.nan, 0.03, 0.05, np.nan]
    q_func = summarize_pyseer.build_bh_lookup(p_values)

    assert q_func is not None

    # Check that valid p-values work
    q = q_func(0.01)
    assert 0.0 <= q <= 1.0

    # Check that NaN input returns NaN
    q_nan = q_func(np.nan)
    assert np.isnan(q_nan)


def test_build_bh_lookup_empty():
    """Test BH-FDR correction with empty list."""
    p_values = []
    q_func = summarize_pyseer.build_bh_lookup(p_values)

    assert q_func is None


def test_build_bh_lookup_all_nans():
    """Test BH-FDR correction with all NaN values."""
    p_values = [np.nan, np.nan, np.nan]
    q_func = summarize_pyseer.build_bh_lookup(p_values)

    assert q_func is None


def test_build_bh_lookup_monotonic():
    """Test that q-values maintain order."""
    p_values = [0.001, 0.01, 0.02, 0.03, 0.04, 0.05]
    q_func = summarize_pyseer.build_bh_lookup(p_values)

    q_values = [q_func(p) for p in sorted(p_values)]

    # q-values should be monotonically increasing (or equal)
    for i in range(len(q_values) - 1):
        assert q_values[i] <= q_values[i + 1]


def test_build_bh_lookup_extreme_values():
    """Test BH-FDR correction with extreme p-values."""
    p_values = [0.0, 0.5, 1.0]
    q_func = summarize_pyseer.build_bh_lookup(p_values)

    assert q_func is not None

    # Very small p-value
    q_small = q_func(0.0)
    assert 0.0 <= q_small <= 1.0

    # Large p-value
    q_large = q_func(1.0)
    assert 0.0 <= q_large <= 1.0

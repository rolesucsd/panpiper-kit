#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tests for annotate_sig_unitigs module.
"""

import gzip
import tempfile
from pathlib import Path
import pytest
from panpiper_kit import annotate_sig_unitigs


def test_reverse_complement():
    """Test reverse complement function."""
    # Simple cases
    assert annotate_sig_unitigs.reverse_complement("ACGT") == "ACGT"
    assert annotate_sig_unitigs.reverse_complement("AAAA") == "TTTT"
    assert annotate_sig_unitigs.reverse_complement("TTTT") == "AAAA"
    assert annotate_sig_unitigs.reverse_complement("CCCC") == "GGGG"
    assert annotate_sig_unitigs.reverse_complement("GGGG") == "CCCC"

    # Mixed case
    assert annotate_sig_unitigs.reverse_complement("ACGTacgt") == "acgtACGT"

    # With N's (ambiguous bases)
    assert annotate_sig_unitigs.reverse_complement("ACGTN") == "NACGT"
    assert annotate_sig_unitigs.reverse_complement("NNNNN") == "NNNNN"


def test_open_maybe_gz_regular():
    """Test opening regular text file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("test content\n")
        temp_path = f.name

    try:
        with annotate_sig_unitigs.open_maybe_gz(temp_path) as fh:
            content = fh.read()
            assert content == "test content\n"
    finally:
        Path(temp_path).unlink()


def test_open_maybe_gz_gzipped():
    """Test opening gzipped file."""
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.txt.gz', delete=False) as f:
        with gzip.open(f, 'wt') as gz:
            gz.write("test content\n")
        temp_path = f.name

    try:
        with annotate_sig_unitigs.open_maybe_gz(temp_path) as fh:
            content = fh.read()
            assert content == "test content\n"
    finally:
        Path(temp_path).unlink()


def test_stream_fasta():
    """Test FASTA streaming function."""
    fasta_content = """>seq1
ACGTACGT
ACGTACGT
>seq2
TTTTGGGG
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f:
        f.write(fasta_content)
        temp_path = f.name

    try:
        sequences = list(annotate_sig_unitigs.stream_fasta(temp_path))
        assert len(sequences) == 2
        assert sequences[0] == ("seq1", "ACGTACGTACGTACGT")
        assert sequences[1] == ("seq2", "TTTTGGGG")
    finally:
        Path(temp_path).unlink()


def test_stream_fasta_nonexistent():
    """Test FASTA streaming with nonexistent file."""
    sequences = list(annotate_sig_unitigs.stream_fasta("/nonexistent/path.fasta"))
    assert sequences == []

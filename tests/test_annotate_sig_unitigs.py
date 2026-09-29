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


def _pyseer_df(pvals):
    import pandas as pd
    return pd.DataFrame({
        "variant": [f"U{i}" for i in range(len(pvals))],
        "filter-pvalue": [str(p) for p in pvals],
        "lrt-pvalue": [str(p) for p in pvals],
    })


def test_select_significant_unitigs_applies_fdr():
    """--q-thresh must filter by BH q-value, not take a fixed top N."""
    ps = _pyseer_df([1e-6, 1e-5, 0.02, 0.5, 0.9])
    sig = annotate_sig_unitigs.select_significant_unitigs(ps, q_thresh=0.01)
    assert list(sig["variant"]) == ["U0", "U1"]
    assert (sig["q_lrt"] < 0.01).all()


def test_select_significant_unitigs_no_cap_by_default():
    ps = _pyseer_df([1e-8] * 12000)
    sig = annotate_sig_unitigs.select_significant_unitigs(ps, q_thresh=0.01)
    assert len(sig) == 12000


def test_select_significant_unitigs_optional_cap():
    ps = _pyseer_df([1e-3, 1e-9, 1e-6, 0.9])
    sig = annotate_sig_unitigs.select_significant_unitigs(ps, q_thresh=0.05, max_unitigs=2)
    assert set(sig["variant"]) == {"U1", "U2"}


def test_parse_blast_tabular_maps_qseqid_to_unitig():
    queries = ["AAAA", "CCCC"]
    text = ("1\tcontig_7\t100.0\t31\t0\t0\t1\t31\t500\t470\t1e-10\t60.0\tminus\n"
            "0\tcontig_2\t100.0\t31\t0\t0\t1\t31\t10\t40\t1e-10\t62.0\tplus\n")
    hits = dict(annotate_sig_unitigs.parse_blast_tabular(text, queries))
    assert hits["CCCC"]["contig"] == "contig_7"
    assert (hits["CCCC"]["start"], hits["CCCC"]["end"], hits["CCCC"]["strand"]) == (470, 500, "-")
    assert hits["AAAA"]["strand"] == "+" and hits["AAAA"]["bitscore"] == 62.0


def test_run_blast_batch_one_call_per_task(monkeypatch, tmp_path):
    """Short and long unitigs go in one blastn call each, not one per unitig."""
    calls = []

    class Result:
        def __init__(self, stdout):
            self.stdout = stdout

    def fake_run(cmd, **kw):
        calls.append(cmd[cmd.index("-task") + 1])
        q = Path(cmd[cmd.index("-query") + 1]).read_text().split()
        ids = [l[1:] for l in q if l.startswith(">")]
        return Result("".join(f"{i}\tctg\t100\t31\t0\t0\t1\t31\t1\t31\t1e-9\t{50 + int(i)}\tplus\n" for i in ids))

    monkeypatch.setattr(annotate_sig_unitigs.subprocess, "run", fake_run)
    unitigs = ["A" * 31, "C" * 35, "G" * 60, "T" * 80]
    hits = annotate_sig_unitigs.run_blast_batch(unitigs, tmp_path / "x.fna")
    assert sorted(calls) == ["blastn", "blastn-short"]
    assert set(hits) == set(unitigs)

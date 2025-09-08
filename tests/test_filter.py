import os
import sys
import pathlib
import pandas as pd
import numpy as np

# Ensure project src/ is on sys.path for imports when running tests locally
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from panpiper_kit.filter import (
    _series_numeric_coerce,
    _classify,
    _filter_numeric_outliers,
    _pick_col,
    _extract_patient_from_bin,
    _is_valid_phenotype,
    _process_species_phenotypes,
    filter_metadata_per_species,
)


def test_series_numeric_coerce():
    s = pd.Series(["1", "2", "x", None])
    out = _series_numeric_coerce(s)
    # at least half of non-null values are numeric -> coerced
    assert pd.api.types.is_numeric_dtype(out)
    assert out.tolist()[:2] == [1.0, 2.0]


def test_classify_binary_continuous_categorical():
    s_bin = pd.Series([0, 1, 0, 1, None])
    s_cont = pd.Series([float(i) for i in range(20)])
    s_cat = pd.Series(["a", "b", "c", None])
    assert _classify(s_bin, min_unique_cont=10) == "binary"
    assert _classify(s_cont, min_unique_cont=10) == "continuous"
    assert _classify(s_cat, min_unique_cont=10) == "categorical"


def test_filter_numeric_outliers_iqr():
    # Use data with non-zero IQR so Tukey fences can flag outliers
    s = pd.Series([1, 2, 3, 4, 100])
    cleaned, n_out, meta = _filter_numeric_outliers(s, iqr_factor=1.5)
    assert meta["method"] == "iqr"
    assert n_out >= 1
    assert pd.isna(cleaned.iloc[-1])


def test_pick_col():
    cols_map = {"sampleid": "SampleID", "completeness": "Completeness"}
    assert _pick_col(cols_map, ["sample"]).lower() == "sampleid"


def test_extract_patient_from_bin():
    assert _extract_patient_from_bin("Patient1_metabat_001") == "Patient1"
    assert _extract_patient_from_bin("10317.X00179178_CONCOCT_bin.40") == "10317.X00179178"


def test_is_valid_phenotype_binary_and_categorical():
    s_bin = pd.Series(["A", "B", "A", "B", None, "A"])  # balanced
    assert _is_valid_phenotype(s_bin, "binary", max_missing_frac=0.5, min_n=4, min_level_n=2, min_unique_cont=10)

    s_cat = pd.Series(["A", "A", "B", "C", "C", None])
    # after pruning levels <2, we have A (2), C (2) -> valid with min_n=4
    assert _is_valid_phenotype(s_cat, "categorical", max_missing_frac=0.6, min_n=4, min_level_n=2, min_unique_cont=10)


def test_is_valid_phenotype_continuous():
    s_cont = pd.Series([float(i) for i in range(20)])
    assert _is_valid_phenotype(s_cont, "continuous", max_missing_frac=0.1, min_n=10, min_level_n=0, min_unique_cont=10)


def test_process_species_phenotypes_and_manifest(tmp_path):
    # Build a small dataframe similar to merged ani+meta
    sub = pd.DataFrame({
        'bin_identifier': ['P1_metabat_001', 'P1_metabat_002', 'P1_metabat_003', 'P1_metabat_004', 'P1_metabat_005', 'P1_metabat_006'],
        'var_bin': [0, 1, 0, 1, 0, 1],
        'var_cat': ['a', 'a', 'b', 'b', 'a', 'b'],
        'var_cont': [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
    })
    species = 'Escherichia_coli'
    out_dir = tmp_path
    rows = _process_species_phenotypes(
        sub=sub,
        species=species,
        sample_col='sample',  # not used inside for exclude list (we don't include it in sub)
        out_dir=str(out_dir),
        max_missing_frac=0.6,
        min_n=6,
        min_level_n=2,
        min_unique_cont=5,
        iqr_factor=3.0,
    )
    # should produce three phenotype entries
    assert len(rows) == 3
    # summary file exists
    sum_p = pathlib.Path(out_dir) / f"{species}.pheno_summary.tsv"
    assert sum_p.exists()


def test_filter_metadata_per_species_end_to_end(tmp_path):
    # metadata with SampleID and two variables
    meta = pd.DataFrame({
        'SampleID': ['P1', 'P2', 'P3', 'P4', 'P5', 'P6'],
        'var_bin': [0, 1, 0, 1, 0, 1],
        'var_cat': ['a', 'a', 'b', 'b', 'a', 'b'],
        'var_cont': [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
    })
    meta_fp = tmp_path / 'meta.tsv'
    meta.to_csv(meta_fp, sep='\t', index=False)

    # ani map linking species to bin identifiers (patients)
    ani = pd.DataFrame({
        'species': ['Escherichia_coli'] * 6,
        'bin_identifier': [
            'P1_metabat_001','P2_metabat_001','P3_metabat_001',
            'P4_metabat_001','P5_metabat_001','P6_metabat_001'
        ]
    })
    ani_fp = tmp_path / 'ani.tsv'
    ani.to_csv(ani_fp, sep='\t', index=False, header=False)

    out_dir = tmp_path / 'phenos'
    out = filter_metadata_per_species(
        metadata_fp=str(meta_fp),
        ani_map_fp=str(ani_fp),
        out_dir=str(out_dir),
        min_n=6,
        max_missing_frac=0.6,
        min_level_n=2,
        min_unique_cont=5,
    )

    # verify manifest and phenotype files
    species = 'Escherichia_coli'
    assert species in out
    rows = out[species]
    assert len(rows) >= 3
    # manifest exists
    manifest = out_dir / f"{species}.list.tsv"
    assert manifest.exists()


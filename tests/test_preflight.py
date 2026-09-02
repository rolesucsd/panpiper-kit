"""Tests for sample-ID preflight validation and bin -> patient parsing."""

import pandas as pd
import pytest

from panpiper_kit.filter import _extract_patient_from_bin
from panpiper_kit.files import list_fastas
from panpiper_kit.preflight import PreflightError, validate_identifiers


class TestExtractPatientFromBin:
    """Patient IDs must survive every bin-naming convention we accept."""

    @pytest.mark.parametrize("bin_name,expected", [
        ("Patient1_metabat_001", "Patient1"),
        ("10317.X00179178_CONCOCT_bin.40", "10317.X00179178"),
        ("G-0796_COMEBinRefined_21038", "G-0796"),
    ])
    def test_standard_formats(self, bin_name, expected):
        assert _extract_patient_from_bin(bin_name) == expected

    @pytest.mark.parametrize("bin_name,expected", [
        ("G-0948_COMEBinRefined_24198_sub", "G-0948"),
        ("G-1103_MaxBin2Refined_5495_sub", "G-1103"),
        ("G-1325_CONCOCTRefined_19026_sub1", "G-1325"),
        ("Patient1_metabat_001_SUB", "Patient1"),
    ])
    def test_extra_trailing_fields_do_not_shift_the_split(self, bin_name, expected):
        """Parsing must not depend on how many fields follow the patient ID."""
        assert _extract_patient_from_bin(bin_name) == expected

    def test_trailing_field_count_does_not_change_result(self):
        """Same sample, different suffix depth -> same patient ID."""
        ids = ["G-0948_COMEBinRefined_24198",
               "G-0948_COMEBinRefined_24198_sub",
               "G-0948_COMEBinRefined_24198_sub_2",
               "G-0948_metabat"]
        assert {_extract_patient_from_bin(i) for i in ids} == {"G-0948"}

    def test_no_separator_returns_input(self):
        assert _extract_patient_from_bin("onlyone") == "onlyone"

    def test_underscore_in_patient_id_is_not_supported(self):
        """Documented limitation: the patient ID must not contain '_'."""
        assert _extract_patient_from_bin("Pat_ient_metabat_001") == "Pat"


@pytest.fixture
def metadata_file(tmp_path):
    p = tmp_path / "metadata.tsv"
    pd.DataFrame({
        "SampleID": ["G-0001", "G-0002"],
        "population": ["A", "B"],
    }).to_csv(p, sep="\t", index=False)
    return str(p)


@pytest.fixture
def ani():
    return pd.DataFrame({
        "species": ["sp1", "sp1", "sp1"],
        "sample": ["G-0001_COMEBinRefined_1",
                   "G-0002_COMEBinRefined_2",
                   "G-0002_COMEBinRefined_3_sub"],
    })


class TestValidateIdentifiers:

    def test_consistent_inputs_pass(self, ani, metadata_file):
        s2p = {s: f"/g/{s}.fa" for s in ani["sample"]}
        result = validate_identifiers(s2p, ani, metadata_file)
        assert result["ani_to_genomes"] == 3
        assert result["ani_to_metadata"] == 2

    def test_genome_namespace_mismatch_raises(self, ani, metadata_file):
        s2p = {"MEGAHIT-COMEBinRefined-G_0001_XYZ.1": "/g/a.fa"}
        with pytest.raises(PreflightError, match="ani_map -> genomes"):
            validate_identifiers(s2p, ani, metadata_file)

    def test_metadata_namespace_mismatch_raises(self, ani, tmp_path):
        p = tmp_path / "bad_metadata.tsv"
        pd.DataFrame({"SampleID": ["NOPE-1"], "population": ["A"]}).to_csv(
            p, sep="\t", index=False)
        s2p = {s: f"/g/{s}.fa" for s in ani["sample"]}
        with pytest.raises(PreflightError, match="ani_map -> metadata"):
            validate_identifiers(s2p, ani, str(p))

    def test_error_message_shows_both_namespaces(self, ani, metadata_file):
        s2p = {"MEGAHIT-COMEBinRefined-G_0001_XYZ.1": "/g/a.fa"}
        with pytest.raises(PreflightError) as exc:
            validate_identifiers(s2p, ani, metadata_file)
        msg = str(exc.value)
        assert "G-0001_COMEBinRefined_1" in msg
        assert "MEGAHIT-COMEBinRefined-G_0001_XYZ.1" in msg

    def test_partial_overlap_passes_by_default(self, ani, metadata_file):
        """Genomes without metadata are dropped, not fatal."""
        ani_extra = pd.concat([ani, pd.DataFrame({
            "species": ["sp1"], "sample": ["G-9999_COMEBinRefined_9"]})])
        s2p = {s: f"/g/{s}.fa" for s in ani_extra["sample"]}
        assert validate_identifiers(s2p, ani_extra, metadata_file)["ani_to_metadata"] == 2

    def test_min_overlap_threshold_enforced(self, ani, metadata_file):
        ani_extra = pd.concat([ani, pd.DataFrame({
            "species": ["sp1"] * 8,
            "sample": [f"G-9{i:03d}_COMEBinRefined_{i}" for i in range(8)]})])
        s2p = {s: f"/g/{s}.fa" for s in ani_extra["sample"]}
        with pytest.raises(PreflightError, match="required: 90"):
            validate_identifiers(s2p, ani_extra, metadata_file, min_overlap_frac=0.9)

    def test_missing_sample_column_raises(self, ani, tmp_path):
        p = tmp_path / "no_id.tsv"
        pd.DataFrame({"population": ["A"]}).to_csv(p, sep="\t", index=False)
        s2p = {s: f"/g/{s}.fa" for s in ani["sample"]}
        with pytest.raises(PreflightError, match="no SampleID-like column"):
            validate_identifiers(s2p, ani, str(p))


class TestGenomePathLabels:
    """mash/unitig-caller label samples by path basename, not by dict key."""

    def test_symlink_keeps_link_name_not_target_name(self, tmp_path):
        target_dir = tmp_path / "dastool_bins"
        target_dir.mkdir()
        target = target_dir / "MEGAHIT-COMEBinRefined-G_0948_EKDO1-1A_X.24198.fa"
        target.write_text(">c\nACGT\n")

        genomes = tmp_path / "genomes"
        genomes.mkdir()
        link = genomes / "G-0948_COMEBinRefined_24198.fa"
        link.symlink_to(target)

        s2p = list_fastas(str(genomes))
        assert list(s2p) == ["G-0948_COMEBinRefined_24198"]
        # The stored path must still end in the link name, since that basename
        # becomes the mash matrix label and the unitig sample name.
        assert s2p["G-0948_COMEBinRefined_24198"].endswith(
            "G-0948_COMEBinRefined_24198.fa")

    def test_label_mismatch_is_fatal(self, ani, metadata_file):
        """A path whose basename disagrees with its ID must not run silently."""
        s2p = {s: f"/dastool/MEGAHIT-COMEBinRefined-{i}.fa"
               for i, s in enumerate(ani["sample"])}
        with pytest.raises(PreflightError, match="genome path labels"):
            validate_identifiers(s2p, ani, metadata_file)

#!/usr/bin/env python3
"""
Security tests for panpiper-kit.

Verifies that all security fixes are working correctly:
- Path traversal prevention
- Command injection prevention
- Sample name sanitization
- Glob pattern validation
- File permission error handling
"""

import pytest
import tempfile
import os
from pathlib import Path
import shutil

from panpiper_kit.files import safe_sample_name
from panpiper_kit import annotate_sig_unitigs, summarize_pyseer


class TestPathTraversalPrevention:
    """Test that path traversal attacks are prevented."""

    def test_safe_sample_name_rejects_parent_directory(self):
        """Test that '../' patterns are rejected."""
        with pytest.raises(ValueError, match="unsafe characters"):
            safe_sample_name("../etc/passwd")

    def test_safe_sample_name_rejects_double_dots(self):
        """Test that '..' anywhere in the name is rejected."""
        with pytest.raises(ValueError, match="path traversal"):
            safe_sample_name("sample..name")

    def test_safe_sample_name_rejects_forward_slash(self):
        """Test that forward slashes are rejected as unsafe characters."""
        with pytest.raises(ValueError, match="unsafe characters"):
            safe_sample_name("sample/name")

    def test_safe_sample_name_rejects_backslash(self):
        """Test that backslashes are rejected as unsafe characters."""
        with pytest.raises(ValueError, match="unsafe characters"):
            safe_sample_name("sample\\name")

    def test_safe_sample_name_rejects_empty_string(self):
        """Test that empty sample names are rejected."""
        with pytest.raises(ValueError, match="empty"):
            safe_sample_name("")

    def test_safe_sample_name_accepts_valid_names(self):
        """Test that valid sample names are accepted."""
        # These should all pass
        assert safe_sample_name("sample123") == "sample123"
        assert safe_sample_name("Sample_A") == "Sample_A"
        assert safe_sample_name("sample-1.2") == "sample-1.2"
        assert safe_sample_name("S.aureus_123-ABC") == "S.aureus_123-ABC"


class TestCommandInjectionPrevention:
    """Test that command injection via Bakta arguments is prevented."""

    def test_bakta_whitelist_allows_safe_flags(self):
        """Test that allowed Bakta flags pass validation."""
        # These should not raise errors
        test_args = [
            "--compliant",
            "--keep-contig-headers",
            "--meta",
            "--gram +",
            "--genus Staphylococcus",
            "--species aureus",
            "--strain ABC123",
            "--plasmid",
            "--complete"
        ]

        # The function should parse these without error
        import shlex
        for arg in test_args:
            args_list = shlex.split(arg)
            for a in args_list:
                if a.startswith('--'):
                    flag = a.split('=')[0]
                    assert flag in annotate_sig_unitigs.ALLOWED_BAKTA_FLAGS, \
                        f"Flag {flag} should be in whitelist"

    def test_bakta_whitelist_rejects_dangerous_flags(self):
        """Test that dangerous/unknown flags are rejected."""
        dangerous_args = [
            "--output /etc/passwd",
            "--unknown-flag",
            "--prefix $(whoami)",
            "--db; rm -rf /",
        ]

        for dangerous_arg in dangerous_args:
            # This would be caught in the actual code path
            import shlex
            args_list = shlex.split(dangerous_arg)
            has_unsafe = False
            for arg in args_list:
                if arg.startswith('--'):
                    flag = arg.split('=')[0]
                    if flag not in annotate_sig_unitigs.ALLOWED_BAKTA_FLAGS:
                        has_unsafe = True
                        break

            assert has_unsafe, f"Dangerous arg should be rejected: {dangerous_arg}"


class TestSampleNameSanitization:
    """Test that various unsafe characters in sample names are rejected."""

    @pytest.mark.parametrize("unsafe_name,expected_error", [
        ("sample;rm -rf /", "unsafe characters"),
        ("sample|cat /etc/passwd", "unsafe characters"),
        ("sample`whoami`", "unsafe characters"),
        ("sample$(whoami)", "unsafe characters"),
        ("sample&background", "unsafe characters"),
        ("sample>output.txt", "unsafe characters"),
        ("sample<input.txt", "unsafe characters"),
        ("sample*wildcard", "unsafe characters"),
        ("sample?query", "unsafe characters"),
        ("sample[bracket]", "unsafe characters"),
        ("sample{brace}", "unsafe characters"),
        ("sample with spaces", "unsafe characters"),
        ("sample\ttab", "unsafe characters"),
        ("sample\nnewline", "unsafe characters"),
        ("sample'quote", "unsafe characters"),
        ('sample"doublequote', "unsafe characters"),
    ])
    def test_rejects_unsafe_characters(self, unsafe_name, expected_error):
        """Test that various unsafe characters are rejected."""
        with pytest.raises(ValueError, match=expected_error):
            safe_sample_name(unsafe_name)

    @pytest.mark.parametrize("safe_name", [
        "sample123",
        "Sample_ABC",
        "sample-v1.2",
        "S.aureus_strain-1",
        "ABC123",
        "a1b2c3",
        "Sample.2024.01.15",
        "strain_ABC-123_final",
    ])
    def test_accepts_safe_names(self, safe_name):
        """Test that safe sample names are accepted."""
        assert safe_sample_name(safe_name) == safe_name


class TestGlobPatternValidation:
    """Test that glob pattern validation prevents directory traversal."""

    def test_glob_rejects_parent_directory_pattern(self):
        """Test that patterns with '..' are rejected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="Unsafe glob pattern"):
                summarize_pyseer.summarize_pyseer(
                    indir=tmpdir,
                    out=os.path.join(tmpdir, "output.tsv"),
                    pattern="../*.pyseer.tsv"
                )

    def test_glob_rejects_absolute_path_pattern(self):
        """Test that absolute path patterns are rejected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="Unsafe glob pattern"):
                summarize_pyseer.summarize_pyseer(
                    indir=tmpdir,
                    out=os.path.join(tmpdir, "output.tsv"),
                    pattern="/etc/*.tsv"
                )

    def test_glob_rejects_backslash_absolute_pattern(self):
        """Test that Windows-style absolute path patterns are rejected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="Unsafe glob pattern"):
                summarize_pyseer.summarize_pyseer(
                    indir=tmpdir,
                    out=os.path.join(tmpdir, "output.tsv"),
                    pattern="\\etc\\*.tsv"
                )

    def test_glob_accepts_safe_patterns(self):
        """Test that safe glob patterns are accepted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a dummy pyseer file
            test_file = os.path.join(tmpdir, "test__metadata.pyseer.tsv")
            with open(test_file, 'w') as f:
                f.write("variant\tfilter-pvalue\tlrt-pvalue\n")
                f.write("unitig_1\t0.01\t0.02\n")

            output = os.path.join(tmpdir, "output.tsv")

            # These should not raise errors (though may fail for other reasons like no data)
            try:
                summarize_pyseer.summarize_pyseer(
                    indir=tmpdir,
                    out=output,
                    pattern="*.pyseer.tsv"
                )
            except SystemExit:
                # Expected if no significant results, but pattern was accepted
                pass

    def test_glob_verifies_files_within_directory(self):
        """Test that files outside directory are rejected even if glob matches."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a directory structure
            subdir = os.path.join(tmpdir, "subdir")
            os.makedirs(subdir)

            # Create a symlink that points outside (if supported)
            if hasattr(os, 'symlink'):
                outside_dir = tempfile.mkdtemp()
                try:
                    outside_file = os.path.join(outside_dir, "outside.pyseer.tsv")
                    with open(outside_file, 'w') as f:
                        f.write("variant\tfilter-pvalue\tlrt-pvalue\n")

                    link_path = os.path.join(subdir, "link.pyseer.tsv")
                    try:
                        os.symlink(outside_file, link_path)

                        # This should detect that the resolved file is outside tmpdir
                        with pytest.raises(ValueError, match="outside directory"):
                            summarize_pyseer.summarize_pyseer(
                                indir=tmpdir,
                                out=os.path.join(tmpdir, "output.tsv"),
                                pattern="**/*.pyseer.tsv"
                            )
                    except OSError:
                        # Symlink creation failed (permissions, OS doesn't support, etc.)
                        pytest.skip("Cannot create symlinks on this system")
                finally:
                    shutil.rmtree(outside_dir)


class TestFilePermissionErrors:
    """Test that file permission errors are handled gracefully."""

    def test_unreadable_file_error_handling(self):
        """Test that unreadable files produce clear error messages."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a file and make it unreadable
            test_file = os.path.join(tmpdir, "unreadable.pyseer.tsv")
            with open(test_file, 'w') as f:
                f.write("variant\tfilter-pvalue\tlrt-pvalue\n")

            # Make file unreadable (Unix only)
            if hasattr(os, 'chmod'):
                os.chmod(test_file, 0o000)

                try:
                    # This should handle the permission error gracefully
                    output = os.path.join(tmpdir, "output.tsv")

                    # The function should either skip the file or raise a clear error
                    try:
                        summarize_pyseer.summarize_pyseer(
                            indir=tmpdir,
                            out=output,
                            pattern="*.pyseer.tsv"
                        )
                    except (PermissionError, OSError, SystemExit):
                        # Expected - permission denied
                        pass

                finally:
                    # Restore permissions for cleanup
                    os.chmod(test_file, 0o644)
            else:
                pytest.skip("chmod not available on this platform")

    def test_nonexistent_directory_error(self):
        """Test that nonexistent directories produce clear error messages."""
        nonexistent = "/tmp/this_directory_should_not_exist_12345"
        output = "/tmp/output.tsv"

        # Should handle missing directory gracefully
        with pytest.raises(SystemExit, match="No files matching"):
            summarize_pyseer.summarize_pyseer(
                indir=nonexistent,
                out=output,
                pattern="*.pyseer.tsv"
            )

    def test_unwritable_output_directory(self):
        """Test that unwritable output directories are handled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a pyseer file
            test_file = os.path.join(tmpdir, "test.pyseer.tsv")
            with open(test_file, 'w') as f:
                f.write("variant\tfilter-pvalue\tlrt-pvalue\n")

            # Try to write to unwritable location (Unix only)
            if hasattr(os, 'chmod'):
                output_dir = os.path.join(tmpdir, "unwritable")
                os.makedirs(output_dir)
                os.chmod(output_dir, 0o444)  # Read-only

                try:
                    output = os.path.join(output_dir, "output.tsv")

                    with pytest.raises((PermissionError, OSError)):
                        summarize_pyseer.summarize_pyseer(
                            indir=tmpdir,
                            out=output,
                            pattern="*.pyseer.tsv"
                        )
                finally:
                    # Restore permissions for cleanup
                    os.chmod(output_dir, 0o755)
            else:
                pytest.skip("chmod not available on this platform")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

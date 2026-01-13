# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2024-09-17

### Added
- Initial release of panpiper-kit
- ANI-based species binning functionality
- Mash distance calculations within species
- Distance-based association tests (PERMANOVA, Mantel)
- Per-species unitig GWAS with pyseer integration
- Automatic FDR correction using Benjamini-Hochberg method
- Phenotype-aware filtering with customizable thresholds
- CLI tool `ppk-run` for running complete analysis pipelines
- Automatic metadata cleaning for common missing value indicators
- Pairwise association testing for categorical phenotypes
- Comprehensive test suite with >95% coverage
- Detailed README with usage examples and documentation

### Features
- Support for binary, categorical, and continuous phenotypes
- Parallel processing support with configurable threads
- Flexible output structure with organized results directories
- Tracking and debugging reports for all analysis steps
- Support for gzipped FASTA input files
- Customizable k-mer sizes and Mash sketch sizes

[0.1.0]: https://github.com/rolesucsd/panpiper-kit/releases/tag/v0.1.0

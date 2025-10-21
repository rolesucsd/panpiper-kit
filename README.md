# panpiper-kit

**panpiper-kit** is a Python toolkit + CLI (`ppk-run`) for running **fast within-species population genomics analyses**. It combines:

- **ANI-based species bins** (you supply an ANI map of genomes → species clusters).
- **Mash distances** within each species to capture fine-scale lineage structure.
- **Distance-based association tests** between lineage structure and metadata:
  - **PERMANOVA** for categorical/binary traits (does lineage structure separate cases vs controls?).
  - **Mantel (Spearman)** for continuous traits (do pairwise Mash distances correlate with phenotype differences?).
- **Per-species GWAS with pyseer** on unitigs (from `unitig-caller`):
  - Runs for **binary** and **continuous** phenotypes only.
  - Automatically applies **Benjamini–Hochberg (BH/FDR)** correction.

All inputs and outputs are plain TSVs, designed to be easily piped into downstream stats/plots.

---

## Why use this?

Typical GWAS pipelines (`pyseer` etc.) assume you already know what to test (unitigs, SNPs, accessory genes). But often the first question is simpler:

> *Within a species, do the lineages split by phenotype?*

`panpiper-kit` automates this exploratory step:

- **Step 1**: You give it genomes + an ANI species assignment + metadata.
- **Step 2**: It runs Mash distances within each species and tests whether metadata explains those distances.
- **Step 3**: For phenotypes with variation, it runs a **unitig GWAS per species**, controlling multiple testing.

So you can immediately see which species × phenotypes show meaningful structure, and then drill down on specific unitig hits.

---

## Key features

- **Species-aware**: Never runs all-vs-all Mash; everything is contained within ANI-defined species bins.
- **No arbitrary clusters**: Doesn’t invent “phylogroups.” Lineage effects are measured directly from the Mash distance matrix.
- **Phenotype-aware filtering**:
  - Drops metadata columns with no variation or too few samples.
  - Drops phenotype categories that don’t meet a user-specified minimum count.
  - Continuous phenotypes must have enough unique values (default: ≥6).
- **Dual-layer results**:
  - *Lineage-level* (PERMANOVA/Mantel p-values).
  - *Locus-level* (pyseer unitig GWAS hits).
- **Speed-first**: Uses `mash` + `unitig-caller` + `pyseer` with sensible defaults (`-k 18 -s 5000`, multi-threaded).

---

## Inputs

1. **Genomes**: directory of `.fa/.fna/.fasta[.gz]` files.  
   Basename = bin identifier (supports formats like `{patient}_{binner}_{bin_identifier}` or `{patient}.{id}_{binner}_{bin_identifier}`).
   Must match bin identifiers in ANI map.

2. **Metadata file (`metadata.tsv`)**:  
   - Tab-delimited, must contain `SampleID` column + any number of phenotype columns.
   - Phenotypes can be binary, categorical, or continuous.
   - SampleID contains patient names that will be matched to bin identifiers.
   - **Automatic cleaning**: Common missing value indicators like "Not collected", "Unknown", "N/A", etc. are automatically converted to NaN.

   Example:
   ```
   #SampleID    status    age    country
   Patient1     case      34     US
   Patient2     control   29     Not collected
   Patient3     case      37     Unknown
   Patient4     control   31     N/A
   ```

3. **ANI map (`ani_map.tsv`)**:  
   - Tab-delimited, two columns: `species<TAB>bin_identifier`
   - Defines the species bin for each bin identifier.
   - Bin identifiers should match FASTA file basenames.

   Example:
   ```
   spA    Patient1_metabat_001
   spA    Patient1_metabat_002
   spA    10317.X00179178_CONCOCT_bin.40
   spA    10317.X00179178_CONCOCT_bin.41
   ```

---

## Outputs

The main output directory (`--out`) contains the following organized structure:

### Main Results
- `assoc/mash_lineage_assoc_by_species.fdr.tsv`  
  → Combined results: one row per (species × phenotype) with test type, stat, raw p-value, BH q-value.

- `assoc/pyseer_summary.tsv`  
  → Summary of significant pyseer hits across all species with FDR correction.

### Per-Species Results (`assoc_by_species/`)
- `<species>__<phenotype>.dist_assoc.tsv`  
  → Raw PERMANOVA/Mantel output for a specific phenotype.

- `<species>__<phenotype>.pyseer.tsv`  
  → Raw unitig GWAS results for binary/continuous phenotypes.

- `<species>__<phenotype>.pyseer.fdr.tsv`  
  → Unitig GWAS results with BH/FDR correction applied.

- `<species>.pairwise_assoc.tsv`  
  → Combined pairwise PERMANOVA results for categorical phenotypes with significant global association.

- `<species>.pairwise.pyseer.tsv`  
  → Combined pairwise pyseer results for significant categorical comparisons.

### Individual Pairwise Comparisons (`assoc_by_species/<species>/pairwise/`)
For categorical phenotypes with significant global association, individual Pyseer result files are created in species-specific subdirectories:

- `<species>/pairwise/<species>__<variable>__<group>_vs_rest.pyseer.tsv`  
  → Pyseer results for one-vs-rest comparison (if sample size sufficient).

- `<species>/pairwise/<species>__<variable>__<group1>_vs_<group2>.pyseer.tsv`  
  → Pyseer results for group-vs-group comparison (if sample size sufficient).

*Note: Binary phenotype files (`.binary.tsv`) are created temporarily for Pyseer input but are not permanently saved.*

### Intermediate Files (`mash_by_species/`)
- `<species>/mash.tsv`  
  → Mash distance matrix for the species.

- `<species>/refs.txt`  
  → List of FASTA file paths used for unitig calling.

### Unitig Files (`unitigs_by_species/`)
- `<species>/uc.pyseer`  
  → Pyseer-ready unitig presence/absence matrix per species (reusable).

- `<species>/unitig_caller.log`  
  → Log from unitig-caller execution.

### Tracking and Debugging
- `expected_pyseer_runs.tsv`  
  → List of all expected Pyseer runs based on phenotype data.

- `pyseer_tracking_report.tsv`  
  → Detailed tracking of all Pyseer runs with status (completed, failed, skipped) and error messages.

- `pyseer_tracking_report_summary.tsv`  
  → Summary statistics of Pyseer run outcomes.

### Temporary Files (`tmp/`)
- `phenos/<species>.list.tsv`  
  → Manifest of phenotype files for each species.

- `phenos/<species>__<variable>.pheno.tsv`  
  → Cleaned phenotype files ready for analysis.

- `phenos/<species>.pheno_summary.tsv`  
  → Detailed summary of phenotype processing for each species.

---

## Installation

We recommend using `mamba` or `conda` to install external dependencies (`mash`, `unitig-caller`, `pyseer`).

```bash
# 1. Create env from the example spec
mamba env create -f examples/env.yaml
mamba activate panpiper-kit

# 2. Install panpiper-kit
pip install -e .
```

This gives you the CLI `ppk-run`.

---

## Usage

Basic run:

```bash
ppk-run   --genomes genomes/   --metadata metadata.tsv   --ani-map ani_map.tsv   --out out_ppk   --threads 32 --perms 999   --mash-k 18 --mash-s 5000   --min-n 6 --max-missing-frac 0.2 --min-level-n 3 --min-unique-cont 6   --kmer 31
```

### Key parameters

- `--min-n` : minimum usable samples per species (default 6).
- `--max-missing-frac` : max fraction of missing values allowed for a phenotype (default 0.2).
- `--min-level-n` : min samples required per category level (default 3).
- `--min-unique-cont` : min unique values required for continuous phenotypes (default 6).
- `--perms` : number of permutations for PERMANOVA/Mantel (default 999).
- `--kmer` : unitig k-mer size (default 31).
- `--missing-values` : additional missing value indicators to treat as NaN (space-separated).

### Metadata cleaning

The toolkit automatically converts common missing value indicators to NaN to prevent spurious categories:

**Default missing indicators** (case-insensitive):
- `not collected`, `not available`, `unknown`
- `n/a`, `na`, `none`, `missing`
- `not specified`, `not provided`, `not reported`
- `not applicable`, `not determined`, `not tested`, `not done`
- `pending`, `tbd`, `to be determined`
- `not recorded`, `not documented`, `not assessed`, `not evaluated`

**Custom missing values**:
```bash
ppk-run --missing-values "custom_value1" "custom_value2" --genomes genomes/ --metadata metadata.tsv --ani-map ani_map.tsv --out results/
```

---

## Caveats

- **PERMANOVA** can give significant p-values just from unequal variance across groups. Check dispersion (`betadisper`) if in doubt.
- **Mantel** is conservative; low power for continuous traits. Consider dbRDA if you need more.
- **pyseer** here is run with `--no-distances` (no structure correction) for speed. If lineage confounding is a concern, rerun with `--distances`.
- **Mash** is O(N²) per species. With thousands of genomes in one species, consider approximate clustering (e.g. Dashing).

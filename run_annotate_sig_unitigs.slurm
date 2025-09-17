#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH -c 32                        # total CPUs to allocate
#SBATCH --mem=128G                   # <-- EDIT
#SBATCH -t 24:00:00                  # <-- EDIT
#SBATCH -o annotate_sig_unitigs.out.%j
#SBATCH -e annotate_sig_unitigs.err.%j

set -euo pipefail

########## USER SETTINGS ##########
# Inputs (your paths)
PYSEER_TSV="input/10317.0363182635_CONCOCT_bin.26.pairwise.pyseer.tsv"
UNITIG_MAP="input/10317.0363182635_CONCOCT_bin.26/uc.pyseer"
FASTA_DIR="input/fastas"             # Directory with per-sample FASTA files
FASTA_PATTERN="{sample}.fasta"      # Pattern for FASTA files (supports .gz)

# Annotation settings
ANNO_DIR="input/bakta_annotations"  # Directory with existing Bakta TSVs (optional)
ANNO_PATTERN="{sample}/{sample}.tsv" # Pattern for annotation files

# Bakta settings (if running Bakta)
BAKTA_RUN=true                      # Set to false if you have pre-computed Bakta annotations
BAKTA_DB="/path/to/bakta/db"        # Path to Bakta database
BAKTA_OUT_DIR="output/bakta"        # Where to put Bakta outputs
# Bakta prefix is automatically set to sample name
BAKTA_CORES=8                       # Threads per Bakta run

# Output
OUT_PREFIX="output/10317.0363182635_CONCOCT_bin.26"

# Analysis settings
Q_THRESH=0.01                       # FDR threshold
ALLOW_REVCOMP=false                 # Search reverse complement too
WORKERS=6                           # Parallel workers for sample processing

# Conda env to use (must have panpiper-kit + bakta installed)
CONDA_ENV="panpiper-kit"            # conda/mamba env name
###################################

echo "[annotate_sig_unitigs] starting at $(date)"
mkdir -p "$(dirname "${OUT_PREFIX}")"
mkdir -p "${BAKTA_OUT_DIR}"

# --- Sanity checks ---
for f in "${PYSEER_TSV}" "${UNITIG_MAP}"; do
  [[ -s "$f" ]] || { echo "[FATAL] Missing input file: $f" >&2; exit 1; }
done
[[ -d "${FASTA_DIR}" ]] || { echo "[FATAL] FASTA dir not found: ${FASTA_DIR}" >&2; exit 1; }

if [[ "${BAKTA_RUN}" == "true" ]]; then
  [[ -d "${BAKTA_DB}" ]] || { echo "[FATAL] Bakta DB not found: ${BAKTA_DB}" >&2; exit 1; }
fi

# --- Activate conda environment ---
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"

# --- Set thread limits for external tools ---
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

# --- Build command ---
CMD="python -m panpiper_kit.annotate_sig_unitigs"
CMD="${CMD} --pyseer '${PYSEER_TSV}'"
CMD="${CMD} --unitig-map '${UNITIG_MAP}'"
CMD="${CMD} --fasta-dir '${FASTA_DIR}'"
CMD="${CMD} --fasta-pattern '${FASTA_PATTERN}'"
CMD="${CMD} --out-prefix '${OUT_PREFIX}'"
CMD="${CMD} --q-thresh ${Q_THRESH}"
CMD="${CMD} --workers ${WORKERS}"

# Add annotation directory if provided
if [[ -n "${ANNO_DIR:-}" ]]; then
  CMD="${CMD} --anno-dir '${ANNO_DIR}'"
  CMD="${CMD} --anno-pattern '${ANNO_PATTERN}'"
fi

# Add Bakta options if running Bakta
if [[ "${BAKTA_RUN}" == "true" ]]; then
  CMD="${CMD} --bakta-run"
  CMD="${CMD} --bakta-db '${BAKTA_DB}'"
  CMD="${CMD} --bakta-out-dir '${BAKTA_OUT_DIR}'"
  CMD="${CMD} --bakta-prefix-pattern '${BAKTA_PREFIX_PATTERN}'"
  CMD="${CMD} --bakta-cores ${BAKTA_CORES}"
fi

# Add reverse complement option
if [[ "${ALLOW_REVCOMP}" == "true" ]]; then
  CMD="${CMD} --allow-revcomp"
fi

# --- Run the command ---
echo "[annotate_sig_unitigs] running: ${CMD}"
eval "${CMD}" 2>&1 | tee "${OUT_PREFIX}_annotate_sig_unitigs.log"

echo "[annotate_sig_unitigs] finished at $(date)"
echo "[annotate_sig_unitigs] outputs:"
echo "  - ${OUT_PREFIX}_long.tsv (detailed annotations per unitig hit)"
echo "  - ${OUT_PREFIX}_summary.tsv (summary per unitig)"
echo "  - ${OUT_PREFIX}_annotate_sig_unitigs.log (run log)"

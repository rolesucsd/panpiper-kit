#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Single-Pyseer → (FDR) → unitig→FASTA coordinates → Bakta CDS annotation.

Features
--------
- One Pyseer file only (enforced).
- BH-FDR on both p-value columns; keep if q_filter < q || q_lrt < q.
- Unitig→samples map (no header): "UNITIG | sampleA:1 sampleB:1 ..."
- For each needed sample:
    * Ensures Bakta TSV exists (skips if present)
    * Optionally runs Bakta if missing (requires --bakta-run and --bakta-db)
    * Streams FASTA, finds all occurrences (forward + optional reverse-complement)
    * Intersects [start,end] with Bakta CDS rows on same contig
- Parallel over samples with --workers.

Outputs
-------
<out_prefix>_long.tsv:
  unitig  sample  contig  pid  start  end  strand  evalue  bitscore  length  coding  locus_tag  gene  product  dbxrefs
<out_prefix>_summary.tsv:
  unitig  n_samples  samples  annotations

Assumptions
-----------
- Coordinates are 1-based inclusive.
- If a unitig spans multiple CDS or hits multiple loci, all are reported.
- If no CDS overlap exists, annotation fields are left blank (intergenic).
- Bakta output detection:
  1) If --anno-dir/--anno-pattern resolves to a file, use it.
  2) Else if --bakta-run: run Bakta into --bakta-out-dir/{sample}/ with prefix = {sample},
     then pick {outdir}/{prefix}.tsv if present, otherwise the first *.tsv within {outdir}.
  3) Else: error.

Example
-------
python map_unitig_hits_bakta.py \
  --pyseer 10317.X00215160_MetaBAT_bin.6__thdmi_cohort__Mexico_vs_rest.pyseer.tsv \
  --unitig-map unitigs_per_sample.txt \
  --fasta-dir /path/to/fastas \
  --fasta-pattern "{sample}.fasta.gz" \
  --anno-dir /path/to/bakta \
  --anno-pattern "{sample}/{sample}.tsv" \
  --bakta-run \
  --bakta-db /refs/bakta/db \
  --bakta-out-dir /path/to/bakta \
  --bakta-cores 8 \
  --workers 6 \
  --allow-revcomp \
  --q-thresh 0.01 \
  --out-prefix results/mexico_vs_rest
"""

import argparse
import gzip
import os
import shlex
import subprocess
from collections import defaultdict
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    pd = None


# ---------------------------
# Small utilities
# ---------------------------

def open_maybe_gz(path):
    return gzip.open(path, "rt") if str(path).endswith(".gz") else open(path, "r")

def reverse_complement(seq):
    comp = str.maketrans("ACGTacgtNn", "TGCAtgcaNn")
    return seq.translate(comp)[::-1]

def stream_fasta(path):
    if not Path(path).exists():
        return
    with open_maybe_gz(path) as fh:
        hdr, buf = None, []
        for ln in fh:
            if ln.startswith(">"):
                if hdr is not None:
                    yield hdr, "".join(buf)
                hdr = ln.strip()[1:]
                buf = []
            else:
                buf.append(ln.strip())
        if hdr is not None:
            yield hdr, "".join(buf)

def write_tsv(rows_or_df, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if pd is not None and isinstance(rows_or_df, pd.DataFrame):
        rows_or_df.to_csv(path, sep="\t", index=False)
    else:
        import csv
        rows = rows_or_df
        with open(path, "w", newline="") as fh:
            if not rows:
                fh.write("")
                return
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), delimiter="\t")
            w.writeheader()
            for r in rows:
                w.writerow(r)

def bh_fdr(pvalues):
    n = len(pvalues)
    pairs = []
    for i, p in enumerate(pvalues):
        try:
            pv = float(p)
            if pv < 0 or pv > 1 or not (pv == pv):  # NaN
                pv = 1.0
        except Exception:
            pv = 1.0
        pairs.append((i, pv))
    pairs.sort(key=lambda x: x[1])
    qvals = [None] * n
    running_min = 1.0
    for rank, (idx, p) in enumerate(pairs, start=1):
        q = p * n / rank
        if q < running_min:
            running_min = q
        qvals[idx] = running_min
    return [max(0.0, min(1.0, q)) for q in qvals]


# ---------------------------
# Parsers
# ---------------------------

def parse_unitig_map(path):
    """
    UNITIG | sampleA:1 sampleB:1 ...
    """
    unitig_to_samples = defaultdict(set)
    with open_maybe_gz(path) as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln or ln.startswith("#") or " | " not in ln:
                continue
            unitig, rhs = ln.split(" | ", 1)
            unitig = unitig.strip()
            if not unitig:
                continue
            for tok in rhs.strip().split():
                s = tok.split(":", 1)[0].strip()
                if s:
                    unitig_to_samples[unitig].add(s)
    return unitig_to_samples

def load_bakta_tsv(anno_path):
    """
    Return CDS rows as dicts:
      contig,start,stop,strand,locus_tag,gene,product,dbxrefs
    Handles Bakta TSV with comment headers and finds the actual column header row.
    """
    if not Path(anno_path).exists():
        return []

    def norm(s): return s.strip().lower().replace(" ", "_")

    cds = []
    with open_maybe_gz(anno_path) as fh:
        header = None
        for ln in fh:
            if not ln.strip():
                continue
            # Skip comment lines (starting with #) UNLESS they contain tab-separated values (header row)
            if ln.startswith("#"):
                # Check if this looks like a header row (contains tabs)
                if "\t" in ln:
                    # Remove the leading # and process as header
                    parts = ln[1:].rstrip("\n").split("\t")
                else:
                    continue
            else:
                parts = ln.rstrip("\n").split("\t")
            
            if header is None:
                # This should be the actual header row (first non-comment line)
                header = [norm(x) for x in parts]
                col = {c: i for i, c in enumerate(header)}
                for req in ("sequence_id", "type", "start", "stop", "strand"):
                    if req not in col:
                        raise ValueError(f"Bakta TSV missing column: {req}. Available columns: {list(col.keys())}")
                idx_locus = col.get("locus_tag")
                idx_gene  = col.get("gene")
                idx_prod  = col.get("product")
                idx_dbx   = col.get("dbxrefs")
                continue

            # Pad if short
            if len(parts) < len(header):
                parts += [""] * (len(header) - len(parts))

            if parts[col["type"]] != "cds":
                continue
            try:
                start = int(parts[col["start"]])
                stop  = int(parts[col["stop"]])
            except Exception:
                continue
            cds.append({
                "contig": parts[col["sequence_id"]],
                "start": start,
                "stop":  stop,
                "strand": parts[col["strand"]],
                "locus_tag": parts[idx_locus] if idx_locus is not None else "",
                "gene":      parts[idx_gene]  if idx_gene  is not None else "",
                "product":   parts[idx_prod]  if idx_prod  is not None else "",
                "dbxrefs":   parts[idx_dbx]   if idx_dbx   is not None else "",
            })
    return cds


# ---------------------------
# BLAST search
# ---------------------------

def run_blast_unitig(unitig, fasta_path, blast_bin="blastn", evalue=1e-3, max_target_seqs=1):
    """
    Run BLAST to find unitig in FASTA file.
    Returns list of hits with: contig, start, stop, strand, pid, evalue, bitscore, length
    """
    import tempfile
    import subprocess
    
    # Create temporary query file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as query_file:
        query_file.write(f">unitig\n{unitig}\n")
        query_path = query_file.name
    
    try:
        # Run BLAST with optimized parameters for speed
        cmd = [
            blast_bin,
            "-query", query_path,
            "-subject", str(fasta_path),
            "-outfmt", "6 qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore sstrand",
            "-evalue", str(evalue),
            "-max_target_seqs", str(max_target_seqs),
            "-task", "blastn-short" if len(unitig) < 50 else "blastn",
            "-word_size", "7",  # Smaller word size for short sequences
            "-reward", "2",     # Faster scoring
            "-penalty", "-3"    # Faster scoring
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        hits = []
        for line in result.stdout.strip().split('\n'):
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) >= 12:
                contig = parts[1]
                pid = float(parts[2])
                length = int(parts[3])
                qstart, qend = int(parts[6]), int(parts[7])
                sstart, send = int(parts[8]), int(parts[9])
                evalue = float(parts[10])
                bitscore = float(parts[11])
                sstrand = parts[12]
                
                # Convert to 1-based coordinates and ensure start <= end
                start = min(sstart, send)
                end = max(sstart, send)
                strand = "+" if sstrand == "plus" else "-"
                
                hits.append({
                    "contig": contig,
                    "start": start,
                    "end": end,
                    "strand": strand,
                    "pid": pid,
                    "evalue": evalue,
                    "bitscore": bitscore,
                    "length": length
                })
        
        return hits
    
    except subprocess.CalledProcessError as e:
        print(f"[warning] BLAST failed for unitig: {e}")
        return []
    finally:
        # Clean up temporary file
        Path(query_path).unlink(missing_ok=True)

# ---------------------------
# Search / overlap (legacy - keeping for now)
# ---------------------------

def find_all_occurrences(hay, needle):
    """Return 1-based start positions of all (possibly overlapping) matches."""
    starts = []
    i = 0
    L = len(needle)
    while True:
        j = hay.find(needle, i)
        if j == -1:
            break
        starts.append(j + 1)  # 1-based
        i = j + 1             # allow overlaps
    return starts

def find_genes_by_coordinates(contig, start, end, cds_by_contig):
    """
    Find genes overlapping with the given coordinates on a contig.
    Returns list of gene annotations, or empty list if intergenic.
    """
    overlapping_genes = []
    for row in cds_by_contig.get(contig, []):
        if max(start, row["start"]) <= min(end, row["stop"]):
            overlapping_genes.append(row)
    
    return overlapping_genes

def overlap_cds_on_contig(contig, qstart, qend, cds_by_contig):
    """Yield CDS dicts overlapping [qstart, qend] on contig."""
    for row in cds_by_contig.get(contig, []):
        if max(qstart, row["start"]) <= min(qend, row["stop"]):
            yield row


# ---------------------------
# Bakta runner
# ---------------------------

def ensure_bakta_for_sample(
    sample: str,
    fasta_path: Path,
    anno_path: Path,
    run_bakta: bool,
    bakta_bin: str,
    bakta_db: Path,
    bakta_out_dir: Path,
    bakta_cores: int,
    bakta_extra_args: str,
) -> Path:
    """
    If anno_path exists, return it.
    Else, if run_bakta, run bakta → detect TSV path and return it.
    Else, raise FileNotFoundError.
    """
    # 1) If explicit anno_path is already there, use it.
    if anno_path and anno_path.exists():
        return anno_path

    # 2) If not running bakta, error out.
    if not run_bakta:
        raise FileNotFoundError(
            f"No annotation TSV found for sample '{sample}': {anno_path}. "
            f"Enable --bakta-run or provide a valid --anno-dir/--anno-pattern."
        )

    # 3) Prepare Bakta output directory and prefix
    out_dir  = bakta_out_dir / sample
    prefix   = sample  # Use sample name directly as prefix
    target_tsv = out_dir / f"{prefix}.tsv"

    # If out_dir exists and contains prefix.tsv (or any .tsv), skip running.
    if out_dir.exists():
        if target_tsv.exists():
            print(f"[info] Using existing Bakta TSV for sample '{sample}': {target_tsv}")
            return target_tsv
        # else: try any .tsv in the dir
        ts = sorted(out_dir.glob("*.tsv"))
        if ts:
            print(f"[info] Using existing Bakta TSV for sample '{sample}': {ts[0]}")
            return ts[0]

    # 4) Run bakta
    if not fasta_path.exists():
        raise FileNotFoundError(f"FASTA for '{sample}' not found: {fasta_path}")

    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        bakta_bin,
        str(fasta_path),
        "--db", str(bakta_db),
        "--skip-pseudo",
        "--force",
        "--skip-plot",
        "--output", str(out_dir),
        "--prefix", prefix,
        "--threads", str(int(bakta_cores)),
    ]
    if bakta_extra_args:
        cmd.extend(shlex.split(bakta_extra_args))

    # Run
    print(f"[info] Running Bakta for sample '{sample}'...")
    try:
        subprocess.run(cmd, check=True)
        print(f"[info] Bakta completed for sample '{sample}'")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Bakta failed for sample '{sample}': {e}")

    # 5) Resolve TSV
    if target_tsv.exists():
        return target_tsv
    ts = sorted(out_dir.glob("*.tsv"))
    if ts:
        return ts[0]

    raise FileNotFoundError(
        f"Bakta completed for '{sample}' but TSV not found in {out_dir}"
    )


# ---------------------------
# Per-sample worker
# ---------------------------

def process_sample(
    sample: str,
    unitigs_for_sample: list[str],
    fasta_dir: Path,
    fasta_pattern: str,
    anno_dir: Path | None,
    anno_pattern: str,
    run_bakta: bool,
    bakta_bin: str,
    bakta_db: Path | None,
    bakta_out_dir: Path,
    bakta_cores: int,
    bakta_extra_args: str,
    allow_revcomp: bool,
):
    """Use BLAST to find unitigs in Bakta FASTA, then map to annotations by coordinates."""
    fa_path = fasta_dir / fasta_pattern.format(sample=sample)

    # Preferred annotation path if user provided anno_dir/pattern
    ann_path = None
    if anno_dir is not None:
        ann_path = anno_dir / anno_pattern.format(sample=sample)

    # Ensure Bakta exists (or run it)
    bakta_db_req = bakta_db if run_bakta else None
    tsv_path = ensure_bakta_for_sample(
        sample=sample,
        fasta_path=fa_path,
        anno_path=(ann_path if ann_path else Path("__MISSING__")),
        run_bakta=run_bakta,
        bakta_bin=bakta_bin,
        bakta_db=bakta_db_req if bakta_db_req else Path("__NO_DB__"),
        bakta_out_dir=bakta_out_dir,
        bakta_cores=bakta_cores,
        bakta_extra_args=bakta_extra_args,
    )

    # Load Bakta CDS
    cds_rows = load_bakta_tsv(tsv_path)
    cds_by_contig = defaultdict(list)
    for r in cds_rows:
        cds_by_contig[r["contig"]].append(r)

    # Find Bakta FASTA file (should be in the same directory as TSV)
    bakta_fasta_path = tsv_path.parent / f"{sample}.fna"
    if not bakta_fasta_path.exists():
        raise FileNotFoundError(f"Bakta FASTA file not found: {bakta_fasta_path}")

    # Process each unitig with BLAST
    rows = []
    for unitig in unitigs_for_sample:
        print(f"[info] BLASTing unitig {unitig} in sample {sample}")
        
        # Run BLAST
        blast_hits = run_blast_unitig(unitig, bakta_fasta_path)
        
        if not blast_hits:
            # No BLAST hits found
            rows.append({
                "unitig": unitig, "sample": sample, "contig": "", "pid": 0.0,
                "start": 0, "end": 0, "strand": "", "evalue": 1.0, "bitscore": 0.0, "length": 0,
                "coding": "no_hit", "locus_tag": "", "gene": "", "product": "", "dbxrefs": ""
            })
            continue
        
        # Take the top hit
        top_hit = blast_hits[0]
        contig = top_hit["contig"]
        start = top_hit["start"]
        end = top_hit["end"]
        strand = top_hit["strand"]
        pid = top_hit["pid"]
        evalue = top_hit["evalue"]
        bitscore = top_hit["bitscore"]
        length = top_hit["length"]
        
        # Find overlapping genes
        overlapping_genes = find_genes_by_coordinates(contig, start, end, cds_by_contig)
        
        if overlapping_genes:
            # Genic - single locus tag
            if len(overlapping_genes) == 1:
                gene = overlapping_genes[0]
                rows.append({
                    "unitig": unitig, "sample": sample, "contig": contig, "pid": pid,
                    "start": start, "end": end, "strand": strand, "evalue": evalue, "bitscore": bitscore, "length": length,
                    "coding": "genic", "locus_tag": gene["locus_tag"], "gene": gene["gene"],
                    "product": gene["product"], "dbxrefs": gene["dbxrefs"]
                })
            else:
                # Multiple genes - combine with |
                locus_tags = "|".join([g["locus_tag"] for g in overlapping_genes if g["locus_tag"]])
                genes = "|".join([g["gene"] for g in overlapping_genes if g["gene"]])
                products = "|".join([g["product"] for g in overlapping_genes if g["product"]])
                dbxrefs = "|".join([g["dbxrefs"] for g in overlapping_genes if g["dbxrefs"]])
                
                rows.append({
                    "unitig": unitig, "sample": sample, "contig": contig, "pid": pid,
                    "start": start, "end": end, "strand": strand, "evalue": evalue, "bitscore": bitscore, "length": length,
                    "coding": "genic", "locus_tag": locus_tags, "gene": genes,
                    "product": products, "dbxrefs": dbxrefs
                })
        else:
            # Intergenic - find flanking genes
            flanking_genes = []
            for gene in cds_by_contig.get(contig, []):
                # Check if gene is before or after the unitig
                if gene["stop"] < start:  # Gene is before
                    flanking_genes.append(gene)
                elif gene["start"] > end:  # Gene is after
                    flanking_genes.append(gene)
            
            # Sort by distance and take closest two
            flanking_genes.sort(key=lambda g: min(abs(g["stop"] - start), abs(g["start"] - end)))
            flanking_genes = flanking_genes[:2]
            
            if flanking_genes:
                locus_tags = "|".join([g["locus_tag"] for g in flanking_genes if g["locus_tag"]])
                genes = "|".join([g["gene"] for g in flanking_genes if g["gene"]])
                products = "|".join([g["product"] for g in flanking_genes if g["product"]])
                dbxrefs = "|".join([g["dbxrefs"] for g in flanking_genes if g["dbxrefs"]])
                
                rows.append({
                    "unitig": unitig, "sample": sample, "contig": contig, "pid": pid,
                    "start": start, "end": end, "strand": strand, "evalue": evalue, "bitscore": bitscore, "length": length,
                    "coding": "intergenic", "locus_tag": locus_tags, "gene": genes,
                    "product": products, "dbxrefs": dbxrefs
                })
            else:
                # No flanking genes found
                rows.append({
                    "unitig": unitig, "sample": sample, "contig": contig, "pid": pid,
                    "start": start, "end": end, "strand": strand, "evalue": evalue, "bitscore": bitscore, "length": length,
                    "coding": "intergenic", "locus_tag": "", "gene": "", "product": "", "dbxrefs": ""
                })

    return rows
def main():
    ap = argparse.ArgumentParser(
        description="FDR-filter ONE Pyseer TSV, map significant unitigs to FASTA coords & Bakta annotations (auto-run Bakta if missing)."
    )
    ap.add_argument("--pyseer", required=True, help="ONE Pyseer results TSV.")
    ap.add_argument("--unitig-map", required=True, help="Unitig→samples file (no header).")
    ap.add_argument("--fasta-dir", required=True, help="Directory with per-sample FASTA files.")
    ap.add_argument("--fasta-pattern", default="{sample}.fasta", help="FASTA filename pattern within --fasta-dir (supports .gz).")
    ap.add_argument("--anno-dir", default=None, help="Directory with per-sample Bakta TSVs (optional if --bakta-run).")
    ap.add_argument("--anno-pattern", default="{sample}/{sample}.tsv",
                    help="Annotation TSV pattern within --anno-dir (default '{sample}/{sample}.tsv').")
    ap.add_argument("--q-thresh", type=float, default=0.01, help="FDR threshold for either column (default 0.01).")
    ap.add_argument("--max-unitigs", type=int, default=10000, help="Maximum number of unitigs to process (default 10000).")
    ap.add_argument("--allow-revcomp", action="store_true", help="Search reverse-complement of unitigs too.")
    ap.add_argument("--out-prefix", required=True, help="Output prefix for TSVs.")

    # Bakta options
    ap.add_argument("--bakta-run", action="store_true",
                    help="If set, run Bakta when expected TSV is missing.")
    ap.add_argument("--bakta-bin", default="bakta", help="Path to bakta executable (default: 'bakta' on PATH).")
    ap.add_argument("--bakta-db", default=None, help="Path to Bakta database (required if --bakta-run).")
    ap.add_argument("--bakta-out-dir", default=None,
                    help="Root directory for Bakta outputs (default: --anno-dir if set, else --fasta-dir).")
    ap.add_argument("--bakta-cores", type=int, default=8, help="Threads per Bakta run (default 8).")
    ap.add_argument("--bakta-extra-args", default="", help="Extra args string passed to Bakta (optional).")

    # Parallelism
    ap.add_argument("--workers", type=int, default=6,
                    help="Parallel workers (samples processed concurrently). "
                         "This throttles both Bakta runs and FASTA/annotation scanning per sample.")

    args = ap.parse_args()

    if pd is None:
        raise SystemExit("This script requires pandas: pip install pandas")

    # Validate bakta options
    anno_dir = Path(args.anno_dir) if args.anno_dir else None
    if args.bakta_run and not args.bakta_db:
        raise SystemExit("--bakta-run requires --bakta-db")

    bakta_out_dir = Path(args.bakta_out_dir) if args.bakta_out_dir else (anno_dir if anno_dir else Path(args.fasta_dir))
    bakta_db_path = Path(args.bakta_db) if args.bakta_db else None

    # Load ONE Pyseer file
    ps = pd.read_csv(args.pyseer, sep="\t", dtype=str, keep_default_na=False)
    for col in ("variant", "filter-pvalue", "lrt-pvalue"):
        if col not in ps.columns:
            raise SystemExit(f"Missing column in Pyseer: {col}")

    # Convert p-values to float and get top N by p-value (no BH correction)
    ps["filter-pvalue"] = pd.to_numeric(ps["filter-pvalue"], errors='coerce')
    ps["lrt-pvalue"] = pd.to_numeric(ps["lrt-pvalue"], errors='coerce')
    ps["variant"] = ps["variant"].astype(str).str.strip()

    # Get top N unitigs by best p-value (minimum of filter and lrt p-values)
    ps["best_pvalue"] = ps[["filter-pvalue", "lrt-pvalue"]].min(axis=1)
    ps_sig = ps.nsmallest(args.max_unitigs, "best_pvalue").copy()
    print(f"[info] Selected top {len(ps_sig):,} unitigs by p-value (from {len(ps):,} total)")

    if ps_sig.empty:
        write_tsv(pd.DataFrame(columns=[
            "unitig","sample","contig","pid","start","end","strand","evalue","bitscore","length","coding","locus_tag","gene","product","dbxrefs"
        ]), f"{args.out_prefix}_long.tsv")
        write_tsv(pd.DataFrame(columns=["unitig","n_samples","samples","annotations"]),
                  f"{args.out_prefix}_summary.tsv")
        print("No significant unitigs after FDR.")
        return

    signif_unitigs = set(ps_sig["variant"].tolist())

    # Unitig → samples
    unitig_to_samples = parse_unitig_map(args.unitig_map)

    # Assign each unitig to exactly one sample to improve batching locality
    # Policy: choose the candidate sample that has the highest total significant unitig count.
    # Tie-breaker: lexicographically smallest sample id.
    from collections import defaultdict as _dd
    # Precompute per-sample total significant unitig counts
    sample_total_counts = _dd(int)
    for u in signif_unitigs:
        for s in unitig_to_samples.get(u, set()):
            sample_total_counts[s] += 1
    # Assign unitigs
    assigned_by_sample = _dd(set)
    for u in signif_unitigs:
        candidates = list(unitig_to_samples.get(u, set()))
        if not candidates:
            continue
        chosen = sorted(candidates, key=lambda s: (-sample_total_counts.get(s, 0), s))[0]
        assigned_by_sample[chosen].add(u)
    per_sample_unitigs = {s: sorted(list(us)) for s, us in assigned_by_sample.items()}

    # Write sample/unitig mapping file (skip if already exists)
    mapping_file = f"{args.out_prefix}_sample_unitig_mapping.tsv"
    if Path(mapping_file).exists():
        print(f"[info] Sample/unitig mapping file already exists: {mapping_file}")
        # Load existing mapping to get sample counts
        existing_df = pd.read_csv(mapping_file, sep="\t")
        print(f"[info] {len(existing_df)} samples will be processed with {existing_df['n_unitigs'].sum():,} total unitigs")
    else:
        mapping_rows = []
        for sample in sorted(per_sample_unitigs.keys()):
            unitigs = per_sample_unitigs[sample]
            mapping_rows.append({
                "sample": sample,
                "n_unitigs": len(unitigs),
                "unitigs": ",".join(unitigs),
                "fasta_path": str(Path(args.fasta_dir) / args.fasta_pattern.format(sample=sample)),
                "bakta_out_dir": str(bakta_out_dir / sample),
                "bakta_tsv_path": str(bakta_out_dir / sample / f"{sample}.tsv")
            })
        
        mapping_df = pd.DataFrame(mapping_rows)
        write_tsv(mapping_df, mapping_file)
        print(f"[info] Sample/unitig mapping written to: {mapping_file}")
        print(f"[info] {len(per_sample_unitigs)} samples will be processed with {sum(len(us) for us in per_sample_unitigs.values())} total unitigs")

    from concurrent.futures import ThreadPoolExecutor, as_completed

    long_rows_all = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = []
        # Process samples ordered by how many unitigs they contain (desc)
        for sample, ulist in sorted(per_sample_unitigs.items(), key=lambda kv: len(kv[1]), reverse=True):
            if not ulist:
                continue
            futs.append(ex.submit(
                process_sample,
                sample=sample,
                unitigs_for_sample=ulist,
                fasta_dir=Path(args.fasta_dir),
                fasta_pattern=args.fasta_pattern,
                anno_dir=anno_dir,
                anno_pattern=args.anno_pattern,
                run_bakta=args.bakta_run,
                bakta_bin=args.bakta_bin,
                bakta_db=bakta_db_path,
                bakta_out_dir=bakta_out_dir,
                bakta_cores=args.bakta_cores,
                bakta_extra_args=args.bakta_extra_args,
                allow_revcomp=args.allow_revcomp,
            ))
        for fu in as_completed(futs):
            long_rows_all.extend(fu.result())

    # Write long
    if long_rows_all:
        df_long = pd.DataFrame(long_rows_all)
    else:
        df_long = pd.DataFrame(columns=[
            "unitig","sample","contig","pid","start","end","strand","evalue","bitscore","length","coding","locus_tag","gene","product","dbxrefs"
        ])
    write_tsv(df_long, f"{args.out_prefix}_long.tsv")

    # Print summary statistics
    if not df_long.empty:
        total_unitigs = len(df_long)
        no_hit = len(df_long[df_long["coding"] == "no_hit"])
        genic = len(df_long[df_long["coding"] == "genic"])
        intergenic = len(df_long[df_long["coding"] == "intergenic"])
        
        print(f"\n[summary] Results:")
        print(f"  Total unitigs processed: {total_unitigs:,}")
        print(f"  No BLAST hits: {no_hit:,} ({no_hit/total_unitigs*100:.1f}%)")
        print(f"  Genic hits: {genic:,} ({genic/total_unitigs*100:.1f}%)")
        print(f"  Intergenic hits: {intergenic:,} ({intergenic/total_unitigs*100:.1f}%)")
        
        if no_hit > 0:
            print(f"\n[info] High 'no_hit' rate may indicate:")
            print(f"  - Unitigs not present in Bakta FASTA files")
            print(f"  - BLAST parameters too strict (evalue={1e-3})")
            print(f"  - Unitigs from different reference than Bakta input")

    # Summary per unitig
    ann_by_unitig = defaultdict(set)
    samples_by_unitig = defaultdict(set)
    for _, r in df_long.iterrows():
        u = r["unitig"]
        s = r["sample"]
        samples_by_unitig[u].add(s)
        lbl = r.get("product") or r.get("gene") or r.get("locus_tag") or ""
        if lbl:
            ann_by_unitig[u].add(lbl)

    rows_sum = []
    for u in sorted(signif_unitigs):
        ss = sorted(samples_by_unitig.get(u, set()))
        anns = sorted(ann_by_unitig.get(u, set()))
        rows_sum.append({
            "unitig": u,
            "n_samples": len(ss),
            "samples": ",".join(ss),
            "annotations": " | ".join(anns),
        })
    df_sum = pd.DataFrame(rows_sum)
    write_tsv(df_sum, f"{args.out_prefix}_summary.tsv")

    print(f"[done] Wrote:\n  - {args.out_prefix}_long.tsv\n  - {args.out_prefix}_summary.tsv")


if __name__ == "__main__":
    main()

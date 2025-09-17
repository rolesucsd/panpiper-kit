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
  unitig  sample  contig  start  end  strand  hit_type  locus_tag  gene  product  dbxrefs
<out_prefix>_summary.tsv:
  unitig  n_samples  samples  annotations

Assumptions
-----------
- Coordinates are 1-based inclusive.
- If a unitig spans multiple CDS or hits multiple loci, all are reported.
- If no CDS overlap exists, annotation fields are left blank (intergenic).
- Bakta output detection:
  1) If --anno-dir/--anno-pattern resolves to a file, use it.
  2) Else if --bakta-run: run Bakta into --bakta-out-dir/{sample}/ with prefix = --bakta-prefix-pattern (default {sample}),
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
    Skips lines beginning with '#'.
    """
    if not Path(anno_path).exists():
        return []

    def norm(s): return s.strip().lower().replace(" ", "_")

    cds = []
    with open_maybe_gz(anno_path) as fh:
        header = None
        for ln in fh:
            if not ln.strip() or ln.startswith("#"):
                continue
            parts = ln.rstrip("\n").split("\t")
            if header is None:
                header = [norm(x) for x in parts]
                col = {c: i for i, c in enumerate(header)}
                for req in ("sequence_id", "type", "start", "stop", "strand"):
                    if req not in col:
                        raise ValueError(f"Bakta TSV missing column: {req}")
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
# Search / overlap
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
    bakta_prefix_pattern: str,
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
    prefix   = bakta_prefix_pattern.format(sample=sample)
    target_tsv = out_dir / f"{prefix}.tsv"

    # If out_dir exists and contains prefix.tsv (or any .tsv), skip running.
    if out_dir.exists():
        if target_tsv.exists():
            return target_tsv
        # else: try any .tsv in the dir
        ts = sorted(out_dir.glob("*.tsv"))
        if ts:
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
    try:
        subprocess.run(cmd, check=True)
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
    """Ensure Bakta TSV, then scan FASTA & map to annotations. Returns list of long rows."""
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
        bakta_prefix_pattern=sample,
        bakta_cores=bakta_cores,
        bakta_extra_args=bakta_extra_args,
    )

    # Load Bakta CDS
    cds_rows = load_bakta_tsv(tsv_path)
    cds_by_contig = defaultdict(list)
    for r in cds_rows:
        cds_by_contig[r["contig"]].append(r)

    # Prepare RC map
    rc_map = {}
    if allow_revcomp:
        for u in unitigs_for_sample:
            rc_map[u] = reverse_complement(u)

    # Scan FASTA
    rows = []
    u_set = set(unitigs_for_sample)
    for contig, seq in stream_fasta(fa_path):
        if not u_set:
            break
        for u in list(u_set):
            # forward
            starts = find_all_occurrences(seq, u)
            for s1 in starts:
                e1 = s1 + len(u) - 1
                cds_hits = list(overlap_cds_on_contig(contig, s1, e1, cds_by_contig))
                if cds_hits:
                    for c in cds_hits:
                        rows.append({
                            "unitig": u, "sample": sample, "contig": contig,
                            "start": s1, "end": e1, "strand": "+", "hit_type": "exact",
                            "locus_tag": c["locus_tag"], "gene": c["gene"],
                            "product": c["product"], "dbxrefs": c["dbxrefs"],
                        })
                else:
                    rows.append({
                        "unitig": u, "sample": sample, "contig": contig,
                        "start": s1, "end": e1, "strand": "+", "hit_type": "exact",
                        "locus_tag": "", "gene": "", "product": "", "dbxrefs": "",
                    })

            # reverse
            if allow_revcomp:
                rc = rc_map[u]
                rstarts = find_all_occurrences(seq, rc)
                for s1 in rstarts:
                    e1 = s1 + len(u) - 1
                    cds_hits = list(overlap_cds_on_contig(contig, s1, e1, cds_by_contig))
                    if cds_hits:
                        for c in cds_hits:
                            rows.append({
                                "unitig": u, "sample": sample, "contig": contig,
                                "start": s1, "end": e1, "strand": "-", "hit_type": "revcomp",
                                "locus_tag": c["locus_tag"], "gene": c["gene"],
                                "product": c["product"], "dbxrefs": c["dbxrefs"],
                            })
                    else:
                        rows.append({
                            "unitig": u, "sample": sample, "contig": contig,
                            "start": s1, "end": e1, "strand": "-", "hit_type": "revcomp",
                            "locus_tag": "", "gene": "", "product": "", "dbxrefs": "",
                        })

    # Also record unitigs not found at all (diagnostic)
    found = {(r["unitig"], r["sample"]) for r in rows}
    for u in unitigs_for_sample:
        if (u, sample) not in found:
            rows.append({
                "unitig": u, "sample": sample, "contig": "", "start": "",
                "end": "", "strand": "", "hit_type": "not_found",
                "locus_tag": "", "gene": "", "product": "", "dbxrefs": "",
            })

    return rows


# ---------------------------
# Main
# ---------------------------

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

    ps["q_filter"] = bh_fdr(ps["filter-pvalue"].tolist())
    ps["q_lrt"]    = bh_fdr(ps["lrt-pvalue"].tolist())

    ps_sig = ps[(ps["q_filter"] < args.q_thresh) | (ps["q_lrt"] < args.q_thresh)].copy()
    ps_sig["variant"] = ps_sig["variant"].astype(str).str.strip()

    if ps_sig.empty:
        write_tsv(pd.DataFrame(columns=[
            "unitig","sample","contig","start","end","strand","hit_type","locus_tag","gene","product","dbxrefs"
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
                bakta_prefix_pattern=sample,
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
            "unitig","sample","contig","start","end","strand","hit_type","locus_tag","gene","product","dbxrefs"
        ])
    write_tsv(df_long, f"{args.out_prefix}_long.tsv")

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

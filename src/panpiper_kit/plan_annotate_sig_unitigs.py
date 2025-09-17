#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plan-only helper for annotate_sig_unitigs: outputs per-sample plan without invoking Bakta.

Given a Pyseer TSV and a unitig→samples map, applies BH-FDR on both p-value
columns and filters significant unitigs (q_filter < q || q_lrt < q). Then it
groups significant unitigs by sample and writes a TSV describing, for each
sample, the unitigs to scan and the expected FASTA and Bakta TSV paths.

This script does NOT open FASTA files nor run Bakta.
"""

import argparse
from pathlib import Path
from collections import defaultdict

try:
    import pandas as pd
except ImportError:
    pd = None

# Local copies to avoid importing runtime type annotations from annotate_sig_unitigs on Python <3.10
import gzip
from collections import defaultdict as _dd


def open_maybe_gz(path):
    return gzip.open(path, "rt") if str(path).endswith(".gz") else open(path, "r")


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


def parse_unitig_map(path):
    unitig_to_samples = _dd(set)
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


def main():
    ap = argparse.ArgumentParser(
        description="Plan-only: produce per-sample plan for Bakta/FASTA scanning without running Bakta."
    )
    ap.add_argument("--pyseer", required=True, help="ONE Pyseer results TSV.")
    ap.add_argument("--unitig-map", required=True, help="Unitig→samples file (no header).")
    ap.add_argument("--q-thresh", type=float, default=0.01, help="FDR threshold (default 0.01).")

    ap.add_argument("--fasta-dir", required=True, help="Directory with per-sample FASTA files.")
    ap.add_argument("--fasta-pattern", default="{sample}.fasta", help="FASTA filename pattern (supports .gz).")
    ap.add_argument("--anno-dir", default=None, help="Directory with per-sample Bakta TSVs (optional).")
    ap.add_argument("--anno-pattern", default="{sample}/{sample}.tsv",
                    help="Annotation TSV pattern within --anno-dir (default '{sample}/{sample}.tsv').")
    ap.add_argument("--bakta-out-dir", default=None,
                    help="Root directory for Bakta outputs (default: --anno-dir if set, else --fasta-dir).")
    ap.add_argument("--bakta-prefix-pattern", default="{sample}", help="Bakta --prefix value (default '{sample}').")

    ap.add_argument("--out-prefix", required=True, help="Output prefix for plan TSV.")

    args = ap.parse_args()

    if pd is None:
        raise SystemExit("This script requires pandas: pip install pandas")

    anno_dir = Path(args.anno_dir) if args.anno_dir else None
    bakta_out_dir = Path(args.bakta_out_dir) if args.bakta_out_dir else (anno_dir if anno_dir else Path(args.fasta_dir))

    # Load Pyseer and compute q-values
    ps = pd.read_csv(args.pyseer, sep="\t", dtype=str, keep_default_na=False)
    for col in ("variant", "filter-pvalue", "lrt-pvalue"):
        if col not in ps.columns:
            raise SystemExit(f"Missing column in Pyseer: {col}")

    ps["q_filter"] = bh_fdr(ps["filter-pvalue"].tolist())
    ps["q_lrt"]    = bh_fdr(ps["lrt-pvalue"].tolist())

    ps_sig = ps[(ps["q_filter"] < args.q_thresh) | (ps["q_lrt"] < args.q_thresh)].copy()
    ps_sig["variant"] = ps_sig["variant"].astype(str).str.strip()

    if ps_sig.empty:
        out_path = f"{args.out_prefix}_plan.tsv"
        pd.DataFrame(columns=[
            "sample","n_unitigs","unitigs","fasta_path","anno_expected_path","bakta_expected_dir","bakta_expected_tsv"
        ]).to_csv(out_path, sep="\t", index=False)
        print(f"[plan-only] No significant unitigs. Wrote: {out_path}")
        return

    signif_unitigs = set(ps_sig["variant"].tolist())

    # Unitig → samples; assign each unitig to exactly ONE sample
    # Policy: choose the candidate sample that has the highest count of significant unitigs overall.
    # Tie-breaker: lexicographically smallest sample id.
    unitig_to_samples = parse_unitig_map(args.unitig_map)
    # Precompute per-sample total significant unitig counts
    sample_total_counts = defaultdict(int)
    for u in signif_unitigs:
        for s in unitig_to_samples.get(u, set()):
            sample_total_counts[s] += 1
    # Assign unitigs
    per_sample_sets = defaultdict(set)
    for u in signif_unitigs:
        candidates = list(unitig_to_samples.get(u, set()))
        if not candidates:
            continue
        # Choose max by total count, then lexicographic
        chosen = sorted(candidates, key=lambda s: (-sample_total_counts.get(s, 0), s))[0]
        per_sample_sets[chosen].add(u)

    # Build plan rows
    plan_rows = []
    for sample, ulist in sorted(((s, sorted(list(us))) for s, us in per_sample_sets.items()), key=lambda kv: len(kv[1]), reverse=True):
        fa_path = Path(args.fasta_dir) / args.fasta_pattern.format(sample=sample)
        ann_expected = (anno_dir / args.anno_pattern.format(sample=sample)) if anno_dir is not None else Path("")
        bakta_prefix = args.bakta_prefix_pattern.format(sample=sample)
        bakta_expected_dir = bakta_out_dir / sample
        bakta_expected_tsv = bakta_expected_dir / f"{bakta_prefix}.tsv"
        plan_rows.append({
            "sample": sample,
            "n_unitigs": len(ulist),
            "unitigs": ",".join(ulist),
            "fasta_path": str(fa_path),
            "anno_expected_path": str(ann_expected) if str(ann_expected) != "." else "",
            "bakta_expected_dir": str(bakta_expected_dir),
            "bakta_expected_tsv": str(bakta_expected_tsv),
        })

    out_path = f"{args.out_prefix}_plan.tsv"
    pd.DataFrame(plan_rows).to_csv(out_path, sep="\t", index=False)
    print(f"[plan-only] Wrote: {out_path}")


if __name__ == "__main__":
    main()



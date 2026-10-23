"""Cluster enzyme sequences with MMseqs2 at several identity thresholds.

For each threshold runs `mmseqs easy-cluster` and writes, under results/mmseqs/:
  seq{ID}_cluster.tsv    representative -> member (MMseqs2 output)
  seq{ID}_rep_seq.fasta  one representative sequence per cluster
Then combines them into:
  results/seq_clusters.tsv          accession, seq30, seq50, seq90 (cluster representative)
  results/seq_cluster_summary.tsv   cluster counts and sizes per threshold

Example:
  .venv/bin/python scripts/cluster_sequences.py
  .venv/bin/python scripts/cluster_sequences.py --ids 0.3 0.4 0.5 --coverage 0.5
"""

import argparse
import os
import shutil
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def run_cluster(fasta, prefix, tmp, min_seq_id, coverage, cov_mode, sensitivity):
    cmd = [
        "mmseqs", "easy-cluster", str(fasta), str(prefix), str(tmp),
        "--min-seq-id", str(min_seq_id),
        "-c", str(coverage), "--cov-mode", str(cov_mode),
        "-s", str(sensitivity),
        "--threads", str(os.cpu_count()),
        "-v", "1",
    ]
    subprocess.run(cmd, check=True)
    for extra in ("_all_seqs.fasta",):  # duplicate of the input, not needed
        Path(f"{prefix}{extra}").unlink(missing_ok=True)
    return pd.read_csv(f"{prefix}_cluster.tsv", sep="\t", header=None,
                       names=["representative", "accession"])


def summarize(label, edges):
    sizes = edges.groupby("representative").size()
    return {
        "threshold": label,
        "n_proteins": len(edges),
        "n_clusters": len(sizes),
        "n_singletons": int((sizes == 1).sum()),
        "largest_cluster": int(sizes.max()),
        "mean_cluster_size": round(sizes.mean(), 2),
        "frac_in_largest": round(sizes.max() / len(edges), 3),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fasta", type=Path, default=ROOT / "data" / "enzymes.fasta")
    p.add_argument("--out", type=Path, default=ROOT / "results")
    p.add_argument("--ids", type=float, nargs="+", default=[0.3, 0.5, 0.9],
                   help="--min-seq-id values")
    p.add_argument("--coverage", type=float, default=0.8, help="-c: minimum aligned fraction")
    p.add_argument("--cov-mode", type=int, default=0,
                   help="0 = coverage of query and target, 1 = target only, 2 = query only")
    p.add_argument("--sensitivity", type=float, default=7.5,
                   help="-s: high value so remote (~30%%) homologs are still found")
    args = p.parse_args()

    if not shutil.which("mmseqs"):
        raise SystemExit("mmseqs not found; install with: brew install mmseqs2")
    mm_dir = args.out / "mmseqs"
    mm_dir.mkdir(parents=True, exist_ok=True)
    tmp = mm_dir / "tmp"

    assignments, summary = None, []
    for min_id in args.ids:
        label = f"seq{round(min_id * 100)}"
        print(f"Clustering at --min-seq-id {min_id} -> {label}")
        edges = run_cluster(args.fasta, mm_dir / label, tmp, min_id,
                            args.coverage, args.cov_mode, args.sensitivity)
        summary.append(summarize(label, edges))
        col = edges.rename(columns={"representative": label})[["accession", label]]
        assignments = col if assignments is None else assignments.merge(col, on="accession", how="outer")
    shutil.rmtree(tmp, ignore_errors=True)

    assignments.sort_values("accession").to_csv(args.out / "seq_clusters.tsv", sep="\t", index=False)
    summary = pd.DataFrame(summary)
    summary.insert(1, "coverage", args.coverage)
    summary.insert(2, "cov_mode", args.cov_mode)
    summary.to_csv(args.out / "seq_cluster_summary.tsv", sep="\t", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

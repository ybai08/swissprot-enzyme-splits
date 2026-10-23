"""Cluster AlphaFold models by global structure with Foldseek.

Runs `foldseek easy-cluster` on the models listed as `ok` in
data/structures_manifest.tsv, once per preset:

  fold_e3_c80    E-value < 1e-3, 80% coverage of both structures (homology-level match
                 over most of the protein; Foldseek's usual clustering settings)
  fold_tm50_c50  TM-score > 0.5 (the conventional "same fold" cutoff), 50% coverage,
                 no E-value filter. Looser: catches shared domains in multidomain enzymes.

Writes:
  results/foldseek/{preset}_cluster.tsv    representative -> member (Foldseek output)
  results/struct_clusters.tsv              accession + one column per preset
                                           (blank = no structure, so structurally unassessed)
  results/struct_cluster_summary.tsv       cluster stats per preset, and how many
                                           structure clusters join several seq30 clusters

Example:
  .venv/bin/python scripts/cluster_structures.py
  .venv/bin/python scripts/cluster_structures.py --mask-bfactor 50
"""

import argparse
import os
import re
import shutil
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PRESETS = {
    "fold_e3_c80": ["-e", "1e-3", "-c", "0.8", "--cov-mode", "0"],
    "fold_tm50_c50": ["-e", "10", "-c", "0.5", "--cov-mode", "0",
                      "--tmscore-threshold", "0.5", "--exact-tmscore", "1"],
}


def to_accession(name):
    """Foldseek names entries after the file (and chain); reduce to the UniProt accession."""
    name = re.sub(r"\.(pdb|cif)(\.gz)?$", "", name)
    return re.sub(r"_[A-Za-z0-9]$", "", name)


def run_cluster(input_dir, prefix, tmp, preset_args, mask_bfactor):
    cmd = ["foldseek", "easy-cluster", str(input_dir), str(prefix), str(tmp),
           *preset_args, "--mask-bfactor-threshold", str(mask_bfactor),
           "--threads", str(os.cpu_count()), "-v", "1"]
    subprocess.run(cmd, check=True)
    for extra in ("_all_seqs.fasta", "_rep_seq.fasta"):  # 3Di/AA FASTA dumps, not needed
        Path(f"{prefix}{extra}").unlink(missing_ok=True)
    edges = pd.read_csv(f"{prefix}_cluster.tsv", sep="\t", header=None,
                        names=["representative", "accession"])
    return edges.apply(lambda col: col.map(to_accession))


def summarize(label, edges, seq):
    sizes = edges.groupby("representative").size()
    joined = edges.merge(seq, on="accession", how="left")
    seq_per_cluster = joined.groupby("representative")["seq30"].nunique()
    multi = seq_per_cluster[seq_per_cluster > 1].index
    return {
        "preset": label,
        "n_proteins": len(edges),
        "n_clusters": len(sizes),
        "n_singletons": int((sizes == 1).sum()),
        "largest_cluster": int(sizes.max()),
        "mean_cluster_size": round(sizes.mean(), 2),
        # Structure links that 30%-identity sequence clustering missed.
        "n_clusters_joining_seq30": len(multi),
        "n_proteins_in_those": int(sizes[multi].sum()),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, default=ROOT / "data" / "structures_manifest.tsv")
    p.add_argument("--seq-clusters", type=Path, default=ROOT / "results" / "seq_clusters.tsv")
    p.add_argument("--out", type=Path, default=ROOT / "results")
    p.add_argument("--presets", nargs="+", default=list(PRESETS), choices=list(PRESETS))
    p.add_argument("--mask-bfactor", type=float, default=0,
                   help="ignore residues with pLDDT below this when seeding (0 = off)")
    args = p.parse_args()

    if not shutil.which("foldseek"):
        raise SystemExit("foldseek not found; see README for install")
    manifest = pd.read_csv(args.manifest, sep="\t", dtype=str).fillna("")
    ok = manifest[manifest.status == "ok"]
    fs_dir = args.out / "foldseek"
    fs_dir.mkdir(parents=True, exist_ok=True)

    # Cluster only models that passed the fetch checks: link them into one input folder.
    input_dir = fs_dir / "input"
    shutil.rmtree(input_dir, ignore_errors=True)
    input_dir.mkdir()
    for path in ok.structure_path:
        (input_dir / Path(path).name).symlink_to(ROOT / path)
    tmp = fs_dir / "tmp"

    seq = pd.read_csv(args.seq_clusters, sep="\t")[["accession", "seq30"]]
    assignments = manifest[["accession"]].copy()
    summary = []
    for label in args.presets:
        print(f"Clustering {len(ok)} structures with preset {label}")
        edges = run_cluster(input_dir, fs_dir / label, tmp, PRESETS[label], args.mask_bfactor)
        unknown = set(edges.accession) - set(ok.accession)
        assert not unknown, f"unmapped Foldseek IDs: {sorted(unknown)[:5]}"
        assert len(edges) == len(ok), f"{len(ok) - len(edges)} structures missing from output"
        summary.append(summarize(label, edges, seq))
        col = edges.rename(columns={"representative": label})
        assignments = assignments.merge(col, on="accession", how="left")
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.rmtree(input_dir, ignore_errors=True)

    assignments.sort_values("accession").to_csv(args.out / "struct_clusters.tsv", sep="\t", index=False)
    summary = pd.DataFrame(summary)
    summary.to_csv(args.out / "struct_cluster_summary.tsv", sep="\t", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

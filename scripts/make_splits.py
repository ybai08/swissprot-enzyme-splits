"""Assign enzymes to train/validation/test (80/10/10) by group, compare strategies, audit leakage.

Strategies (each protein's group decides its split; whole groups go to one split):
  random            every protein is its own group (naive baseline)
  sequence          MMseqs2 30%-identity clusters (seq30)
  structure         Foldseek fold_tm50_c50 clusters
  motif             Folddisco strict motif groups
  combined          connected components of the union of seq30 + fold_tm50_c50 + motif_strict
                    (A shares a sequence cluster with B, B a motif group with C -> A, B, C together)
  combined_lenient  as combined, with motif_lenient (the over-merging upper bound)

Group -> split: groups are shuffled (seeded), sorted largest first, and each goes to the
split furthest below its target size. So big groups land in train and singletons fill
val/test.

Leakage audit. The audit is independent of the clusters used to build the splits: it uses
all-vs-all searches, which also catch partial-coverage matches that clustering ignores.
For each val/test protein, count whether it has, against any train protein:
  seq_any    an MMseqs2 hit with E < 1e-3
  seq30      >= 30% identity over >= 50% of the held-out protein
  fold       a Foldseek hit with TM-score >= 0.5, normalized by the held-out protein
  motif      a strict Folddisco motif hit, in either direction
  any        any of the above

Writes:
  results/audit/seq_allvsall.tsv, results/audit/fold_allvsall.tsv   cached searches
  results/split_comparison.tsv     one row per strategy (mean over --seeds)
  results/dataset_manifest.csv     one row per enzyme, `split` from --strategy and seed 0,
                                   plus split_<strategy> columns for every strategy

Example:
  .venv/bin/python scripts/make_splits.py
  .venv/bin/python scripts/make_splits.py --strategy sequence --seeds 20
"""

import argparse
import os
import random
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
SPLITS = {"train": 0.8, "val": 0.1, "test": 0.1}
STRATEGIES = {
    "random": [],
    "sequence": ["seq30"],
    "structure": ["fold_tm50_c50"],
    "motif": ["motif_strict"],
    "combined": ["seq30", "fold_tm50_c50", "motif_strict"],
    "combined_lenient": ["seq30", "fold_tm50_c50", "motif_lenient"],
}


# ---------------------------------------------------------------- grouping

class UnionFind:
    def __init__(self, items):
        self.parent = {x: x for x in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def groups_for(table, columns):
    """Accession -> group id: connected components over shared labels in `columns`.

    Blank labels (no structure, no motif) join nothing.
    """
    uf = UnionFind(table.accession)
    for col in columns:
        for _, members in table.dropna(subset=[col]).groupby(col).accession:
            members = list(members)
            for other in members[1:]:
                uf.union(members[0], other)
    return table.accession.map(uf.find)


def assign(groups, seed):
    """Group-level greedy split: largest group first, into the split most below target."""
    sizes = groups.value_counts()
    ids = list(sizes.index)
    random.Random(seed).shuffle(ids)
    ids.sort(key=lambda g: -sizes[g])  # stable: ties keep the shuffled order
    n, count, where = len(groups), dict.fromkeys(SPLITS, 0), {}
    for g in ids:
        split = max(SPLITS, key=lambda s: SPLITS[s] * n - count[s])
        where[g] = split
        count[split] += sizes[g]
    return groups.map(where)


# ---------------------------------------------------------------- audit searches

def seq_allvsall(fasta, out):
    if not out.exists():
        tmp = out.parent / "tmp_seq"
        subprocess.run(["mmseqs", "easy-search", str(fasta), str(fasta), str(out), str(tmp),
                        "-s", "7.5", "-e", "1e-3", "--threads", str(os.cpu_count()), "-v", "1",
                        "--format-output", "query,target,fident,qcov,tcov,evalue"], check=True)
        shutil.rmtree(tmp, ignore_errors=True)
    hits = pd.read_csv(out, sep="\t", header=None, names=["query", "target", "fident", "qcov", "tcov", "evalue"])
    return hits[hits["query"] != hits["target"]]


def fold_allvsall(struct_dir, out):
    if not out.exists():
        tmp = out.parent / "tmp_fold"
        subprocess.run(["foldseek", "easy-search", str(struct_dir), str(struct_dir), str(out), str(tmp),
                        "--exhaustive-search", "1", "-e", "10", "--threads", str(os.cpu_count()), "-v", "1",
                        "--format-output", "query,target,qtmscore,ttmscore,evalue,prob"], check=True)
        shutil.rmtree(tmp, ignore_errors=True)
    hits = pd.read_csv(out, sep="\t", header=None, names=["query", "target", "qtmscore", "ttmscore", "evalue", "prob"])
    for col in ("query", "target"):
        hits[col] = hits[col].str.replace(r"\.pdb$", "", regex=True)
    return hits[hits["query"] != hits["target"]]


def leak_pairs(seq, fold, motif):
    """Directed (held-out protein, other protein) pairs for each leakage type."""
    pairs = {
        "seq_any": seq[["query", "target"]],
        "seq30": seq.loc[(seq.fident >= 0.3) & (seq.qcov >= 0.5), ["query", "target"]],
        "fold": fold.loc[fold.qtmscore >= 0.5, ["query", "target"]],
        # A motif hit links both proteins, whichever one was the query.
        "motif": pd.concat([motif[["query", "target"]],
                            motif.rename(columns={"query": "target", "target": "query"})[["query", "target"]]]),
    }
    return {k: set(map(tuple, v.itertuples(index=False))) for k, v in pairs.items()}


def audit(split, pairs):
    """Fraction of val / test proteins with each leakage type against train."""
    train = set(split.index[split == "train"])
    out = {}
    for held in ("val", "test"):
        members = set(split.index[split == held])
        leaked = {k: {q for q, t in v if q in members and t in train} for k, v in pairs.items()}
        leaked["any"] = set().union(*leaked.values())
        for k, v in leaked.items():
            out[f"{held}_{k}"] = len(v) / max(len(members), 1)
    return out


# ---------------------------------------------------------------- main

def notes_for(r):
    notes = []
    if r.structure_status != "ok":
        notes.append("no structure: fold and motif similarity unassessed")
    if r.act_site_status == "annotated" and not r.act_site_experimental:
        notes.append("ACT_SITE inferred (similarity/rule), not experimental")
    if pd.notna(r.act_site_plddt) and r.act_site_plddt < 70:
        notes.append("active-site pLDDT < 70")
    if r.n_ec > 1:
        notes.append("multiple EC numbers")
    return "; ".join(notes)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--strategy", default="combined", choices=list(STRATEGIES),
                   help="strategy used for the manifest's `split` column")
    p.add_argument("--seeds", type=int, default=10, help="seeds averaged in the comparison table")
    args = p.parse_args()

    enzymes = pd.read_csv(ROOT / "data" / "enzymes.tsv", sep="\t", dtype=str).fillna("")
    structs = pd.read_csv(ROOT / "data" / "structures_manifest.tsv", sep="\t", dtype=str).fillna("")
    table = (enzymes[["accession"]]
             .merge(pd.read_csv(RES / "seq_clusters.tsv", sep="\t"), on="accession", how="left")
             .merge(pd.read_csv(RES / "struct_clusters.tsv", sep="\t"), on="accession", how="left")
             .merge(pd.read_csv(RES / "motif_groups.tsv", sep="\t"), on="accession", how="left"))

    audit_dir = RES / "audit"
    audit_dir.mkdir(exist_ok=True)
    ok = set(structs.loc[structs.status == "ok", "accession"])
    seq = seq_allvsall(ROOT / "data" / "enzymes.fasta", audit_dir / "seq_allvsall.tsv")
    fold = fold_allvsall(ROOT / "data" / "structures", audit_dir / "fold_allvsall.tsv")
    fold = fold[fold["query"].isin(ok) & fold["target"].isin(ok)]
    motif = pd.read_csv(RES / "folddisco" / "strict_hits.tsv", sep="\t")
    motif = motif[~motif.generic_query]
    pairs = leak_pairs(seq, fold, motif)
    print("Leakage pairs (directed):", {k: len(v) for k, v in pairs.items()})

    ec_class = enzymes.set_index("accession").ec_class
    annotated = enzymes.set_index("accession").act_site_status == "annotated"
    rows, chosen = [], {}
    for name, columns in STRATEGIES.items():
        groups = groups_for(table, columns)
        sizes = groups.value_counts()
        per_seed = []
        for seed in range(args.seeds):
            split = pd.Series(assign(groups, seed).values, index=table.accession)
            if seed == 0:
                chosen[name] = (groups.values, split.values)
            counts = split.value_counts()
            test = split.index[split == "test"]
            per_seed.append({
                **{f"n_{s}": counts.get(s, 0) for s in SPLITS},
                "test_ec_classes": ec_class[test].nunique(),
                "test_act_site_annotated": int(annotated[test].sum()),
                **audit(split, pairs),
            })
        row = {"strategy": name, "grouping": " + ".join(columns) or "per protein",
               "n_groups": len(sizes), "largest_group": int(sizes.max()),
               "largest_group_frac": round(sizes.max() / len(groups), 3)}
        row.update(pd.DataFrame(per_seed).mean().round(3).to_dict())
        rows.append(row)
    comparison = pd.DataFrame(rows)
    comparison.to_csv(RES / "split_comparison.tsv", sep="\t", index=False)
    show = ["strategy", "n_groups", "largest_group", "n_train", "n_val", "n_test",
            "test_seq_any", "test_seq30", "test_fold", "test_motif", "test_any", "val_any"]
    print(comparison[show].to_string(index=False))

    # Manifest.
    e = enzymes.set_index("accession")
    s = structs.set_index("accession")
    m = pd.DataFrame({"accession": table.accession})
    m["entry_name"] = m.accession.map(e["id"])
    m["protein_name"] = m.accession.map(e.protein_name)
    m["organism"] = m.accession.map(e.organism_name)
    m["reviewed"] = m.accession.map(e.reviewed)
    m["ec_label"] = m.accession.map(e.ec)
    m["ec_class"] = m.accession.map(e.ec_class)
    m["n_ec"] = m.accession.map(e.n_ec).astype(int)
    m["length"] = m.accession.map(e.length).astype(int)
    m["fasta_path"] = "data/enzymes.fasta"
    m["act_site_status"] = m.accession.map(e.act_site_status)
    m["act_site_positions"] = m.accession.map(e.act_site_positions)
    m["act_site_experimental"] = m.accession.map(e.act_site_experimental) == "True"
    m["structure_status"] = m.accession.map(s.status)
    m["structure_source"] = m.accession.map(s.structure_source)
    m["structure_path"] = m.accession.map(s.structure_path)
    m["plddt_mean"] = pd.to_numeric(m.accession.map(s.plddt_mean), errors="coerce")
    m["act_site_plddt"] = pd.to_numeric(m.accession.map(s.act_site_plddt), errors="coerce")
    m["msa_url"] = m.accession.map(s.msa_url)
    for col in ["seq30", "seq50", "seq90", "fold_e3_c80", "fold_tm50_c50",
                "motif_status", "motif_strict", "motif_lenient"]:
        m[col] = table[col].values
    motifs = pd.read_csv(RES / "folddisco" / "motifs.tsv", sep="\t", dtype=str).fillna("").set_index("accession")
    m["motif_residues"] = m.accession.map(motifs.motif_residues)
    m["split_group"] = chosen[args.strategy][0]
    m["split"] = chosen[args.strategy][1]
    for name in STRATEGIES:
        m[f"split_{name}"] = chosen[name][1]
    m["notes"] = [notes_for(r) for r in m.itertuples()]
    m.to_csv(RES / "dataset_manifest.csv", index=False)
    print(f"\nManifest: {len(m)} rows, split from '{args.strategy}':", m.split.value_counts().to_dict())


if __name__ == "__main__":
    main()

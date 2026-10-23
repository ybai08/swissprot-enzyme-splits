"""Group enzymes by active-site geometry with Folddisco.

1. Index every `ok` AlphaFold model (results/folddisco/index; input/ holds symlinks the
   index points to, so keep it).
2. Build one motif per enzyme with UniProt ACT_SITE residues. Folddisco hashes residue
   pairs, so a motif needs >= 3 residues. Motifs with 1-2 ACT_SITE residues are topped up
   with the UniProt BINDING residues closest in 3D (heavy-atom distance <= --pocket-radius),
   up to --max-motif residues. Motifs still < 3 residues are not queried.
3. Query each motif against the whole index (batch mode), once per preset:
     strict   exact residue types, every motif residue matched, RMSD <= 1.0 A
     lenient  residues may swap within a chemical class (--aa-subst group: DE, RHK, NQST, ...),
              every motif residue matched, RMSD <= 1.0 A
   (for motifs of 5+ residues, 80% of residues must match instead of all)
   3-residue motifs match by chance more easily, so both presets cap their RMSD at
   --small-motif-rmsd (0.55 A). On this sample, true 3-residue hits (HExxH, HAD DxD) were
   <= 0.51 A and unrelated ones (acidic triplets in EF-Tu, tRNA ligase, ...) >= 0.63 A.
4. A hit from query A to target B is an edge A-B. A query that hits more than
   --max-hit-frac of all structures is treated as a generic motif and its edges are dropped,
   so one common geometry cannot chain unrelated enzymes together.
5. Connected components of the edge graph are motif groups.

Writes, under results/:
  folddisco/motifs.tsv              per enzyme: motif residues, source, pLDDT, query status
  folddisco/{preset}_hits.tsv       every accepted query -> target hit
  motif_groups.tsv                  accession, motif_status, motif_strict, motif_lenient
  motif_group_summary.tsv           group counts per preset

motif_status:
  queried      own motif was searched
  target_only  no usable motif of its own (no / too few annotated residues, or low pLDDT);
               it can still join a group by being hit
  no_structure no AlphaFold model, so active-site similarity is unassessed (group left blank)

Example:
  .venv/bin/python scripts/motif_groups.py
"""

import argparse
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ANSI = re.compile(r"\x1b\[[0-9;]*m")
PRESETS = {
    "strict": {"flags": [], "rmsd": 1.0},
    "lenient": {"flags": ["--aa-subst", "group"], "rmsd": 1.0},
}


def read_atoms(pdb_path):
    """{residue number: (residue name, heavy-atom coords array, CA pLDDT)} for chain A."""
    res = {}
    with open(pdb_path) as fh:
        for line in fh:
            if not line.startswith("ATOM") or line[76:78].strip() == "H":
                continue
            num = int(line[22:26])
            xyz = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
            name, coords, plddt = res.get(num, (line[17:20], [], None))
            coords.append(xyz)
            if line[12:16].strip() == "CA":
                plddt = float(line[60:66])
            res[num] = (name, coords, plddt)
    return {k: (n, np.array(c), p) for k, (n, c, p) in res.items()}


def positions(text, feature):
    """Single-residue positions of a UniProt feature (ACT_SITE / BINDING); '?' entries skipped."""
    out = []
    for m in re.finditer(rf"{feature}\s+([<>?]?\d+)(?:\.\.([<>?]?\d+))?", str(text)):
        if "?" in m.group(0):
            continue
        a = int(m.group(1).strip("<>"))
        b = int(m.group(2).strip("<>")) if m.group(2) else a
        out.extend(range(a, b + 1))
    return sorted(set(out))


def build_motif(row, atoms, radius, max_motif):
    act = [p for p in positions(row.ft_act_site, "ACT_SITE") if p in atoms]
    if not act:
        return [], "none"
    motif, source = list(act), "act_site"
    if len(motif) < 3:
        dist = {}
        for b in positions(row.ft_binding, "BINDING"):
            if b in atoms and b not in motif:
                d = min(np.linalg.norm(atoms[b][1][:, None] - atoms[a][1][None], axis=-1).min() for a in act)
                if d <= radius:
                    dist[b] = d
        extra = sorted(dist, key=dist.get)[:max_motif - len(motif)]
        if extra:
            motif, source = sorted(motif + extra), "act_site+binding"
    return motif, source


def run_queries(index, queries_file, preset, threads):
    cmd = ["folddisco", "query", "-i", str(index), "-q", str(queries_file), "-t", str(threads),
           "--format-output", "tid,node_count,rmsd,idf,matching_residues", *PRESETS[preset]["flags"]]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def min_matched(motif_size):
    """Residues a hit must match: all for 3-4 residue motifs, 80% for larger (as --confident)."""
    return motif_size if motif_size <= 4 else int(np.ceil(0.8 * motif_size))


def read_hits(path, query, motif_size, max_rmsd):
    """Accepted hits of one query: enough residues matched, RMSD under the cap, not itself."""
    cols = ["tid", "node_count", "rmsd", "idf", "matching_residues"]
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=["query", "target", "rmsd", "idf", "matching_residues"])
    hits = pd.read_csv(path, sep="\t", header=None, names=cols, dtype={"matching_residues": str},
                       converters={"tid": lambda s: ANSI.sub("", s)})
    hits["target"] = hits.tid.map(lambda s: Path(s).stem)
    hits = hits[(hits.node_count >= min_matched(motif_size)) & (hits.rmsd <= max_rmsd) & (hits.target != query)]
    hits = hits.sort_values("rmsd").drop_duplicates("target")  # best match per target
    return hits.assign(query=query)[["query", "target", "rmsd", "idf", "matching_residues"]]


def components(nodes, edges):
    parent = {n: n for n in nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    return {n: find(n) for n in nodes}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--enzymes", type=Path, default=ROOT / "data" / "enzymes.tsv")
    p.add_argument("--manifest", type=Path, default=ROOT / "data" / "structures_manifest.tsv")
    p.add_argument("--out", type=Path, default=ROOT / "results")
    p.add_argument("--pocket-radius", type=float, default=8.0,
                   help="max heavy-atom distance (A) from an ACT_SITE residue for added BINDING residues")
    p.add_argument("--max-motif", type=int, default=6)
    p.add_argument("--min-plddt", type=float, default=70.0, help="skip motifs with lower mean pLDDT")
    p.add_argument("--max-hit-frac", type=float, default=0.05,
                   help="drop edges of queries hitting more than this fraction of structures")
    p.add_argument("--small-motif-rmsd", type=float, default=0.55,
                   help="RMSD cap (A) for 3-residue motifs in every preset")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--reindex", action="store_true")
    args = p.parse_args()

    if not shutil.which("folddisco"):
        raise SystemExit("folddisco not found; see README for install")
    enzymes = pd.read_csv(args.enzymes, sep="\t", dtype=str).fillna("")
    manifest = pd.read_csv(args.manifest, sep="\t", dtype=str).fillna("")
    ok = set(manifest.loc[manifest.status == "ok", "accession"])
    fd = args.out / "folddisco"
    fd.mkdir(parents=True, exist_ok=True)

    # Index only models that passed the fetch checks.
    index = fd / "index"
    if args.reindex or not Path(f"{index}.type").exists():
        input_dir = fd / "input"
        shutil.rmtree(input_dir, ignore_errors=True)
        input_dir.mkdir()
        for acc in ok:
            (input_dir / f"{acc}.pdb").symlink_to(ROOT / "data" / "structures" / f"{acc}.pdb")
        subprocess.run(["folddisco", "index", "-p", str(input_dir), "-i", str(index),
                        "-t", str(args.threads)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # Keep input/: Folddisco re-reads the indexed files at query time.

    # Build motifs.
    motifs = []
    for row in enzymes.itertuples(index=False):
        rec = {"accession": row.accession, "motif_residues": "", "motif_size": 0,
               "motif_source": "", "motif_plddt": "", "query_status": ""}
        if row.accession not in ok:
            rec["query_status"] = "no_structure"
            motifs.append(rec)
            continue
        atoms = read_atoms(ROOT / "data" / "structures" / f"{row.accession}.pdb")
        motif, source = build_motif(row, atoms, args.pocket_radius, args.max_motif)
        plddt = round(float(np.mean([atoms[r][2] for r in motif])), 2) if motif else ""
        rec.update(motif_residues=",".join(f"A{r}" for r in motif), motif_size=len(motif),
                   motif_source=source, motif_plddt=plddt,
                   motif_aa=",".join(atoms[r][0] for r in motif))
        if not motif:
            rec["query_status"] = "no_act_site"
        elif len(motif) < 3:
            rec["query_status"] = "motif_too_small"
        elif plddt < args.min_plddt:
            rec["query_status"] = "low_plddt"
        else:
            rec["query_status"] = "queried"
        motifs.append(rec)
    motifs = pd.DataFrame(motifs)
    motifs.to_csv(fd / "motifs.tsv", sep="\t", index=False)
    queried = motifs[motifs.query_status == "queried"]
    print("Motif status:", motifs.query_status.value_counts().to_dict())

    # Query and group.
    n_structures = len(ok)
    groups = enzymes[["accession"]].copy()
    groups["motif_status"] = np.where(~groups.accession.isin(ok), "no_structure",
                                      np.where(groups.accession.isin(queried.accession), "queried", "target_only"))
    summary = []
    for preset, cfg in PRESETS.items():
        out_dir = fd / f"{preset}_raw"
        shutil.rmtree(out_dir, ignore_errors=True)
        out_dir.mkdir()
        qfile = fd / f"{preset}_queries.tsv"
        with open(qfile, "w") as fh:
            for r in queried.itertuples():
                fh.write(f"{ROOT / 'data' / 'structures' / f'{r.accession}.pdb'}\t{r.motif_residues}\t{out_dir / f'{r.accession}.tsv'}\n")
        print(f"Querying {len(queried)} motifs ({preset})")
        run_queries(index, qfile, preset, args.threads)

        cap = lambda size: min(cfg["rmsd"], args.small_motif_rmsd) if size == 3 else cfg["rmsd"]
        hits = pd.concat([read_hits(out_dir / f"{r.accession}.tsv", r.accession, r.motif_size, cap(r.motif_size))
                          for r in queried.itertuples()], ignore_index=True)
        n_hits = hits.groupby("query").size()
        generic = set(n_hits[n_hits > args.max_hit_frac * n_structures].index)
        hits["generic_query"] = hits["query"].isin(generic)
        hits.to_csv(fd / f"{preset}_hits.tsv", sep="\t", index=False)
        edges = hits.loc[~hits.generic_query, ["query", "target"]].itertuples(index=False)

        comp = components(sorted(ok), edges)
        col = f"motif_{preset}"
        groups[col] = groups.accession.map(comp)  # NaN for no_structure
        sizes = groups[col].value_counts()
        summary.append({
            "preset": preset, "rmsd_max": cfg["rmsd"], "rmsd_max_3res": cap(3), "flags": " ".join(cfg["flags"]),
            "n_queried": len(queried), "n_queries_with_hits": int(hits["query"].nunique()),
            "n_generic_queries": len(generic), "n_edges": int((~hits.generic_query).sum()),
            "n_groups": len(sizes), "n_multi_member_groups": int((sizes > 1).sum()),
            "n_proteins_in_multi": int(sizes[sizes > 1].sum()), "largest_group": int(sizes.max()),
        })
        shutil.rmtree(out_dir)
        qfile.unlink()

    groups.sort_values("accession").to_csv(args.out / "motif_groups.tsv", sep="\t", index=False)
    summary = pd.DataFrame(summary)
    summary.to_csv(args.out / "motif_group_summary.tsv", sep="\t", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

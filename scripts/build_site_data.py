"""Export the data behind the project website (docs/index.html) to docs/data.js.

Reads the manifest, the cached audit searches and the split comparison, and writes
`window.SITE_DATA = {...}` so the page also works when opened from disk.

Per strategy (seed 0) it stores each protein's split, which val/test proteins leak
and which links cause it, using the same audit rules as make_splits.py.

Example:
  .venv/bin/python scripts/make_splits.py && .venv/bin/python scripts/build_site_data.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import make_splits as ms  # noqa: E402

SHOWN = ["random", "sequence", "structure", "motif", "combined"]


def short_name(name):
    return name.split(" (")[0].split(" [")[0]


def main():
    m = pd.read_csv(ms.RES / "dataset_manifest.csv")
    ok = set(m.loc[m.structure_status == "ok", "accession"])
    seq = ms.seq_allvsall(ROOT / "data" / "enzymes.fasta", ms.RES / "audit" / "seq_allvsall.tsv")
    fold = ms.fold_allvsall(ROOT / "data" / "structures", ms.RES / "audit" / "fold_allvsall.tsv")
    fold = fold[fold["query"].isin(ok) & fold["target"].isin(ok)]
    motif = pd.read_csv(ms.RES / "folddisco" / "strict_hits.tsv", sep="\t")
    motif = motif[~motif.generic_query]
    pairs = ms.leak_pairs(seq, fold, motif)
    del pairs["seq30"]  # subset of seq_any; the page shows link kinds, not thresholds

    # Undirected links with their kinds; directed pairs are kept for the leak rule.
    kinds = {"seq_any": "sequence", "fold": "fold", "motif": "motif"}
    links = {}
    for key, directed in pairs.items():
        for a, b in directed:
            links.setdefault(tuple(sorted((a, b))), set()).add(kinds[key])
    link_list = sorted(links)
    link_id = {pair: i for i, pair in enumerate(link_list)}
    index = {acc: i for i, acc in enumerate(m.accession)}

    strategies = {}
    for name in SHOWN:
        split = m.set_index("accession")[f"split_{name}"]
        leaked, leak_links = set(), set()
        for directed in pairs.values():
            for q, t in directed:
                if split[q] in ("val", "test") and split[t] == "train":
                    leaked.add(q)
                    leak_links.add(link_id[tuple(sorted((q, t)))])
        strategies[name] = {
            "split": [split[a] for a in m.accession],
            "leaked": sorted(index[a] for a in leaked),
            "leak_links": sorted(leak_links),
        }

    comparison = pd.read_csv(ms.RES / "split_comparison.tsv", sep="\t")
    groups = m.split_group.map(m.split_group.value_counts())
    data = {
        "nodes": [{
            "id": r.accession,
            "name": short_name(r.protein_name),
            "ec": r.ec_label,
            "ec_class": int(r.ec_class),
            "length": int(r.length),
            "organism": r.organism.split(" (")[0],
            "act_site": r.act_site_status == "annotated",
            "plddt": None if pd.isna(r.plddt_mean) else round(r.plddt_mean, 1),
            "has_structure": r.structure_status == "ok",
            "group_size": int(g),
        } for r, g in zip(m.itertuples(), groups)],
        "links": [{"s": index[a], "t": index[b], "kinds": sorted(links[(a, b)])} for a, b in link_list],
        "strategies": strategies,
        "comparison": comparison[comparison.strategy.isin(SHOWN)].to_dict("records"),
    }
    out = ROOT / "docs" / "data.js"
    out.write_text("window.SITE_DATA = " + json.dumps(data, separators=(",", ":")) + ";\n")
    print(f"Wrote {out.relative_to(ROOT)}: {len(data['nodes'])} nodes, {len(data['links'])} links,",
          {k: len(v["leaked"]) for k, v in strategies.items()}, "leaked val/test proteins")


if __name__ == "__main__":
    main()

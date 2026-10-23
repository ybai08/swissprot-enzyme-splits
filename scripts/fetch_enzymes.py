"""Fetch a reproducible random sample of reviewed Swiss-Prot enzymes from UniProt.

Steps:
  1. Stream every accession matching the query (reviewed + EC number + filters).
  2. Draw a seeded random sample, so the subset is not biased toward the API's first page.
  3. Fetch metadata and sequences for the sample in batches.
  4. Write data/enzymes.tsv, data/enzymes.fasta and data/fetch_log.json.

Example:
  .venv/bin/python scripts/fetch_enzymes.py --n 300 --seed 0
  .venv/bin/python scripts/fetch_enzymes.py --n 300 --require-act-site
"""

import argparse
import io
import json
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

API = "https://rest.uniprot.org/uniprotkb"
FIELDS = [
    "accession", "id", "reviewed", "protein_name", "ec", "organism_name", "organism_id",
    "length", "fragment", "annotation_score", "ft_act_site", "ft_binding", "cc_cofactor",
    "xref_pdb", "sequence",
]
# The TSV header uses display names, so columns are renamed by position to the field names.
BATCH = 200
EXPERIMENTAL_ECO = "ECO:0000269"  # experimental evidence used in a manual assertion


def get(url, params, retries=5, **kwargs):
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=300, **kwargs)
            if r.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"HTTP {r.status_code}")
            r.raise_for_status()
            return r
        except requests.RequestException as exc:
            if attempt == retries - 1:
                raise
            wait = 2 ** attempt
            print(f"  retry in {wait}s ({exc})")
            time.sleep(wait)


def build_query(args):
    parts = ["(reviewed:true)", "(ec:*)"]
    if not args.keep_fragments:
        parts.append("(fragment:false)")
    if args.require_act_site:
        parts.append("(ft_act_site:*)")
    parts.append(f"(length:[{args.min_length} TO {args.max_length}])")
    return " AND ".join(parts)


def stream_accessions(query):
    r = get(f"{API}/stream", {"query": query, "format": "list"})
    accessions = sorted(set(r.text.split()))
    return accessions, r.headers.get("X-UniProt-Release"), r.headers.get("X-UniProt-Release-Date")


def fetch_records(accessions):
    frames = []
    for i in range(0, len(accessions), BATCH):
        chunk = accessions[i:i + BATCH]
        r = get(f"{API}/accessions",
                {"accessions": ",".join(chunk), "fields": ",".join(FIELDS), "format": "tsv"})
        df = pd.read_csv(io.StringIO(r.text), sep="\t", dtype=str).fillna("")
        df.columns = FIELDS
        frames.append(df)
        print(f"  fetched {min(i + BATCH, len(accessions))}/{len(accessions)}")
        time.sleep(0.2)
    return pd.concat(frames, ignore_index=True)


def parse_act_sites(text):
    """Return ACT_SITE positions and whether any has experimental evidence.

    UniProt writes e.g. 'ACT_SITE 57; /note="..."; /evidence="ECO:0000269|PubMed:1"'.
    Positions are UniProt sequence numbering, not PDB residue numbering.
    """
    if not text:
        return [], False
    positions, experimental = [], False
    for block in re.split(r"(?=ACT_SITE )", text):
        m = re.match(r"ACT_SITE\s+([<>?]?\d+(?:\.\.[<>?]?\d+)?)", block)
        if m:
            positions.append(m.group(1))
            experimental |= EXPERIMENTAL_ECO in block
    return positions, experimental


def annotate(df):
    df["length"] = df["length"].astype(int)
    df["ec_numbers"] = df["ec"].str.split(r";\s*").apply(lambda xs: [x for x in xs if x])
    df["n_ec"] = df["ec_numbers"].str.len()
    df["ec_class"] = df["ec_numbers"].apply(lambda xs: xs[0].split(".")[0] if xs else "")
    parsed = df["ft_act_site"].apply(parse_act_sites)
    df["act_site_positions"] = parsed.str[0].apply(",".join)
    df["n_act_sites"] = parsed.str[0].str.len()
    df["act_site_experimental"] = parsed.str[1]
    # Missing ACT_SITE means "not annotated", not "no active site".
    df["act_site_status"] = df["n_act_sites"].map(lambda n: "annotated" if n else "unknown")
    df["pdb_ids"] = df["xref_pdb"].str.rstrip(";")
    df["has_pdb"] = df["pdb_ids"] != ""
    return df.drop(columns=["ec_numbers", "xref_pdb"])


def write_fasta(df, path):
    with open(path, "w") as fh:
        for acc, seq in zip(df["accession"], df["sequence"]):
            fh.write(f">{acc}\n")
            fh.write("\n".join(seq[i:i + 60] for i in range(0, len(seq), 60)) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=300, help="sample size (0 = all matches)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min-length", type=int, default=50)
    p.add_argument("--max-length", type=int, default=1000,
                   help="caps structure size for the Foldseek/Folddisco steps")
    p.add_argument("--require-act-site", action="store_true",
                   help="keep only entries with a UniProt ACT_SITE feature")
    p.add_argument("--keep-fragments", action="store_true")
    p.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent / "data")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    query = build_query(args)
    print(f"Query: {query}")
    accessions, release, release_date = stream_accessions(query)
    print(f"{len(accessions):,} matching accessions (UniProt release {release})")
    (args.out / "all_matching_accessions.txt").write_text("\n".join(accessions) + "\n")

    n = len(accessions) if args.n == 0 else min(args.n, len(accessions))
    sample = sorted(random.Random(args.seed).sample(accessions, n))
    df = fetch_records(sample)

    # Guard: every row must be reviewed Swiss-Prot and carry an EC number.
    bad = df[(df["reviewed"] != "reviewed") | (df["ec"] == "")]
    assert bad.empty, f"non-Swiss-Prot or EC-less rows returned: {bad['accession'].tolist()}"
    missing = set(sample) - set(df["accession"])
    assert not missing, f"accessions not returned: {sorted(missing)[:10]}"

    df = annotate(df)
    df.to_csv(args.out / "enzymes.tsv", sep="\t", index=False)
    write_fasta(df, args.out / "enzymes.fasta")

    log = {
        "query": query,
        "uniprot_release": release,
        "uniprot_release_date": release_date,
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "args": {k: str(v) for k, v in vars(args).items()},
        "n_matching": len(accessions),
        "n_sampled": len(df),
        "n_with_act_site": int((df["n_act_sites"] > 0).sum()),
        "n_with_experimental_act_site": int(df["act_site_experimental"].sum()),
        "n_multi_ec": int((df["n_ec"] > 1).sum()),
        "n_with_pdb": int(df["has_pdb"].sum()),
        "ec_class_counts": df["ec_class"].value_counts().sort_index().to_dict(),
    }
    (args.out / "fetch_log.json").write_text(json.dumps(log, indent=2) + "\n")
    print(json.dumps({k: v for k, v in log.items() if k != "args"}, indent=2))


if __name__ == "__main__":
    main()

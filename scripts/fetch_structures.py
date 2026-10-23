"""Download AlphaFold DB models for the enzymes in data/enzymes.tsv.

For each accession: query the AlphaFold DB API, download the PDB model to
data/structures/{accession}.pdb, and record per-model confidence. Missing models
and sequence mismatches are kept in the manifest with a status, never dropped.

Writes:
  data/structures/{accession}.pdb
  data/structures_manifest.tsv   one row per enzyme (all of them, including missing)
  data/structures_log.json       counts and disk usage

pLDDT columns (AlphaFold confidence, 0-100, stored in the PDB B-factor column):
  plddt_mean        mean over CA atoms
  frac_plddt_lt50   fraction of residues < 50 (very low; often disordered)
  frac_plddt_lt70   fraction of residues < 70 (low)
  act_site_plddt    mean pLDDT at UniProt ACT_SITE residues (blank if none annotated)

Example:
  .venv/bin/python scripts/fetch_structures.py
  .venv/bin/python scripts/fetch_structures.py --limit 20
"""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
API = "https://alphafold.ebi.ac.uk/api/prediction/{}"


def get(url, retries=8):
    for attempt in range(retries):
        wait = min(2 ** attempt, 60)
        try:
            r = requests.get(url, timeout=120)
            if r.status_code == 429:  # rate limited: honour Retry-After when given
                wait = max(wait, int(r.headers.get("Retry-After", 5)))
                raise requests.HTTPError("HTTP 429")
            if r.status_code in (500, 502, 503, 504):
                raise requests.HTTPError(f"HTTP {r.status_code}")
            return r
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(wait)


def pick_entry(entries, accession):
    """Choose the full-length monomer model whose UniProt accession matches."""
    for e in entries:
        if (e.get("uniprotAccession") == accession and not e.get("isComplex")
                and e.get("entryId", "").endswith("-F1") and e.get("pdbUrl")):
            return e
    return None


def ca_plddt(pdb_path):
    """Return {residue number: pLDDT} from CA atoms (chain A)."""
    plddt = {}
    with open(pdb_path) as fh:
        for line in fh:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                plddt[int(line[22:26])] = float(line[60:66])
    return plddt


def expand_positions(text):
    """'57,102,10..12' -> [57, 102, 10, 11, 12]; uncertain ('?') positions are skipped."""
    out = []
    for tok in filter(None, str(text).split(",")):
        if "?" in tok:
            continue
        tok = tok.strip("<>")
        if ".." in tok:
            a, b = (int(x.strip("<>")) for x in tok.split(".."))
            out.extend(range(a, b + 1))
        else:
            out.append(int(tok))
    return out


def process(row, struct_dir, overwrite):
    acc = row.accession
    rec = {"accession": acc, "structure_source": "AlphaFoldDB", "status": "", "structure_path": "",
           "af_entry_id": "", "af_version": "", "af_global_plddt": "", "plddt_mean": "",
           "frac_plddt_lt50": "", "frac_plddt_lt70": "", "act_site_plddt": "",
           "seq_matches_uniprot": "", "n_residues_model": "", "msa_url": "", "notes": ""}
    try:
        r = get(API.format(acc))
        entries = r.json() if r.status_code == 200 else []
        entry = pick_entry(entries, acc) if entries else None
        if entry is None:
            rec.update(status="no_model", structure_source="", notes=f"API HTTP {r.status_code}")
            return rec
        rec.update(af_entry_id=entry["entryId"], af_version=entry.get("latestVersion", ""),
                   af_global_plddt=entry.get("globalMetricValue", ""), msa_url=entry.get("msaUrl", ""))
        # AlphaFold models are numbered 1..L on the UniProt sequence, so ACT_SITE
        # positions map directly, but only if the modelled sequence is the current one.
        match = entry.get("uniprotSequence") == row.sequence and entry.get("uniprotStart") == 1
        rec["seq_matches_uniprot"] = match

        dst = struct_dir / f"{acc}.pdb"
        if overwrite or not dst.exists():
            d = get(entry["pdbUrl"])
            d.raise_for_status()
            dst.write_bytes(d.content)
        rec["structure_path"] = str(dst.relative_to(ROOT))

        plddt = ca_plddt(dst)
        values = pd.Series(plddt)
        rec.update(plddt_mean=round(values.mean(), 2),
                   frac_plddt_lt50=round((values < 50).mean(), 3),
                   frac_plddt_lt70=round((values < 70).mean(), 3),
                   n_residues_model=len(values))
        sites = [plddt[p] for p in expand_positions(row.act_site_positions) if p in plddt]
        if sites:
            rec["act_site_plddt"] = round(sum(sites) / len(sites), 2)
        rec["status"] = "ok" if match else "sequence_mismatch"
        if not match:
            rec["notes"] = "model sequence differs from current UniProt; residue numbering unverified"
    except (requests.RequestException, ValueError, KeyError) as exc:
        rec.update(status="error", notes=str(exc)[:200])
    return rec


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--enzymes", type=Path, default=ROOT / "data" / "enzymes.tsv")
    p.add_argument("--out", type=Path, default=ROOT / "data")
    p.add_argument("--limit", type=int, default=0, help="only the first N enzymes (0 = all)")
    p.add_argument("--workers", type=int, default=2, help="parallel requests; the API rate-limits above ~2")
    p.add_argument("--overwrite", action="store_true", help="re-download existing files")
    args = p.parse_args()

    enzymes = pd.read_csv(args.enzymes, sep="\t", dtype=str).fillna("")
    if args.limit:
        enzymes = enzymes.head(args.limit)
    struct_dir = args.out / "structures"
    struct_dir.mkdir(parents=True, exist_ok=True)

    rows = list(enzymes.itertuples(index=False))
    with ThreadPoolExecutor(args.workers) as pool:
        records = []
        for i, rec in enumerate(pool.map(lambda r: process(r, struct_dir, args.overwrite), rows), 1):
            records.append(rec)
            if i % 25 == 0 or i == len(rows):
                print(f"  {i}/{len(rows)}")

    manifest = pd.DataFrame(records)
    manifest.to_csv(args.out / "structures_manifest.tsv", sep="\t", index=False)

    ok = manifest[manifest.status == "ok"]
    plddt = pd.to_numeric(ok.plddt_mean)
    site = pd.to_numeric(ok.act_site_plddt, errors="coerce").dropna()
    log = {
        "n_enzymes": len(manifest),
        "status_counts": manifest.status.value_counts().to_dict(),
        "plddt_mean_median": round(plddt.median(), 2) if len(ok) else None,
        "n_plddt_mean_lt70": int((plddt < 70).sum()),
        "n_with_act_site_plddt": len(site),
        "n_act_site_plddt_lt70": int((site < 70).sum()),
        "structures_mb": round(sum(f.stat().st_size for f in struct_dir.glob("*.pdb")) / 1e6, 1),
    }
    (args.out / "structures_log.json").write_text(json.dumps(log, indent=2) + "\n")
    print(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()

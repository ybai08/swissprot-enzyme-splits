# Swiss-Prot enzyme dataset: leakage-aware train/val/test splits

First mini coding project. Build a dataset of reviewed Swiss-Prot enzymes for an
active-site ML model (function, substrate specificity, cofactor dependency, ...),
then split it 80/10/10 into train/validation/test while minimizing data leakage.

## Leakage to prevent

| Type | Tool | Example |
|---|---|---|
| Identical proteins | dedup / MMseqs2 high identity | same protein in two splits |
| Sequence similarity | MMseqs2 | homologs across splits |
| Structural similarity | Foldseek | similar 3D fold, different sequence |
| Active-site similarity | Folddisco | similar catalytic motif, different fold |

Assign whole groups (not single proteins) to one split. Re-check cross-split
similarity after splitting.

## Layout

```
notebooks/   starter Colab notebook (AI-generated scaffold)
docs/        exercise guide (.docx + .txt copy)
data/        fetched metadata and FASTA; structures/ is not committed (72 MB),
             rebuild with scripts/fetch_structures.py
results/     cluster TSVs, manifests, split comparison tables; the Folddisco
             index is not committed, rebuild with motif_groups.py --reindex
```

## Setup

```bash
uv venv .venv && uv pip install -p .venv -r requirements.txt
```

## Step 1: fetch enzymes

```bash
.venv/bin/python scripts/fetch_enzymes.py --n 300 --seed 0
```

Streams every accession matching
`(reviewed:true) AND (ec:*) AND (fragment:false) AND (length:[50 TO 1000])`,
draws a seeded random sample, then fetches metadata in batches. Flags:
`--require-act-site`, `--keep-fragments`, `--min-length`, `--max-length`, `--n 0` (all).

Outputs in `data/`:

- `enzymes.tsv`: one row per enzyme; adds `ec_class`, `n_ec`, `act_site_positions`
  (UniProt numbering), `act_site_status` (`annotated` / `unknown`), `act_site_experimental`, `pdb_ids`.
- `enzymes.fasta`: sequences keyed by accession.
- `all_matching_accessions.txt`: the full population the sample was drawn from.
- `fetch_log.json`: query, UniProt release, counts.

Run of 2026-10-23 (release 2026_03): 266,683 matches; of 300 sampled, 100 have
ACT_SITE, but only 3 cite experimental evidence (ECO:0000269). The rest are
inferred by similarity (ECO:0000250) or rules (ECO:0000255). That matters if
active sites are labels.

## Step 2: sequence clusters (MMseqs2)

```bash
brew install mmseqs2   # tested with release 18-8cc5c
.venv/bin/python scripts/cluster_sequences.py
```

Runs `mmseqs easy-cluster` at `--min-seq-id` 0.3 / 0.5 / 0.9 with `-c 0.8 --cov-mode 0`
(both sequences at least 80% covered) and `-s 7.5` (high sensitivity for remote homologs).
Override with `--ids`, `--coverage`, `--cov-mode`, `--sensitivity`.

Outputs:

- `results/mmseqs/seq{30,50,90}_cluster.tsv` and `_rep_seq.fasta`: raw MMseqs2 output.
- `results/seq_clusters.tsv`: per accession, the cluster representative at each threshold.
- `results/seq_cluster_summary.tsv`: clusters, singletons, largest cluster per threshold.

Result on the 300-enzyme sample:

| threshold | clusters | singletons | largest |
|---|---|---|---|
| 30% | 266 | 237 | 4 |
| 50% | 282 | 266 | 4 |
| 90% | 300 | 300 | 1 |

A random 300 out of 266k is very diverse: no near-duplicates at 90%, and only
small ortholog groups (same EC) at 30–50%. Clusters nest cleanly (each 90% cluster
sits inside one 50% cluster, and each 50% cluster inside one 30% cluster). Sequence
leakage is small at this size. A larger N, or sampling whole families, would stress-test the split.

## Step 3: predicted structures (AlphaFold DB)

```bash
.venv/bin/python scripts/fetch_structures.py            # --limit 20 to test
```

For each accession: query the AlphaFold DB API, keep the full-length monomer
model (`AF-{acc}-F1`), download it to `data/structures/{accession}.pdb`, and
read per-residue pLDDT from the CA B-factors. Downloads are cached; rerunning
only re-queries the API. The API rate-limits (HTTP 429) above ~2 parallel
requests, so `--workers` defaults to 2 and retries honour `Retry-After`.

`data/structures_manifest.tsv` has one row per enzyme, including those without a model:

- `status`: `ok`, `no_model`, `sequence_mismatch` (model sequence differs from
  current UniProt, so ACT_SITE numbering is unverified), or `error`.
- `plddt_mean`, `frac_plddt_lt50`, `frac_plddt_lt70`, `af_global_plddt`.
- `act_site_plddt`: mean pLDDT at the ACT_SITE residues. AlphaFold models use
  UniProt numbering 1..L, so positions map directly when `status` is `ok`.
- `msa_url`: AlphaFold's own MSA (`.a3m`), usable for the optional MSA step.

Result on the 300-enzyme sample (72 MB):

- 298 `ok`, 0 sequence mismatches, 2 `no_model` (HTTP 404):
  - P58866 (MtmB2): contains pyrrolysine, a non-standard amino acid.
  - E7LXI9: a non-reference yeast strain (VIN 13).
- Median `plddt_mean` 94.1; only Q6ERS0 is below 70.
- All 100 ACT_SITE enzymes have a model. Three have active-site pLDDT below 70:
  P05302 (68.8), P40999 (63.5), Q759Y5 (58.2). Their motif geometry is less trustworthy for Folddisco.

## Step 4: structure clusters (Foldseek)

Foldseek is not in Homebrew. Install the official macOS binary (tested with
commit `463739e`):

```bash
mkdir -p ~/.local/opt && curl -sSfL https://mmseqs.com/foldseek/foldseek-osx-universal.tar.gz | tar -xz -C ~/.local/opt
ln -sf ~/.local/opt/foldseek/bin/foldseek ~/.local/bin/foldseek
```

```bash
.venv/bin/python scripts/cluster_structures.py          # ~30 s for 298 models
```

Clusters only models with `status == ok`, using two presets:

- `fold_e3_c80`: `-e 1e-3 -c 0.8 --cov-mode 0`. Homology-level match over most of both proteins.
- `fold_tm50_c50`: `-e 10 -c 0.5 --cov-mode 0 --tmscore-threshold 0.5 --exact-tmscore 1`.
  Same fold (TM > 0.5) over half the protein. Looser; catches shared domains.

`--mask-bfactor 50` ignores low-pLDDT residues when seeding. It is off by default.

Outputs:

- `results/foldseek/{preset}_cluster.tsv`: raw Foldseek output. IDs are already accessions.
- `results/struct_clusters.tsv`: per accession, the cluster representative per
  preset. Blank means no structure, so the protein is structurally unassessed.
- `results/struct_cluster_summary.tsv`: per preset, the cluster stats and how many
  structure clusters join several 30%-identity sequence clusters.

Result on the 298 models:

| preset | clusters | singletons | largest | clusters joining seq30 clusters (proteins) |
|---|---|---|---|---|
| fold_e3_c80 | 216 | 169 | 8 | 30 (91) |
| fold_tm50_c50 | 200 | 152 | 15 | 32 (110) |

Compare 266 clusters at 30% sequence identity. Structure links about a third of
the proteins that sequence clustering keeps apart, so sequence-only splits leak folds.
Structure clusters also mix functions: 16 of 47 multi-member `fold_e3_c80`
clusters span different EC sub-subclasses (x.y.z), and 8 span different EC classes.
The largest are classic superfolds:

- Rossmann-fold dehydrogenases plus glycosyltransferases (8 proteins, `fold_e3_c80`).
- TIM barrels (15 proteins, `fold_tm50_c50`): TIM, HisF, IGPS, transaldolase,
  aldolases, biotin synthase and others.
- Class II aminoacyl-tRNA synthetases (7 proteins).

A shared fold does not imply shared chemistry. This is why active-site similarity
(Folddisco) is a separate axis.

## Step 5: active-site motif groups (Folddisco)

Install the official macOS binary (tested with commit `3e193e3`):

```bash
curl -sSfL https://mmseqs.com/folddisco/folddisco-macos-universal.tar.gz | tar -xz -C ~/.local/opt
ln -sf ~/.local/opt/folddisco/bin/folddisco ~/.local/bin/folddisco
```

```bash
.venv/bin/python scripts/motif_groups.py --reindex     # ~1 min; later runs reuse the index
```

Method:

1. **Index.** Index the 298 `ok` models into `results/folddisco/index` (66 MB).
   The index points at `results/folddisco/input/` (symlinks), and Folddisco re-reads
   those files at query time, so keep that folder.
2. **Build motifs.** Folddisco hashes residue pairs, so a motif needs at least 3 residues.
   Only 8 of the 100 annotated enzymes have 3 or more ACT_SITE residues. Motifs with 1–2
   ACT_SITE residues get UniProt BINDING residues within 8 Å (heavy-atom distance) of the
   active site, nearest first, up to 6 residues in total.
   - 74 motifs are queryable: 29 pure ACT_SITE and 45 topped up with BINDING residues.
   - 25 stay below 3 residues.
   - 1 is skipped for low pLDDT (P40999, 63.5).
3. **Query.** Search every motif against the whole index with two presets:
   - `strict`: exact residue types.
   - `lenient`: `--aa-subst group`, so residues may swap within a chemical class (D/E, R/H/K, ...).

   Both presets require every residue matched (80% for motifs of 5+ residues) at RMSD ≤ 1.0 Å.
4. **Cap small motifs.** For 3-residue motifs, RMSD is capped at 0.55 Å
   (`--small-motif-rmsd`). On this sample, true 3-residue hits (HExxH zinc proteases,
   HAD DxD phosphatases) were ≤ 0.51 Å. Unrelated ones (an acidic triplet matching EF-Tu,
   Leu-tRNA ligase and SEY1) were ≥ 0.63 Å. The cutoff is calibrated on this sample only;
   recheck it if N grows.
5. **Drop generic motifs.** A query that hits more than 5% of all structures counts as
   generic, and its edges are dropped (`--max-hit-frac`).
6. **Group.** Each accepted hit is an edge. Connected components form the motif groups.

Outputs:

- `results/folddisco/motifs.tsv`: per enzyme, the motif residues, residue types,
  source (`act_site` / `act_site+binding`), pLDDT and query status.
- `results/folddisco/{strict,lenient}_hits.tsv`: every accepted hit, with RMSD and matched residues.
- `results/motif_groups.tsv`: per accession, `motif_status`, `motif_strict` and `motif_lenient`.
  - `motif_status` is `queried`, `target_only`, or `no_structure` (group left blank).
  - `target_only`: no usable motif of its own, but the protein was searched as a target.
    Its active site is still unknown.
- `results/motif_group_summary.tsv`.

Result:

| preset | queries with hits | edges | multi-member groups (proteins) | largest | groups spanning >1 fold |
|---|---|---|---|---|---|
| strict | 14 | 18 | 7 (18) | 5 | 4 of 7 |
| lenient | 22 | 50 | 14 (52) | 20 | 10 of 14 |

Strict groups find shared catalytic sites in different folds (`fold_tm50_c50`
clusters), which Foldseek cannot see:

- Zinc metalloproteases (HExxH): HtpX, FtsH, ADAM, and a fungal extracellular protease.
  Five proteins in 4 folds.
- HAD-family phosphatases and P-type ATPase (DxD).
- Calcineurin and LpxH: both have binuclear metal phosphoesterase sites.

Lenient groups over-merge. One 20-member group joins zinc proteases with
methyltransferases, kinases and EF-Tu through residue-class swaps. `strict` is the
recommended grouping; `lenient` is a conservative upper bound.

Limits:

- 224 of 300 enzymes have no queryable motif, so active-site leakage for them is checked
  only one way (as targets).
- Most motifs rest on similarity- or rule-based annotations, not experimental evidence.

## Step 6: splits, comparison and leakage audit

```bash
.venv/bin/python scripts/make_splits.py          # ~6 s; --strategy picks the manifest's split
```

Strategies. Each strategy defines groups, and every group goes whole into one split:

| strategy | groups from |
|---|---|
| random | each protein alone (naive baseline) |
| sequence | MMseqs2 `seq30` clusters |
| structure | Foldseek `fold_tm50_c50` clusters |
| motif | Folddisco `motif_strict` groups |
| combined | connected components of `seq30` ∪ `fold_tm50_c50` ∪ `motif_strict` |
| combined_lenient | same, with `motif_lenient` |

In `combined`, if A shares a sequence cluster with B and B shares a motif group with C,
then A, B and C end up in one group.

Assignment: groups are shuffled (seeded), sorted largest first, and each goes to the
split furthest below its target. Every strategy hits exactly 240 / 30 / 30.

Audit: the audit is independent of the clusters used to build the splits. It uses
all-vs-all searches (`results/audit/`), which also catch partial-coverage matches
that clustering ignores. A val/test protein counts as leaked if it has, against any
train protein:

- `seq_any`: an MMseqs2 hit with E < 1e-3.
- `seq30`: ≥ 30% identity over ≥ 50% of the held-out protein.
- `fold`: a Foldseek hit with TM-score ≥ 0.5, normalized by the held-out protein.
- `motif`: a strict Folddisco hit.

Fraction of **test** proteins leaked, mean over 10 seeds (`results/split_comparison.tsv`):

| strategy | groups | largest | seq_any | seq30 | fold | motif | any |
|---|---|---|---|---|---|---|---|
| random | 300 | 1 | 0.35 | 0.20 | 0.39 | 0.04 | **0.46** |
| sequence | 266 | 4 | 0.18 | 0.04 | 0.22 | 0.06 | 0.30 |
| structure | 202 | 15 | 0.01 | 0.00 | 0.02 | 0.03 | 0.06 |
| motif | 289 | 5 | 0.31 | 0.18 | 0.38 | 0.00 | 0.42 |
| **combined** | 195 | 15 | 0.02 | 0.00 | 0.03 | 0.00 | **0.05** |
| combined_lenient | 172 | 40 | 0.01 | 0.00 | 0.02 | 0.00 | 0.03 |

Findings:

- A random split leaks for almost half the test set. Over a third of test proteins
  share a fold with a train protein.
- Sequence clustering alone still leaves 22% fold leakage.
- Structure clustering alone also removes most sequence leakage, because homologs
  share folds. It does not remove motif leakage: 3% on test, 7% on val. That is the
  convergent active sites from step 5.
- Each single axis fails on at least one leakage type. Only `combined` brings all
  four near zero, at the same 80/10/10 sizes.
- `combined_lenient` shows no clear gain. Test leakage is 0.033 vs 0.047, but val
  leakage is 0.047 vs 0.033, and its largest group is 2.7× bigger.
- The seed-0 `combined` split has two residual leaks, both weak:
  - Q3BAK5 (test) to a train dehydrogenase: 24% identity over 36% coverage.
  - Q1I6F8 (val) to a train NAD synthetase: TM 0.502.

  The audit searches catch partial-domain matches that the clusters (80% / 50%
  coverage) skip. To remove them, add the audit graph's edges when grouping.

`results/dataset_manifest.csv` (300 rows):

- Identity and labels: accession, names, organism, reviewed, EC label/class.
- Active-site data: ACT_SITE status, positions and evidence.
- Structure: status, path, pLDDT (overall and at the active site), MSA URL.
- Cluster labels from every axis, plus `motif_status` and motif residues.
- `split_group` and `split` (from `combined`, seed 0).
- One `split_<strategy>` column for every alternative.
- `notes`: flags no structure, inferred (non-experimental) ACT_SITE, low active-site
  pLDDT, and multiple EC numbers.

Balance (seed 0, `combined`):

- All 7 EC classes are in train. Test has all 7; val lacks class 7.
- Annotated active sites: 76 train / 10 val / 14 test.
- Length and pLDDT are similar across splits.

### What the split claims, and what can still leak

**Claim:** the test set measures generalization to enzymes with no detectable
sequence homolog (E < 1e-3), no shared fold (TM ≥ 0.5), and no shared annotated
catalytic geometry in train, among Swiss-Prot enzymes of 50–1000 residues.
Grouping makes every val/test protein a singleton family in this sample. That is
the point, but it is a distribution shift: scores will be lower than on a random split.

Can still leak or mislead:

- **Unannotated active sites.** 224 of 300 enzymes had no usable motif. Two
  enzymes with similar unannotated sites in different folds are not linked.
- **Annotation quality.** 97 of 100 ACT_SITE annotations are inferred, not
  experimental, so motif labels and motif groups inherit propagation errors.
- **Labels without structure.** EC labels are shared across groups by design, so a
  model can still learn "this EC number" from distant train enzymes. That is fair
  for function prediction, but say so.
- **Thresholds.** The 0.55 Å 3-residue cap was tuned on this sample. The TM 0.5 and
  30% identity cutoffs are conventions, not guarantees.
- **Population.** Homologs outside the 300 (the other 266k Swiss-Prot enzymes) are
  invisible here. A test protein's close relative may be in a pretrained model's data.

Scaling:

- **UniProt:** paginate or stream the full EC set.
- **Clustering:** cluster sequences first, then fetch structures only for seq90
  representatives. Cluster with Foldseek on AFDB Foldcomp DBs.
- **Folddisco:** run batch queries with `--threads`. Use M-CSA curated catalytic
  residues instead of rule-based ACT_SITE.
- **Group assignment:** expect a giant component at scale. Report its size; if it
  exceeds 80%, split by a stricter axis (seq50 / fold_e3_c80) inside it and note the
  weaker guarantee.

## Plan

1. Fetch reviewed EC-annotated enzymes from UniProt (start with 100–500; paginate for more).
2. MMseqs2 `easy-cluster` at several identities (0.3 / 0.5 / 0.9).
3. Fetch AlphaFold DB structures for a subset; record pLDDT / missing models.
4. Foldseek `easy-cluster` on structures.
5. Folddisco index + motif queries from ACT_SITE residues (check UniProt vs PDB numbering).
6. Merge sequence/structure/motif edges into one graph; connected components = split groups.
7. Compare strategies: random, sequence-aware, structure-aware, motif-aware, combined.
8. Audit: cross-split identity / fold / motif hits, EC balance, missing annotations.

## Deliverables

- `results/dataset_manifest.csv`: accession, EC, active-site status, structure source, cluster IDs, split.
- Cluster/group files from MMseqs2, Foldseek, Folddisco.
- Table comparing candidate splits.
- Short write-up: chosen split, generalization it measures, remaining leakage.

## Links

UniProt REST: https://www.uniprot.org/help/api_queries ·
MMseqs2: https://github.com/soedinglab/MMseqs2 ·
Foldseek: https://github.com/steineggerlab/foldseek ·
Folddisco: https://github.com/steineggerlab/folddisco ·
AlphaFold DB API: https://alphafold.ebi.ac.uk/api-docs ·
M-CSA: https://www.ebi.ac.uk/thornton-srv/m-csa/

"""
EDA script for the Business Entity Resolution Challenge.

WHAT THIS DOES
---------------
Reads train_source1/2/3.tsv and train_ground_truth.tsv, and produces:
  1. Row / unique-ID counts per source
  2. Country distribution + missing-value report per source
  3. Match-count distribution per Source 1 entity (how many S2/S3 matches each has)
  4. Singleton / no-match counts, duplicate checks
  5. Non-ASCII / script detection (flags transliteration, different alphabets)
  6. A single text report (eda_report.txt) + a few PNG charts, saved into ./eda_output/

HOW TO RUN
----------
From the `student_resource/` folder (the one containing `dataset/`):

    python3 eda.py

It will create a folder called `eda_output/` next to this script containing:
    - eda_report.txt   (all the numbers, human-readable)
    - match_count_distribution.png
    - country_distribution.png
"""

import os
import re
import unicodedata
from collections import Counter

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # so it works without a display / on a server
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# CONFIG — edit these if your folders are named differently
# ---------------------------------------------------------------------------
DATA_DIR = "dataset/train"
OUT_DIR = "eda_output"
SOURCE_FILES = {
    "S1": os.path.join(DATA_DIR, "train_source1.tsv"),
    "S2": os.path.join(DATA_DIR, "train_source2.tsv"),
    "S3": os.path.join(DATA_DIR, "train_source3.tsv"),
}
GT_FILE = os.path.join(DATA_DIR, "train_ground_truth.tsv")

os.makedirs(OUT_DIR, exist_ok=True)
report_lines = []


def log(msg=""):
    """Print to console AND remember it for the saved report file."""
    print(msg)
    report_lines.append(str(msg))


def section(title):
    log("\n" + "=" * 70)
    log(title)
    log("=" * 70)


# ---------------------------------------------------------------------------
# STEP 1 — Load everything
# ---------------------------------------------------------------------------
# WHY: sep="\t" is critical — the problem statement warns that reading
# without it silently produces one garbage column. dtype=str keeps IDs like
# "S1-00001" as strings, not something pandas tries to guess.
section("STEP 1: Loading data")

sources = {}
for name, path in SOURCE_FILES.items():
    if not os.path.exists(path):
        log(f"  [!] Missing file: {path} — skipping")
        continue
    df = pd.read_csv(path, sep="\t", dtype=str)
    sources[name] = df
    log(f"  Loaded {name}: {path}  -> {len(df):,} rows, columns: {list(df.columns)}")

gt = None
if os.path.exists(GT_FILE):
    gt = pd.read_csv(GT_FILE, sep="\t", dtype=str)
    log(f"  Loaded ground truth: {GT_FILE} -> {len(gt):,} rows")
else:
    log(f"  [!] Missing ground truth file: {GT_FILE}")


# ---------------------------------------------------------------------------
# STEP 2 — Basic counts per source
# ---------------------------------------------------------------------------
# WHY: sanity-check row counts and make sure entity_id is actually unique
# per source (duplicates here would break everything downstream).
section("STEP 2: Record counts per source")

for name, df in sources.items():
    n_rows = len(df)
    n_unique_ids = df["entity_id"].nunique()
    n_dupe_ids = n_rows - n_unique_ids
    log(f"  {name}: {n_rows:,} rows | {n_unique_ids:,} unique entity_id "
        f"| {n_dupe_ids:,} duplicate entity_id rows")
    # confirm the ID prefix matches the source, e.g. S1- ids only in source1
    prefixes = df["entity_id"].str.extract(r"^(S\d)-")[0].value_counts()
    log(f"       entity_id prefixes found: {dict(prefixes)}")


# ---------------------------------------------------------------------------
# STEP 3 — Country distribution + missing values
# ---------------------------------------------------------------------------
# WHY: tells you what countries exist in TRAIN (the problem statement says
# test adds France, which should NOT appear here — good to confirm) and how
# dirty the fields are (missing name/address/country breaks matching).
section("STEP 3: Country distribution & missing values")

for name, df in sources.items():
    log(f"\n  --- {name} ---")
    log(f"  Country value counts:\n{df['country'].value_counts(dropna=False).to_string()}")

    for col in ["business_name", "business_address", "country"]:
        if col not in df.columns:
            continue
        n_missing = df[col].isna().sum() + (df[col].astype(str).str.strip() == "").sum()
        pct = 100 * n_missing / len(df)
        log(f"  Missing/blank '{col}': {n_missing:,} ({pct:.2f}%)")


# ---------------------------------------------------------------------------
# STEP 4 — Match-count distribution per Source 1 entity
# ---------------------------------------------------------------------------
# WHY: this is the core shape of the matching problem — are most S1 entities
# singletons? Do some have 10+ matches? This directly informs both your
# blocking strategy (how wide a net to cast) and your F0.5 strategy (how
# aggressively to predict matches, since false positives are penalized 2x).
if gt is not None:
    section("STEP 4: Match-count distribution per Source 1 entity")

    def split_ids(cell):
        if pd.isna(cell) or str(cell).strip() == "":
            return []
        return [x.strip() for x in str(cell).split(",") if x.strip()]

    gt["match_list"] = gt["matched_entity_ids"].apply(split_ids)
    gt["n_matches"] = gt["match_list"].apply(len)

    log(f"  Total ground truth rows (S1 entities with labels): {len(gt):,}")
    log(f"  Match count stats:\n{gt['n_matches'].describe().to_string()}")
    log(f"\n  Distribution (0 matches, 1 match, 2 matches, ...):")
    log(gt["n_matches"].value_counts().sort_index().to_string())

    # split matches by which source they came from
    def count_by_source(match_list, prefix):
        return sum(1 for m in match_list if m.startswith(prefix))

    gt["n_s2_matches"] = gt["match_list"].apply(lambda lst: count_by_source(lst, "S2-"))
    gt["n_s3_matches"] = gt["match_list"].apply(lambda lst: count_by_source(lst, "S3-"))
    log(f"\n  Total S2 matches across all S1 entities: {gt['n_s2_matches'].sum():,}")
    log(f"  Total S3 matches across all S1 entities: {gt['n_s3_matches'].sum():,}")

    # chart
    plt.figure(figsize=(8, 5))
    counts = gt["n_matches"].value_counts().sort_index()
    counts.plot(kind="bar")
    plt.title("Number of S1 entities by match count")
    plt.xlabel("Number of matches (S2 + S3 combined)")
    plt.ylabel("Number of S1 entities")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "match_count_distribution.png"))
    plt.close()
    log(f"\n  Saved chart: {OUT_DIR}/match_count_distribution.png")


# ---------------------------------------------------------------------------
# STEP 5 — Singletons, no-match cases, duplicate matches
# ---------------------------------------------------------------------------
# WHY: singletons (0 matches) are scored specially — predicting them wrong
# is a guaranteed 0.0 for that row. You need to know how many there are.
# Also checking whether any S2/S3 id is claimed by more than one S1 entity —
# if so, matching isn't strictly one-to-one and your model needs to know that.
if gt is not None:
    section("STEP 5: Singletons, no-match cases, duplicate matches")

    n_singletons = (gt["n_matches"] == 0).sum()
    pct_singletons = 100 * n_singletons / len(gt)
    log(f"  Singletons (no match at all): {n_singletons:,} ({pct_singletons:.2f}% of S1 entities)")

    # duplicate source1_entity_id rows in ground truth itself (should be 0)
    dupe_gt_rows = gt["source1_entity_id"].duplicated().sum()
    log(f"  Duplicate source1_entity_id rows in ground truth: {dupe_gt_rows:,}")

    # does any S2/S3 id appear as a match for more than one S1 entity?
    all_matches = Counter()
    for lst in gt["match_list"]:
        for m in lst:
            all_matches[m] += 1
    multi_claimed = {k: v for k, v in all_matches.items() if v > 1}
    log(f"  S2/S3 IDs matched to MORE than one S1 entity: {len(multi_claimed):,}")
    if multi_claimed:
        sample = list(multi_claimed.items())[:10]
        log(f"    Example (id -> how many S1 entities claim it): {sample}")

    # exact duplicate business_name + business_address within each source
    for name, df in sources.items():
        if "business_name" not in df.columns or "business_address" not in df.columns:
            continue
        dupe_mask = df.duplicated(subset=["business_name", "business_address"], keep=False)
        log(f"  {name}: {dupe_mask.sum():,} rows share an exact name+address with another row")


# ---------------------------------------------------------------------------
# STEP 6 — Script / language detection
# ---------------------------------------------------------------------------
# WHY: business_name / business_address may contain non-Latin scripts
# (Devanagari for India, accented French, etc). This affects how you build
# string-similarity features later — you may need transliteration-aware
# comparison, not just plain Levenshtein/Jaccard on raw text.
section("STEP 6: Script / non-ASCII character detection")

def detect_scripts(text):
    """Return a set of unicode script names found in the text (rough heuristic)."""
    scripts = set()
    if not isinstance(text, str):
        return scripts
    for ch in text:
        if ch.isalpha() and ord(ch) > 127:
            try:
                name = unicodedata.name(ch)
                # crude bucket by first word of the unicode character name
                scripts.add(name.split(" ")[0])
            except ValueError:
                pass
    return scripts

for name, df in sources.items():
    if "business_name" not in df.columns:
        continue
    non_ascii_mask = df["business_name"].apply(
        lambda x: isinstance(x, str) and any(ord(c) > 127 for c in x)
    )
    n_non_ascii = non_ascii_mask.sum()
    pct = 100 * n_non_ascii / len(df)
    log(f"  {name}: {n_non_ascii:,} business_name values ({pct:.2f}%) contain non-ASCII characters")

    if n_non_ascii > 0:
        all_scripts = Counter()
        for txt in df.loc[non_ascii_mask, "business_name"].head(2000):  # cap for speed
            all_scripts.update(detect_scripts(txt))
        top = all_scripts.most_common(10)
        log(f"       Most common non-ASCII character types: {top}")


# ---------------------------------------------------------------------------
# STEP 7 — Country distribution chart (all sources combined)
# ---------------------------------------------------------------------------
section("STEP 7: Saving country distribution chart")

fig, axes = plt.subplots(1, len(sources), figsize=(5 * len(sources), 4))
if len(sources) == 1:
    axes = [axes]
for ax, (name, df) in zip(axes, sources.items()):
    df["country"].value_counts(dropna=False).plot(kind="bar", ax=ax)
    ax.set_title(name)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "country_distribution.png"))
plt.close()
log(f"  Saved chart: {OUT_DIR}/country_distribution.png")


# ---------------------------------------------------------------------------
# STEP 8 — Save the full text report
# ---------------------------------------------------------------------------
report_path = os.path.join(OUT_DIR, "eda_report.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))

print(f"\n\nDONE. Full report saved to: {report_path}")
print(f"Charts saved to: {OUT_DIR}/")
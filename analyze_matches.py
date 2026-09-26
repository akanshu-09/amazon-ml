from pathlib import Path
import pandas as pd
import re
import unicodedata

TRAIN = Path(
    r"C:\Users\anany\Desktop\amazon ml\student_resource\dataset\train"
)

print("Loading data...")

S1 = pd.read_csv(TRAIN / "train_source1.tsv", sep="\t")
S2 = pd.read_csv(TRAIN / "train_source2.tsv", sep="\t")
S3 = pd.read_csv(TRAIN / "train_source3.tsv", sep="\t")
GT = pd.read_csv(TRAIN / "train_ground_truth.tsv", sep="\t")


def norm(x):
    if pd.isna(x):
        return ""

    x = unicodedata.normalize("NFKC", str(x).lower())
    x = re.sub(r"[^\w\s]", " ", x)
    x = re.sub(r"\s+", " ", x)

    return x.strip()


print("Normalizing names...")

S1["name_norm"] = S1["business_name"].map(norm)
S2["name_norm"] = S2["business_name"].map(norm)
S3["name_norm"] = S3["business_name"].map(norm)

S1["country_norm"] = (
    S1["country"].fillna("").astype(str).str.lower().str.strip()
)

S2["country_norm"] = (
    S2["country"].fillna("").astype(str).str.lower().str.strip()
)

S3["country_norm"] = (
    S3["country"].fillna("").astype(str).str.lower().str.strip()
)


# ------------------------------------------------------------
# LOOKUPS
# ------------------------------------------------------------

print("Creating lookups...")

s1_lookup = S1.set_index("entity_id")
s2_lookup = S2.set_index("entity_id")
s3_lookup = S3.set_index("entity_id")


# ------------------------------------------------------------
# ANALYSIS
# ------------------------------------------------------------

exact_name = 0
same_country = 0
different_country = 0
checked = 0

print("Analyzing ground-truth matches...")

for _, row in GT.iterrows():

    s1_id = row["source1_entity_id"]

    # Safety check
    if pd.isna(s1_id) or s1_id not in s1_lookup.index:
        continue

    s1_row = s1_lookup.loc[s1_id]

    s1_name = s1_row["name_norm"]
    s1_country = s1_row["country_norm"]

    # Handle missing ground truth
    if pd.isna(row["matched_entity_ids"]):
        continue

    matched_ids = str(row["matched_entity_ids"]).split(",")

    for entity_id in matched_ids:

        entity_id = entity_id.strip()

        # Ignore empty/nan values
        if not entity_id or entity_id.lower() == "nan":
            continue

        # Find target
        if entity_id.startswith("S2-"):

            if entity_id not in s2_lookup.index:
                continue

            target = s2_lookup.loc[entity_id]

        elif entity_id.startswith("S3-"):

            if entity_id not in s3_lookup.index:
                continue

            target = s3_lookup.loc[entity_id]

        else:
            continue

        checked += 1

        # Name comparison
        if (
            s1_name
            and s1_name == target["name_norm"]
        ):
            exact_name += 1

        # Country comparison
        if s1_country == target["country_norm"]:
            same_country += 1
        else:
            different_country += 1

        # Stop after 100,000 actual pairs
        if checked >= 100000:
            break

    if checked >= 100000:
        break


# ------------------------------------------------------------
# RESULTS
# ------------------------------------------------------------

print()
print("=" * 60)
print("FIRST 100,000 TRUE MATCH PAIRS")
print("=" * 60)

print("Pairs checked:", checked)

if checked > 0:

    print(
        "Exact normalized name:",
        round(exact_name / checked * 100, 2),
        "%"
    )

    print(
        "Same country:",
        round(same_country / checked * 100, 2),
        "%"
    )

    print(
        "Different country:",
        round(different_country / checked * 100, 2),
        "%"
    )

else:
    print("No valid ground-truth pairs were found.")
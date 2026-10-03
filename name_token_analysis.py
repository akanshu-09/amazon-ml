from pathlib import Path
import pandas as pd
import re
import unicodedata

TRAIN = Path(
    r"C:\Users\anany\Desktop\amazon ml\student_resource\dataset\train"
)

print("Loading data...")

S1 = pd.read_csv(
    TRAIN / "train_source1.tsv",
    sep="\t",
    usecols=["entity_id", "business_name"]
)

S2 = pd.read_csv(
    TRAIN / "train_source2.tsv",
    sep="\t",
    usecols=["entity_id", "business_name"]
)

S3 = pd.read_csv(
    TRAIN / "train_source3.tsv",
    sep="\t",
    usecols=["entity_id", "business_name"]
)

GT = pd.read_csv(
    TRAIN / "train_ground_truth.tsv",
    sep="\t"
)


def normalize(text):
    if pd.isna(text):
        return ""

    text = unicodedata.normalize("NFKC", str(text).lower())
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def tokens(text):
    return set(
        word
        for word in normalize(text).split()
        if len(word) >= 2
    )


print("Normalizing...")

S1["name_norm"] = S1["business_name"].map(normalize)
S2["name_norm"] = S2["business_name"].map(normalize)
S3["name_norm"] = S3["business_name"].map(normalize)


print("Creating lookups...")

s1_lookup = S1.set_index("entity_id")
s2_lookup = S2.set_index("entity_id")
s3_lookup = S3.set_index("entity_id")


# ------------------------------------------------------------
# Analyze first 100,000 true pairs
# ------------------------------------------------------------

checked = 0

shared_token = 0
shared_two_tokens = 0
exact_name = 0

print("Analyzing true matches...")

for _, row in GT.iterrows():

    s1_id = row["source1_entity_id"]

    if pd.isna(s1_id) or s1_id not in s1_lookup.index:
        continue

    if pd.isna(row["matched_entity_ids"]):
        continue

    s1_name = s1_lookup.loc[s1_id]["name_norm"]
    s1_tokens = set(s1_name.split())

    if not s1_tokens:
        continue

    matched_ids = str(
        row["matched_entity_ids"]
    ).split(",")

    for entity_id in matched_ids:

        entity_id = entity_id.strip()

        if not entity_id or entity_id.lower() == "nan":
            continue

        if entity_id.startswith("S2-"):

            if entity_id not in s2_lookup.index:
                continue

            target_name = s2_lookup.loc[
                entity_id
            ]["name_norm"]

        elif entity_id.startswith("S3-"):

            if entity_id not in s3_lookup.index:
                continue

            target_name = s3_lookup.loc[
                entity_id
            ]["name_norm"]

        else:
            continue

        target_tokens = set(target_name.split())

        common = s1_tokens.intersection(target_tokens)

        checked += 1

        if s1_name == target_name:
            exact_name += 1

        if len(common) >= 1:
            shared_token += 1

        if len(common) >= 2:
            shared_two_tokens += 1

        if checked >= 100000:
            break

    if checked >= 100000:
        break


# ------------------------------------------------------------
# Results
# ------------------------------------------------------------

print()
print("=" * 60)
print("NAME TOKEN ANALYSIS")
print("=" * 60)

print("Pairs checked:", checked)

if checked > 0:

    print(
        "Exact normalized name:",
        round(exact_name / checked * 100, 2),
        "%"
    )

    print(
        "At least 1 shared token:",
        round(shared_token / checked * 100, 2),
        "%"
    )

    print(
        "At least 2 shared tokens:",
        round(shared_two_tokens / checked * 100, 2),
        "%"
    )
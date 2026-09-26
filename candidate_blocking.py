from pathlib import Path
import pandas as pd
import re
import unicodedata

# ============================================================
# PATHS
# ============================================================

TRAIN = Path(
    r"C:\Users\anany\Desktop\amazon ml\student_resource\dataset\train"
)

OUTPUT = Path(
    r"C:\Users\anany\Desktop\amazon_ml_challenge\output"
)

OUTPUT.mkdir(exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

# Test on 50,000 S1 records first
S1_SAMPLE = 50000


# ============================================================
# NORMALIZATION
# ============================================================

def normalize(text):

    if pd.isna(text):
        return ""

    text = unicodedata.normalize(
        "NFKC",
        str(text).lower()
    )

    text = re.sub(
        r"[^\w\s]",
        " ",
        text,
        flags=re.UNICODE
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


def get_tokens(text):

    return {
        token
        for token in text.split()
        if len(token) >= 2
    }


# ============================================================
# LOAD DATA
# ============================================================

print("Loading S1...")

s1 = pd.read_csv(
    TRAIN / "train_source1.tsv",
    sep="\t",
    nrows=S1_SAMPLE
)

print("S1:", len(s1))


print("Loading S2...")

s2 = pd.read_csv(
    TRAIN / "train_source2.tsv",
    sep="\t",
    usecols=[
        "entity_id",
        "business_name",
        "business_address",
        "country"
    ]
)

print("S2:", len(s2))


print("Loading S3...")

s3 = pd.read_csv(
    TRAIN / "train_source3.tsv",
    sep="\t",
    usecols=[
        "entity_id",
        "business_name",
        "business_address",
        "country"
    ]
)

print("S3:", len(s3))


# ============================================================
# COMBINE TARGET DATA
# ============================================================

targets = pd.concat(
    [s2, s3],
    ignore_index=True
)

del s2
del s3

print("Total S2 + S3:", len(targets))


# ============================================================
# NORMALIZE
# ============================================================

print("Normalizing...")

s1["name_norm"] = (
    s1["business_name"]
    .fillna("")
    .map(normalize)
)

s1["country_norm"] = (
    s1["country"]
    .fillna("")
    .astype(str)
    .str.lower()
    .str.strip()
)

targets["name_norm"] = (
    targets["business_name"]
    .fillna("")
    .map(normalize)
)

targets["country_norm"] = (
    targets["country"]
    .fillna("")
    .astype(str)
    .str.lower()
    .str.strip()
)


# ============================================================
# BUILD TOKEN INDEX
# ============================================================

print("Building token index...")

token_index = {}

for idx, name in enumerate(targets["name_norm"]):

    if idx % 500000 == 0:
        print("Indexed:", idx)

    tokens = get_tokens(name)

    for token in tokens:

        key = (
            targets.iloc[idx]["country_norm"],
            token
        )

        if key not in token_index:
            token_index[key] = []

        token_index[key].append(idx)


print("Token index created.")
print("Number of blocks:", len(token_index))


# ============================================================
# GENERATE CANDIDATES
# ============================================================

print("Generating candidates...")

candidate_rows = []

for i, row in s1.iterrows():

    if i % 5000 == 0:
        print("Processed:", i)

    country = row["country_norm"]
    tokens = get_tokens(row["name_norm"])

    candidate_indices = set()

    # Retrieve candidates sharing at least one
    # normalized name token AND country.
    for token in tokens:

        key = (country, token)

        if key in token_index:

            candidate_indices.update(
                token_index[key]
            )

    # Create candidate rows
    for idx in candidate_indices:

        target = targets.iloc[idx]

        candidate_rows.append(
            {
                "source1_entity_id":
                    row["entity_id"],

                "candidate_entity_id":
                    target["entity_id"]
            }
        )


# ============================================================
# SAVE
# ============================================================

candidates = pd.DataFrame(
    candidate_rows
)

print()
print("=" * 60)
print("CANDIDATE GENERATION COMPLETE")
print("=" * 60)

print(
    "Candidate pairs:",
    len(candidates)
)

print(
    "Unique S1:",
    candidates["source1_entity_id"].nunique()
)

print(
    "Average candidates per S1:",
    round(
        len(candidates) /
        max(
            candidates["source1_entity_id"].nunique(),
            1
        ),
        2
    )
)


output_file = OUTPUT / "candidate_pairs_token.tsv"

candidates.to_csv(
    output_file,
    sep="\t",
    index=False
)

print()
print("Saved:")
print(output_file)
"""
Feature Extraction and Matrix Construction for Entity Matching.

This module computes rich pairwise similarity features for (S1, Candidate) pairs
generated during candidate blocking and pruning.

Features extracted per pair:
1. Name features: Exact match, Levenshtein ratio, Token Jaccard, 3-gram Jaccard,
   Token overlap count, Length ratio, First token match.
2. Address features: Levenshtein ratio, Token Jaccard, 3-gram Jaccard, Token overlap count,
   Length ratio, Street number match, Generic postal code pattern match.
3. Global features: Country match, Country missing, Name missing, Address missing,
   Cheap similarity score.
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

# Add parent directory to sys.path to enable direct module execution
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import rapidfuzz.distance.Levenshtein as rf_lev
    def levenshtein_ratio(s1: str, s2: str) -> float:
        """Fast C++ normalized Levenshtein similarity in range [0.0, 1.0]."""
        if not s1 and not s2:
            return 1.0
        if not s1 or not s2:
            return 0.0
        return float(rf_lev.normalized_similarity(s1, s2))
except ImportError:
    import difflib
    def levenshtein_ratio(s1: str, s2: str) -> float:
        """Fallback normalized similarity using difflib SequenceMatcher."""
        if not s1 and not s2:
            return 1.0
        if not s1 or not s2:
            return 0.0
        return float(difflib.SequenceMatcher(None, s1, s2).ratio())

try:
    from candidates.blocking import (
        extract_address_tokens,
        extract_char_ngrams,
        extract_name_tokens,
        load_ground_truth,
        normalize_business_name,
        normalize_country,
        normalize_text,
    )
    from candidates.prune import cheap_similarity_score, jaccard_similarity
except ImportError:
    from src.candidates.blocking import (
        extract_address_tokens,
        extract_char_ngrams,
        extract_name_tokens,
        load_ground_truth,
        normalize_business_name,
        normalize_country,
        normalize_text,
    )
    from src.candidates.prune import cheap_similarity_score, jaccard_similarity

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ==============================================================================
# Postal Code & Numeric Pattern Matchers
# ==============================================================================

# Generic postal code heuristic:
# Matches 5-to-6 digit numbers (US 5-digit zip, India 6-digit PIN, France 5-digit code)
# or alphanumeric patterns like UK/Canada/international codes (e.g. SW1A 1AA, K1A 0B1, 75008).
#
# LIMITATIONS:
# 1. Does not validate country-specific checksums or valid postal district ranges.
# 2. May occasionally match standalone 5-digit street numbers or PO Boxes if not formatted as postal.
# Keeping it general ensures zero hardcoding for open-set international test distributions.
GENERIC_POSTAL_PATTERN = re.compile(
    r"\b(\d{5,6}|[A-Z]\d[A-Z]\s?\d[A-Z]\d|[A-Z]{1,2}\d[A-Z\d]?\s?\d[A-Z]{2})\b",
    re.IGNORECASE,
)
DIGIT_ONLY_PATTERN = re.compile(r"\b\d+\b")


def extract_postal_codes(text: str) -> Set[str]:
    """Extract generic postal-code-like tokens from normalized text."""
    if not text:
        return set()
    matches = GENERIC_POSTAL_PATTERN.findall(text)
    return {m.replace(" ", "").upper() for m in matches if len(m.strip()) >= 4}


def extract_numeric_tokens(text: str) -> Set[str]:
    """Extract standalone digit tokens (likely street, building, or unit numbers)."""
    if not text:
        return set()
    return set(DIGIT_ONLY_PATTERN.findall(text))


# ==============================================================================
# Pre-Normalized Entity Profile for High-Throughput Feature Extraction
# ==============================================================================

class EntityFeatureProfile:
    """Pre-normalized and pre-tokenized entity data to avoid redundant per-pair parsing."""
    __slots__ = (
        "raw_name",
        "raw_addr",
        "raw_country",
        "norm_name",
        "norm_addr",
        "norm_country",
        "name_tokens",
        "name_ngrams",
        "name_first_token",
        "addr_tokens",
        "addr_ngrams",
        "addr_numbers",
        "addr_postal_codes",
        "is_name_missing",
        "is_addr_missing",
        "is_country_missing",
    )

    def __init__(self, raw_name: Optional[str], raw_addr: Optional[str], raw_country: Optional[str]):
        self.raw_name = "" if raw_name is None or pd.isna(raw_name) else str(raw_name).strip()
        self.raw_addr = "" if raw_addr is None or pd.isna(raw_addr) else str(raw_addr).strip()
        self.raw_country = "" if raw_country is None or pd.isna(raw_country) else str(raw_country).strip()

        self.is_name_missing = not bool(self.raw_name)
        self.is_addr_missing = not bool(self.raw_addr)
        self.is_country_missing = not bool(self.raw_country)

        self.norm_name = normalize_business_name(self.raw_name)
        self.norm_addr = normalize_text(self.raw_addr)
        self.norm_country = normalize_country(self.raw_country)

        self.name_tokens = extract_name_tokens(self.norm_name, min_len=2)
        self.name_ngrams = set(extract_char_ngrams(self.norm_name, n=3))
        self.name_first_token = self.name_tokens[0] if self.name_tokens else ""

        self.addr_tokens = extract_address_tokens(self.norm_addr, min_len=2)
        self.addr_ngrams = set(extract_char_ngrams(self.norm_addr, n=3))
        self.addr_numbers = extract_numeric_tokens(self.norm_addr)
        self.addr_postal_codes = extract_postal_codes(self.norm_addr)


# ==============================================================================
# Pair Feature Extraction Function
# ==============================================================================

def build_pair_features(s1_row: Any, cand_row: Any) -> Dict[str, Any]:
    """
    Computes a rich feature vector comparing an S1 entity against a candidate S2/S3 entity.
    
    Accepts either EntityFeatureProfile objects (recommended for speed) or dict/namedtuple/Series
    with keys ('business_name', 'business_address', 'country').
    
    Returns:
        dict of computed feature name -> feature value
    """
    # 1. Adapt input objects to profile representation
    if isinstance(s1_row, EntityFeatureProfile):
        p1 = s1_row
    else:
        p1 = EntityFeatureProfile(
            s1_row.get("business_name") if hasattr(s1_row, "get") else getattr(s1_row, "business_name", ""),
            s1_row.get("business_address") if hasattr(s1_row, "get") else getattr(s1_row, "business_address", ""),
            s1_row.get("country") if hasattr(s1_row, "get") else getattr(s1_row, "country", ""),
        )

    if isinstance(cand_row, EntityFeatureProfile):
        p2 = cand_row
    else:
        p2 = EntityFeatureProfile(
            cand_row.get("business_name") if hasattr(cand_row, "get") else getattr(cand_row, "business_name", ""),
            cand_row.get("business_address") if hasattr(cand_row, "get") else getattr(cand_row, "business_address", ""),
            cand_row.get("country") if hasattr(cand_row, "get") else getattr(cand_row, "country", ""),
        )

    features: Dict[str, Any] = {}

    # --------------------------------------------------------------------------
    # NAME FEATURES
    # --------------------------------------------------------------------------
    features["name_exact_match"] = int(bool(p1.norm_name and p1.norm_name == p2.norm_name))
    features["name_levenshtein_ratio"] = levenshtein_ratio(p1.norm_name, p2.norm_name)

    s1_name_tok_set = set(p1.name_tokens)
    cand_name_tok_set = set(p2.name_tokens)
    name_overlap_toks = s1_name_tok_set.intersection(cand_name_tok_set)

    features["name_jaccard_tokens"] = jaccard_similarity(s1_name_tok_set, cand_name_tok_set)
    features["name_jaccard_ngrams"] = jaccard_similarity(p1.name_ngrams, p2.name_ngrams)
    features["name_token_overlap_count"] = len(name_overlap_toks)

    # Length ratio: min(len1, len2) / max(len1, len2)
    len1_name, len2_name = len(p1.norm_name), len(p2.norm_name)
    max_len_name = max(len1_name, len2_name)
    features["name_length_ratio"] = (min(len1_name, len2_name) / max_len_name) if max_len_name > 0 else 0.0

    # First token match (handles core brand name matches when trailing words differ)
    features["name_first_token_match"] = int(
        bool(p1.name_first_token and p1.name_first_token == p2.name_first_token)
    )

    # --------------------------------------------------------------------------
    # ADDRESS FEATURES
    # --------------------------------------------------------------------------
    features["addr_levenshtein_ratio"] = levenshtein_ratio(p1.norm_addr, p2.norm_addr)

    s1_addr_tok_set = set(p1.addr_tokens)
    cand_addr_tok_set = set(p2.addr_tokens)
    addr_overlap_toks = s1_addr_tok_set.intersection(cand_addr_tok_set)

    features["addr_jaccard_tokens"] = jaccard_similarity(s1_addr_tok_set, cand_addr_tok_set)
    features["addr_jaccard_ngrams"] = jaccard_similarity(p1.addr_ngrams, p2.addr_ngrams)
    features["addr_token_overlap_count"] = len(addr_overlap_toks)

    len1_addr, len2_addr = len(p1.norm_addr), len(p2.norm_addr)
    max_len_addr = max(len1_addr, len2_addr)
    features["addr_length_ratio"] = (min(len1_addr, len2_addr) / max_len_addr) if max_len_addr > 0 else 0.0

    # Number match (e.g. matching street or unit numbers)
    features["addr_number_match"] = int(bool(p1.addr_numbers and (p1.addr_numbers & p2.addr_numbers)))

    # Postal pattern match
    has_matching_postal = bool(p1.addr_postal_codes and (p1.addr_postal_codes & p2.addr_postal_codes))
    features["addr_has_postal_pattern_match"] = int(has_matching_postal)

    # --------------------------------------------------------------------------
    # GLOBAL FEATURES
    # --------------------------------------------------------------------------
    features["country_match"] = int(bool(p1.norm_country and p1.norm_country == p2.norm_country))
    features["country_missing"] = int(bool(p1.is_country_missing or p2.is_country_missing))
    features["name_missing"] = int(bool(p1.is_name_missing or p2.is_name_missing))
    features["addr_missing"] = int(bool(p1.is_addr_missing or p2.is_addr_missing))

    # Cheap similarity score directly from prune.py logic
    features["cheap_similarity_score"] = cheap_similarity_score(
        p1.raw_name, p1.raw_addr, p2.raw_name, p2.raw_addr
    )

    return features


# ==============================================================================
# Feature Matrix Builder
# ==============================================================================

def build_feature_matrix(
    candidates_dict: Dict[str, List[str]],
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    ground_truth_dict: Optional[Dict[str, List[str]]] = None,
    log_every_n: int = 50000,
) -> pd.DataFrame:
    """
    Constructs the complete feature matrix DataFrame for all candidate pairs.
    
    Optimized for memory and speed:
    - Builds pre-normalized EntityFeatureProfile lookups up front.
    - Emits progress logs every log_every_n pairs.
    
    Args:
        candidates_dict: dict mapping s1_id -> list of candidate IDs
        s1_df: DataFrame containing S1 entities
        s2_df: DataFrame containing S2 entities
        s3_df: DataFrame containing S3 entities
        ground_truth_dict: Optional dict mapping s1_id -> list of true matching IDs
        log_every_n: Frequency of progress logging (default: 50,000 pairs)
        
    Returns:
        pandas.DataFrame with columns: s1_id, candidate_id, features..., label
    """
    start_time = time.time()
    total_pairs = sum(len(cands) for cands in candidates_dict.values())
    logging.info(
        f"--- Building Feature Matrix for {total_pairs:,} pairs ({len(candidates_dict):,} S1 entities) ---"
    )

    # Identify referenced IDs to only profile entities needed
    referenced_s1_ids = set(candidates_dict.keys())
    referenced_cand_ids: Set[str] = set()
    for cands in candidates_dict.values():
        referenced_cand_ids.update(cands)

    logging.info(
        f"Indexing {len(referenced_s1_ids):,} S1 profiles and {len(referenced_cand_ids):,} candidate profiles..."
    )

    # 1. Build profile lookup for S1
    s1_profiles: Dict[str, EntityFeatureProfile] = {}
    for eid, name, addr, country in zip(
        s1_df["entity_id"], s1_df["business_name"], s1_df["business_address"], s1_df["country"]
    ):
        if eid in referenced_s1_ids:
            s1_profiles[eid] = EntityFeatureProfile(name, addr, country)

    # 2. Build profile lookup for S2 and S3
    cand_profiles: Dict[str, EntityFeatureProfile] = {}
    for df in (s2_df, s3_df):
        for eid, name, addr, country in zip(
            df["entity_id"], df["business_name"], df["business_address"], df["country"]
        ):
            if eid in referenced_cand_ids:
                cand_profiles[eid] = EntityFeatureProfile(name, addr, country)

    logging.info("Extracting pairwise features...")

    rows: List[Dict[str, Any]] = []
    processed_count = 0
    loop_start_time = time.time()

    # Pre-index ground truth sets for O(1) membership test
    gt_lookup: Dict[str, Set[str]] = {}
    if ground_truth_dict is not None:
        gt_lookup = {s1_id: set(matches) for s1_id, matches in ground_truth_dict.items()}

    for s1_id, cand_ids in candidates_dict.items():
        s1_prof = s1_profiles.get(s1_id)
        if not s1_prof:
            continue

        s1_gt_set = gt_lookup.get(s1_id) if ground_truth_dict is not None else None

        for cand_id in cand_ids:
            cand_prof = cand_profiles.get(cand_id)
            if not cand_prof:
                continue

            pair_feats = build_pair_features(s1_prof, cand_prof)
            pair_feats["s1_id"] = s1_id
            pair_feats["candidate_id"] = cand_id

            # Assign label
            if ground_truth_dict is not None:
                pair_feats["label"] = 1 if (s1_gt_set is not None and cand_id in s1_gt_set) else 0
            else:
                pair_feats["label"] = np.nan

            rows.append(pair_feats)
            processed_count += 1

            if processed_count % log_every_n == 0:
                elapsed_loop = time.time() - loop_start_time
                rate = processed_count / max(elapsed_loop, 0.001)
                logging.info(
                    f"Processed {processed_count:,}/{total_pairs:,} pairs ({processed_count / total_pairs * 100:.1f}%) "
                    f"[{rate:,.0f} pairs/sec]"
                )

    total_elapsed = time.time() - start_time
    logging.info(
        f"Feature extraction finished in {total_elapsed:.2f}s ({len(rows):,} total records built)."
    )

    df = pd.DataFrame(rows)

    # Reorder columns so identifier and label columns are cleanly organized
    id_cols = ["s1_id", "candidate_id"]
    label_cols = ["label"]
    feature_cols = [c for c in df.columns if c not in id_cols and c not in label_cols]
    ordered_cols = id_cols + feature_cols + label_cols

    return df[ordered_cols]


# ==============================================================================
# Main / CLI Execution
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Build pairwise feature matrix for entity matching."
    )
    parser.add_argument(
        "--candidates",
        type=str,
        default="output/train_candidates_pruned.json",
        help="Path to pruned candidates JSON file (default: output/train_candidates_pruned.json)",
    )
    parser.add_argument(
        "--s1",
        type=str,
        default="dataset/train/train_source1.tsv",
        help="Path to S1 TSV file (default: dataset/train/train_source1.tsv)",
    )
    parser.add_argument(
        "--s2",
        type=str,
        default="dataset/train/train_source2.tsv",
        help="Path to S2 TSV file (default: dataset/train/train_source2.tsv)",
    )
    parser.add_argument(
        "--s3",
        type=str,
        default="dataset/train/train_source3.tsv",
        help="Path to S3 TSV file (default: dataset/train/train_source3.tsv)",
    )
    parser.add_argument(
        "--ground-truth",
        type=str,
        default="dataset/train/train_ground_truth.tsv",
        help="Path to ground truth TSV file (default: dataset/train/train_ground_truth.tsv)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/train_features.parquet",
        help="Path to save output features file (default: output/train_features.parquet)",
    )

    args = parser.parse_args()

    if not os.path.exists(args.candidates):
        logging.error(f"Candidates file not found at '{args.candidates}'.")
        sys.exit(1)

    logging.info(f"Loading candidates from {args.candidates}...")
    with open(args.candidates, "r", encoding="utf-8") as f:
        candidates_dict = json.load(f)

    logging.info("Loading source datasets...")
    s1_df = pd.read_csv(args.s1, sep="\t", dtype=str)
    s2_df = pd.read_csv(args.s2, sep="\t", dtype=str)
    s3_df = pd.read_csv(args.s3, sep="\t", dtype=str)

    ground_truth_dict = None
    if args.ground_truth and os.path.exists(args.ground_truth):
        logging.info(f"Loading ground truth labels from {args.ground_truth}...")
        ground_truth_dict = load_ground_truth(args.ground_truth)

    # Build feature matrix
    feature_df = build_feature_matrix(
        candidates_dict=candidates_dict,
        s1_df=s1_df,
        s2_df=s2_df,
        s3_df=s3_df,
        ground_truth_dict=ground_truth_dict,
    )

    # Print summary statistics
    total_pairs = len(feature_df)
    print("\n" + "=" * 70)
    print(" FEATURE MATRIX SUMMARY")
    print("=" * 70)
    print(f" Total Pairs Extracted:       {total_pairs:,}")
    print(f" Unique S1 Entities:          {feature_df['s1_id'].nunique():,}")
    print(f" Unique Candidates:           {feature_df['candidate_id'].nunique():,}")

    if "label" in feature_df.columns and not feature_df["label"].isna().all():
        pos_count = int(feature_df["label"].sum())
        neg_count = total_pairs - pos_count
        pos_rate = (pos_count / total_pairs * 100) if total_pairs > 0 else 0.0
        print(f" Positive Matches (Label=1):  {pos_count:,} ({pos_rate:.2f}%)")
        print(f" Negative Matches (Label=0):  {neg_count:,} ({100 - pos_rate:.2f}%)")
        print(f" Imbalance Ratio:             1 : {neg_count / max(pos_count, 1):.1f}")

    print("\n" + "-" * 70)
    print(" FEATURE DISTRIBUTIONS (Sanity Check)")
    print("-" * 70)
    feature_cols = [c for c in feature_df.columns if c not in ("s1_id", "candidate_id", "label")]
    stats_df = feature_df[feature_cols].agg(["min", "mean", "max"]).T
    stats_df["mean"] = stats_df["mean"].round(4)
    print(stats_df.to_string())
    print("=" * 70 + "\n")

    # Save to Parquet
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    logging.info(f"Saving feature matrix to {args.output}...")

    try:
        feature_df.to_parquet(args.output, index=False, engine="pyarrow")
        logging.info(f"Successfully saved feature matrix to {args.output} using pyarrow.")
    except Exception as e:
        logging.warning(f"Failed to write parquet ({e}). Falling back to fastparquet or CSV...")
        try:
            feature_df.to_parquet(args.output, index=False)
        except Exception:
            csv_path = args.output.replace(".parquet", ".csv")
            feature_df.to_csv(csv_path, index=False)
            logging.info(f"Saved feature matrix to CSV fallback at {csv_path}.")


if __name__ == "__main__":
    main()

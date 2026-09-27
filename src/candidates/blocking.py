"""
Blocking and Candidate Generation Module for Entity Matching.

This module implements multi-strategy blocking to generate candidate S2/S3 entity matches
for each S1 reference entity without full O(N*M) pairwise comparisons.

Strategies implemented:
1. Name token blocking with dynamic frequency-based stoplist
2. Name character n-gram MinHash LSH blocking (typo-tolerant)
3. Address token & postal/number blocking with dynamic stoplist
4. Country-aware union combination & filtering
5. Candidate recall measurement and ablation evaluation
"""

import argparse
import collections
import json
import logging
import os
import re
import string
import sys
import time
from typing import Dict, Iterable, List, Optional, Set, Tuple
import zlib
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ==============================================================================
# Text Normalization & Tokenization Helpers
# ==============================================================================

# Regex pattern for stripping common legal business suffixes across languages
LEGAL_SUFFIXES_PATTERN = re.compile(
    r"\b(inc|incorporated|llc|l\.l\.c|ltd|limited|pvt|private|corp|corporation|"
    r"co|company|gmbh|sa|plc|pty|bv|llp|lp|holdings|group|services|enterprises)\b",
    re.IGNORECASE,
)

PUNCTUATION_TRANSLATOR = str.maketrans(string.punctuation, " " * len(string.punctuation))


def normalize_text(text: Optional[str]) -> str:
    """Lowercase text and replace punctuation with spaces."""
    if text is None or pd.isna(text):
        return ""
    return str(text).lower().translate(PUNCTUATION_TRANSLATOR)


def normalize_business_name(name: Optional[str]) -> str:
    """Normalize business name by lowercasing, stripping punctuation, and removing legal suffixes."""
    cleaned = normalize_text(name)
    cleaned = LEGAL_SUFFIXES_PATTERN.sub(" ", cleaned)
    return " ".join(cleaned.split())


def extract_name_tokens(normalized_name: str, min_len: int = 2) -> List[str]:
    """Extract word tokens of at least min_len characters."""
    return [token for token in normalized_name.split() if len(token) >= min_len]


def extract_char_ngrams(text: str, n: int = 3) -> List[str]:
    """Extract character n-grams from text (ignoring spaces)."""
    compact = "".join(text.split())
    if len(compact) < n:
        return [compact] if compact else []
    return [compact[i : i + n] for i in range(len(compact) - n + 1)]


def extract_address_tokens(normalized_address: str, min_len: int = 2) -> List[str]:
    """
    Extract significant address tokens including street numbers, postal/PIN codes,
    and area/street name tokens.
    """
    tokens = []
    for token in normalized_address.split():
        if len(token) >= min_len:
            tokens.append(token)
    return tokens


# ==============================================================================
# Dynamic Frequency Stoplist Helper
# ==============================================================================

def build_token_frequency_stoplist(
    token_lists: Iterable[List[str]],
    max_doc_freq_ratio: float = 0.01,
    min_count_threshold: int = 500,
    total_docs: Optional[int] = None,
) -> Set[str]:
    """
    Derives a stoplist of overly common tokens based on empirical document frequency.
    
    This avoids hardcoded/guessed stopwords and dynamically adapts to the dataset.
    Tokens appearing in more than `max_doc_freq_ratio` of documents (or above `min_count_threshold`)
    are marked as stopwords and excluded from blocking indexes.
    """
    doc_freq: Dict[str, int] = collections.defaultdict(int)
    count = 0
    for tokens in token_lists:
        count += 1
        for unique_tok in set(tokens):
            doc_freq[unique_tok] += 1

    num_docs = total_docs if total_docs is not None else max(count, 1)
    max_count = max(int(num_docs * max_doc_freq_ratio), min_count_threshold)

    stopwords = {tok for tok, freq in doc_freq.items() if freq > max_count}
    logging.info(
        f"Derived stoplist with {len(stopwords):,} tokens (threshold: > {max_count:,} docs or > {max_doc_freq_ratio*100:.2f}%)"
    )
    return stopwords


# ==============================================================================
# Strategy 1: Name Token Blocking
# ==============================================================================

def block_by_name_tokens(
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    max_doc_freq_ratio: float = 0.005,
    min_token_len: int = 3,
    max_candidates_per_token: int = 500,
) -> Dict[str, Set[str]]:
    """
    Blocks S1 entities with S2/S3 entities sharing at least one significant normalized business name token.
    
    Tokens with frequency exceeding the dynamic stoplist threshold or bucket size limit
    are filtered out to prevent combinatorial explosion.
    
    Returns:
        dict: s1_id -> set of candidate S2/S3 entity IDs
    """
    logging.info("--- Starting Strategy 1: Name Token Blocking ---")
    start_time = time.time()

    # 1. Normalize business names and extract tokens
    s23_records: List[Tuple[str, List[str]]] = []
    all_s23_token_lists: List[List[str]] = []

    for df in (s2_df, s3_df):
        for eid, name in zip(df["entity_id"], df["business_name"]):
            norm = normalize_business_name(name)
            tokens = extract_name_tokens(norm, min_len=min_token_len)
            s23_records.append((eid, tokens))
            all_s23_token_lists.append(tokens)

    # 2. Derive dynamic stoplist from S2/S3 name tokens
    stoplist = build_token_frequency_stoplist(
        all_s23_token_lists, max_doc_freq_ratio=max_doc_freq_ratio, total_docs=len(s23_records)
    )

    # 3. Build inverted index: token -> list of S2/S3 entity IDs
    logging.info("Building inverted index for S2/S3 name tokens...")
    inverted_index: Dict[str, List[str]] = collections.defaultdict(list)
    for eid, tokens in s23_records:
        for tok in set(tokens):
            if tok not in stoplist:
                inverted_index[tok].append(eid)

    # Filter out any token buckets that grew too large
    pruned_index = {
        tok: eids
        for tok, eids in inverted_index.items()
        if len(eids) <= max_candidates_per_token
    }
    logging.info(
        f"Inverted index built with {len(pruned_index):,} active tokens (pruned {len(inverted_index) - len(pruned_index):,} oversized buckets)."
    )

    # 4. Query S1 entities against inverted index
    logging.info("Querying S1 entities against name token inverted index...")
    candidates: Dict[str, Set[str]] = collections.defaultdict(set)
    for s1_id, name in zip(s1_df["entity_id"], s1_df["business_name"]):
        norm = normalize_business_name(name)
        s1_tokens = extract_name_tokens(norm, min_len=min_token_len)
        for tok in set(s1_tokens):
            if tok in pruned_index:
                candidates[s1_id].update(pruned_index[tok])

    elapsed = time.time() - start_time
    total_pairs = sum(len(c) for c in candidates.values())
    logging.info(
        f"Strategy 1 completed in {elapsed:.2f}s: generated {total_pairs:,} candidate pairs across {len(candidates):,} S1 entities."
    )
    return candidates


# ==============================================================================
# Strategy 2: Name Character N-gram MinHash LSH Blocking
# ==============================================================================

def _hash_ngram(ngram: str, seed_a: int, seed_b: int, prime: int = 2147483647) -> int:
    """Simple 32-bit universal hash function for n-grams."""
    h = hash(ngram) & 0x7FFFFFFF
    return (seed_a * h + seed_b) % prime


def block_by_name_ngrams(
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    n: int = 3,
    num_hashes: int = 16,
    num_bands: int = 8,
    max_bucket_size: int = 500,
) -> Dict[str, Set[str]]:
    """
    Character n-gram blocking on normalized business names using MinHash LSH.
    
    This catches spelling variations, transliterations, and typos that token blocking misses,
    while scaling linearly O(N) by grouping MinHash signatures into LSH bands.
    
    Returns:
        dict: s1_id -> set of candidate S2/S3 entity IDs
    """
    logging.info(f"--- Starting Strategy 2: Name Character {n}-gram MinHash LSH Blocking ---")
    start_time = time.time()

    rows_per_band = num_hashes // num_bands
    prime = 2147483647

    # Precompute (a, b) coefficient pairs as plain Python ints once, outside the per-record loop.
    # Using numpy only to generate random integers, then converting to Python lists so that
    # all per-record computation is pure Python — avoids the overhead of numpy array allocation
    # per record at millions-scale.
    np.random.seed(42)
    a_list: List[int] = np.random.randint(1, prime - 1, size=num_hashes, dtype=np.int64).tolist()
    b_list: List[int] = np.random.randint(0, prime - 1, size=num_hashes, dtype=np.int64).tolist()

    def compute_minhash_signature(text: str) -> Optional[List[int]]:
        """Pure-Python MinHash signature computation (no per-record numpy allocation)."""
        ngrams = extract_char_ngrams(text, n=n)
        if not ngrams:
            return None
        # Hash each n-gram once with zlib.crc32 (deterministic across runs, unlike Python hash())
        ngram_hashes = [zlib.crc32(ng.encode("utf-8")) & 0x7FFFFFFF for ng in ngrams]
        # For each (a, b) hash function, find the minimum hash value over all n-grams
        return [min((a * h + b) % prime for h in ngram_hashes) for a, b in zip(a_list, b_list)]

    # 1. Build LSH Band Inverted Index from S2/S3
    logging.info("Indexing S2 and S3 character n-grams via MinHash LSH bands...")
    lsh_index: Dict[Tuple[int, Tuple[int, ...]], List[str]] = collections.defaultdict(list)
    s23_index_count = 0

    for df in (s2_df, s3_df):
        for eid, name in zip(df["entity_id"], df["business_name"]):
            norm = normalize_business_name(name)
            sig = compute_minhash_signature(norm)
            if sig is None:
                continue
            for band_idx in range(num_bands):
                start_h = band_idx * rows_per_band
                end_h = start_h + rows_per_band
                band_key = (band_idx, tuple(sig[start_h:end_h]))
                lsh_index[band_key].append(eid)
            s23_index_count += 1
            if s23_index_count % 100_000 == 0:
                logging.info(f"  [Strategy 2] Indexed {s23_index_count:,} S2/S3 records into LSH bands...")

    # Prune giant buckets to avoid explosive pairs from identical blank/short signatures
    pruned_lsh = {k: v for k, v in lsh_index.items() if len(v) <= max_bucket_size}
    logging.info(
        f"MinHash LSH index built with {len(pruned_lsh):,} active band buckets (pruned {len(lsh_index) - len(pruned_lsh):,} oversized buckets)."
    )

    # 2. Query S1 entities
    logging.info("Querying S1 entities against LSH index...")
    candidates: Dict[str, Set[str]] = collections.defaultdict(set)
    for s1_id, name in zip(s1_df["entity_id"], s1_df["business_name"]):
        norm = normalize_business_name(name)
        sig = compute_minhash_signature(norm)
        if sig is None:
            continue
        for band_idx in range(num_bands):
            start_h = band_idx * rows_per_band
            end_h = start_h + rows_per_band
            band_key = (band_idx, tuple(sig[start_h:end_h]))
            if band_key in pruned_lsh:
                candidates[s1_id].update(pruned_lsh[band_key])

    elapsed = time.time() - start_time
    total_pairs = sum(len(c) for c in candidates.values())
    logging.info(
        f"Strategy 2 completed in {elapsed:.2f}s: generated {total_pairs:,} candidate pairs across {len(candidates):,} S1 entities."
    )
    return candidates


# ==============================================================================
# Strategy 3: Address Token & Number/Postal Blocking
# ==============================================================================

def block_by_address_tokens(
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    max_doc_freq_ratio: float = 0.003,
    min_token_len: int = 3,
    max_candidates_per_token: int = 500,
) -> Dict[str, Set[str]]:
    """
    Blocks S1 entities with S2/S3 entities sharing high-signal address tokens
    (such as street numbers, postal/PIN codes, and specific area tokens).
    
    Returns:
        dict: s1_id -> set of candidate S2/S3 entity IDs
    """
    logging.info("--- Starting Strategy 3: Address Token Blocking ---")
    start_time = time.time()

    # 1. Normalize addresses and extract tokens
    s23_records: List[Tuple[str, List[str]]] = []
    all_s23_addr_tokens: List[List[str]] = []

    for df in (s2_df, s3_df):
        for eid, addr in zip(df["entity_id"], df["business_address"]):
            norm = normalize_text(addr)
            tokens = extract_address_tokens(norm, min_len=min_token_len)
            s23_records.append((eid, tokens))
            all_s23_addr_tokens.append(tokens)

    # 2. Derive dynamic address stoplist (filters out ubiquitous words like "street", "road", "ave", etc.)
    stoplist = build_token_frequency_stoplist(
        all_s23_addr_tokens, max_doc_freq_ratio=max_doc_freq_ratio, total_docs=len(s23_records)
    )

    # 3. Build inverted index for address tokens
    logging.info("Building inverted index for S2/S3 address tokens...")
    inverted_index: Dict[str, List[str]] = collections.defaultdict(list)
    for eid, tokens in s23_records:
        for tok in set(tokens):
            if tok not in stoplist:
                inverted_index[tok].append(eid)

    # Prune oversized buckets
    pruned_index = {
        tok: eids
        for tok, eids in inverted_index.items()
        if len(eids) <= max_candidates_per_token
    }
    logging.info(
        f"Address inverted index built with {len(pruned_index):,} active tokens (pruned {len(inverted_index) - len(pruned_index):,} oversized buckets)."
    )

    # 4. Query S1 entities
    logging.info("Querying S1 entities against address token inverted index...")
    candidates: Dict[str, Set[str]] = collections.defaultdict(set)
    for s1_id, addr in zip(s1_df["entity_id"], s1_df["business_address"]):
        norm = normalize_text(addr)
        s1_tokens = extract_address_tokens(norm, min_len=min_token_len)
        for tok in set(s1_tokens):
            if tok in pruned_index:
                candidates[s1_id].update(pruned_index[tok])

    elapsed = time.time() - start_time
    total_pairs = sum(len(c) for c in candidates.values())
    logging.info(
        f"Strategy 3 completed in {elapsed:.2f}s: generated {total_pairs:,} candidate pairs across {len(candidates):,} S1 entities."
    )
    return candidates


# ==============================================================================
# Strategy 4: Combine Blocks & Country-Aware Filtering
# ==============================================================================

def normalize_country(country: Optional[str]) -> str:
    """Normalize country strings without hardcoding any country names."""
    if country is None or pd.isna(country):
        return ""
    return str(country).strip().lower()


def combine_blocks(
    *block_dicts: Dict[str, Set[str]],
    s1_country_map: Optional[Dict[str, str]] = None,
    s23_country_map: Optional[Dict[str, str]] = None,
) -> Dict[str, Set[str]]:
    """
    Unions candidate sets across all blocking strategies per S1 entity, dedupes them,
    and applies country-aware filtering.
    
    Country Filtering Logic:
    - If both S1 and candidate S2/S3 have a known country value and they disagree,
      we filter out the candidate as an impossible match.
    - If EITHER the S1 entity OR the candidate has a missing/empty country, we KEEP
      the candidate. This prevents false negatives when country metadata is incomplete.
    
    Returns:
        dict: s1_id -> deduplicated and country-filtered set of candidate IDs
    """
    logging.info("--- Combining Blocking Strategies and Applying Country Filter ---")
    start_time = time.time()

    # Collect all unique s1_ids across all blocking dicts
    all_s1_ids: Set[str] = set()
    for bd in block_dicts:
        all_s1_ids.update(bd.keys())

    combined: Dict[str, Set[str]] = {}
    total_candidates_before = 0
    total_candidates_after = 0

    apply_country_filter = bool(s1_country_map and s23_country_map)

    for s1_id in all_s1_ids:
        # Union candidate sets from all strategies
        cand_union: Set[str] = set()
        for bd in block_dicts:
            if s1_id in bd:
                cand_union.update(bd[s1_id])

        total_candidates_before += len(cand_union)

        if apply_country_filter:
            s1_c = s1_country_map.get(s1_id, "")
            filtered_cands: Set[str] = set()
            for cand_id in cand_union:
                cand_c = s23_country_map.get(cand_id, "")
                # Discard only if BOTH are non-empty and DIFFERENT
                if s1_c and cand_c and s1_c != cand_c:
                    continue
                filtered_cands.add(cand_id)
            combined[s1_id] = filtered_cands
            total_candidates_after += len(filtered_cands)
        else:
            combined[s1_id] = cand_union
            total_candidates_after += len(cand_union)

    elapsed = time.time() - start_time
    logging.info(
        f"Combined {len(block_dicts)} blocking strategies in {elapsed:.2f}s:\n"
        f"  Total candidate pairs before country filter: {total_candidates_before:,}\n"
        f"  Total candidate pairs after country filter:  {total_candidates_after:,} "
        f"(filtered {total_candidates_before - total_candidates_after:,} cross-country pairs)"
    )
    return combined


# ==============================================================================
# Strategy 5: Candidate Recall & Quality Measurement
# ==============================================================================

def measure_candidate_recall(
    candidates_dict: Dict[str, Set[str]],
    ground_truth_dict: Dict[str, List[str]],
    strategy_name: str = "Combined Blocking",
) -> Dict[str, float]:
    """
    Measures candidate generation quality against true ground truth matches.
    
    Evaluates only for S1 entities present in candidates_dict (or all GT entities if evaluated).
    
    Computes:
    - Candidate Recall: Fraction of all true positive matches retained in candidate sets
    - Per-entity macro recall
    - Candidate statistics: Mean, Median, Min, Max candidates per S1
    - Total candidate pair count
    
    Returns:
        dict with computed metric values
    """
    total_true_matches = 0
    total_hits = 0
    per_entity_recalls: List[float] = []

    candidate_counts = [len(candidates_dict.get(s1_id, set())) for s1_id in ground_truth_dict]

    for s1_id, true_matches in ground_truth_dict.items():
        actual_set = set(true_matches)
        if not actual_set:
            continue  # Singletons have no positive pairs to recall in candidate generation

        num_actual = len(actual_set)
        total_true_matches += num_actual

        cand_set = candidates_dict.get(s1_id, set())
        hits = len(actual_set.intersection(cand_set))
        total_hits += hits

        per_entity_recalls.append(hits / num_actual)

    overall_recall = (total_hits / total_true_matches) if total_true_matches > 0 else 0.0
    macro_recall = float(np.mean(per_entity_recalls)) if per_entity_recalls else 0.0

    counts_arr = np.array(candidate_counts) if candidate_counts else np.array([0])
    total_pairs = int(np.sum(counts_arr))
    avg_cands = float(np.mean(counts_arr))
    median_cands = float(np.median(counts_arr))
    max_cands = int(np.max(counts_arr))

    print("\n" + "=" * 65)
    print(f" BLOCKING QUALITY REPORT: {strategy_name}")
    print("=" * 65)
    print(f" Evaluated S1 Entities (in GT):   {len(ground_truth_dict):,}")
    print(f" Total True Match Pairs in GT:    {total_true_matches:,}")
    print(f" True Match Pairs Captured (Hits):{total_hits:,}")
    print(f" Candidate Recall (Micro):        {overall_recall * 100:.2f}%")
    print(f" Candidate Recall (Macro/Entity): {macro_recall * 100:.2f}%")
    print("-" * 65)
    print(f" Total Candidate Pairs Generated: {total_pairs:,}")
    print(f" Avg Candidates per S1:           {avg_cands:.2f}")
    print(f" Median Candidates per S1:        {median_cands:.1f}")
    print(f" Max Candidates for any S1:       {max_cands:,}")
    print("=" * 65 + "\n")

    return {
        "candidate_recall": overall_recall,
        "macro_candidate_recall": macro_recall,
        "total_true_matches": total_true_matches,
        "total_hits": total_hits,
        "total_candidate_pairs": total_pairs,
        "avg_candidates_per_s1": avg_cands,
        "median_candidates_per_s1": median_cands,
        "max_candidates_per_s1": max_cands,
    }


# ==============================================================================
# Helper to Load Ground Truth TSV
# ==============================================================================

def load_ground_truth(tsv_path: str) -> Dict[str, List[str]]:
    """Loads ground truth TSV mapping source1_entity_id -> list of matched_entity_ids."""
    df = pd.read_csv(tsv_path, sep="\t", dtype=str)
    gt: Dict[str, List[str]] = {}
    for _, row in df.iterrows():
        s1_id = row["source1_entity_id"]
        matches_str = row.get("matched_entity_ids", "")
        if pd.isna(matches_str) or not str(matches_str).strip() or str(matches_str).lower() == "nan":
            gt[s1_id] = []
        else:
            gt[s1_id] = [m.strip() for m in str(matches_str).split(",") if m.strip()]
    return gt


# ==============================================================================
# Main / CLI Execution
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Multi-strategy candidate blocking and quality measurement for entity matching."
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
        default="output/train_candidates.json",
        help="Path to output JSON file for candidates (default: output/train_candidates.json)",
    )
    parser.add_argument(
        "--sample-s1",
        type=int,
        default=None,
        help="Optional: sample first N S1 entities for rapid iteration and testing",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="If set, apply inline top-K pruning per S1 entity after combining strategies. "
             "Avoids materializing full unpruned candidates in memory. Recommended for full-scale runs.",
    )

    args = parser.parse_args()

    logging.info(f"Loading source datasets...\n  S1: {args.s1}\n  S2: {args.s2}\n  S3: {args.s3}")
    s1_df = pd.read_csv(args.s1, sep="\t", dtype=str)
    s2_df = pd.read_csv(args.s2, sep="\t", dtype=str)
    s3_df = pd.read_csv(args.s3, sep="\t", dtype=str)

    if args.sample_s1 is not None and args.sample_s1 > 0:
        logging.info(f"Subsampling first {args.sample_s1:,} S1 entities for evaluation...")
        s1_df = s1_df.iloc[: args.sample_s1]

    # Build country maps
    logging.info("Building country lookup maps for filtering...")
    s1_country_map = {
        eid: normalize_country(c) for eid, c in zip(s1_df["entity_id"], s1_df["country"])
    }
    s23_country_map = {}
    for df in (s2_df, s3_df):
        for eid, c in zip(df["entity_id"], df["country"]):
            s23_country_map[eid] = normalize_country(c)

    # -------------------------------------------------------------------------
    # FULL-SCALE PATH: Fused streaming blocking + per-entity top-K pruning
    # When --top-k is set, we never materialize unpruned candidate pairs in memory.
    # We build the inverted indexes for S2/S3, then stream through S1 entities,
    # lookup candidates from all strategies, score with cheap_similarity_score,
    # and keep only the top-K matches per entity.
    # -------------------------------------------------------------------------
    if args.top_k is not None:
        logging.info(
            f"Full-scale fused mode: building inverted indexes & pruning to top-K={args.top_k} per S1 entity..."
        )
        try:
            from candidates.prune import EntityProfile as PruneProfile
            from candidates.prune import _score_profiles
        except ImportError:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from prune import EntityProfile as PruneProfile, _score_profiles

        # Single pass over S2 and S3 to build all 3 inverted indexes & profiles simultaneously
        logging.info("Indexing S2 and S3 (Name Tokens, MinHash LSH, Address Tokens) in a single pass...")

        prime = 2147483647
        num_hashes, num_bands = 20, 5
        rows_per_band = 4
        np.random.seed(42)
        a_list: List[int] = np.random.randint(1, prime - 1, size=num_hashes, dtype=np.int64).tolist()
        b_list: List[int] = np.random.randint(0, prime - 1, size=num_hashes, dtype=np.int64).tolist()

        def _get_sig(text: str) -> Optional[List[int]]:
            ngrams = extract_char_ngrams(text, n=3)
            if not ngrams:
                return None
            h_ngs = [zlib.crc32(ng.encode("utf-8")) & 0x7FFFFFFF for ng in ngrams]
            return [min((a * h + b) % prime for h in h_ngs) for a, b in zip(a_list, b_list)]

        idx_name_tok: Dict[str, List[str]] = collections.defaultdict(list)
        idx_minhash: Dict[Tuple[int, Tuple[int, ...]], List[str]] = collections.defaultdict(list)
        idx_addr_tok: Dict[str, List[str]] = collections.defaultdict(list)
        cand_profiles: Dict[str, PruneProfile] = {}

        name_tok_freq: Dict[str, int] = collections.defaultdict(int)
        addr_tok_freq: Dict[str, int] = collections.defaultdict(int)

        count_s23 = 0
        t0_s23 = time.time()

        for df in (s2_df, s3_df):
            for eid, name, addr in zip(df["entity_id"], df["business_name"], df["business_address"]):
                cand_profiles[eid] = PruneProfile(name, addr)

                norm_n = normalize_business_name(name)
                norm_a = normalize_text(addr)

                # Name tokens
                n_toks = set(extract_name_tokens(norm_n, min_len=3))
                for tok in n_toks:
                    idx_name_tok[tok].append(eid)
                    name_tok_freq[tok] += 1

                # Address tokens
                a_toks = set(extract_address_tokens(norm_a, min_len=2))
                for tok in a_toks:
                    idx_addr_tok[tok].append(eid)
                    addr_tok_freq[tok] += 1

                # MinHash LSH
                sig = _get_sig(norm_n)
                if sig is not None:
                    for b_idx in range(num_bands):
                        b_key = (b_idx, tuple(sig[b_idx * rows_per_band : (b_idx + 1) * rows_per_band]))
                        idx_minhash[b_key].append(eid)

                count_s23 += 1
                if count_s23 % 500_000 == 0:
                    logging.info(f"  Indexed {count_s23:,} S2/S3 entities in {time.time()-t0_s23:.2f}s...")

        logging.info(f"S2/S3 single-pass indexing complete ({count_s23:,} entities in {time.time()-t0_s23:.2f}s). Pruning buckets...")

        max_doc_freq_name = max(int(count_s23 * 0.005), 100)
        stoplist_name = {tok for tok, freq in name_tok_freq.items() if freq > max_doc_freq_name}
        idx_name_tok_pruned = {
            tok: eids for tok, eids in idx_name_tok.items()
            if tok not in stoplist_name and len(eids) <= 500
        }

        max_doc_freq_addr = max(int(count_s23 * 0.01), 100)
        stoplist_addr = {tok for tok, freq in addr_tok_freq.items() if freq > max_doc_freq_addr}
        idx_addr_tok_pruned = {
            tok: eids for tok, eids in idx_addr_tok.items()
            if tok not in stoplist_addr and len(eids) <= 500
        }

        idx_minhash_pruned = {k: v for k, v in idx_minhash.items() if len(v) <= 2000}
        logging.info(
            f"Active buckets ready: Name Tokens={len(idx_name_tok_pruned):,}, "
            f"MinHash LSH={len(idx_minhash_pruned):,}, Address Tokens={len(idx_addr_tok_pruned):,}."
        )

        # 4. Stream S1 Entities, Union & Score & Keep Top-K
        logging.info(f"Streaming {len(s1_df):,} S1 entities through multi-strategy blocking & top-{args.top_k} pruning...")
        fused_candidates: Dict[str, List[str]] = {}
        processed = 0
        start_fuse = time.time()

        for s1_id, name, addr, s1_country in zip(
            s1_df["entity_id"], s1_df["business_name"], s1_df["business_address"], s1_df["country"]
        ):
            s1_c_norm = s1_country_map.get(s1_id, "")
            s1_prof = PruneProfile(name, addr)

            cand_union: Set[str] = set()

            # S1 Name Tokens lookup
            s1_norm_name = normalize_business_name(name)
            s1_name_toks = extract_name_tokens(s1_norm_name, min_len=3)
            for tok in set(s1_name_toks):
                if tok in idx_name_tok_pruned:
                    cand_union.update(idx_name_tok_pruned[tok])

            # S1 MinHash LSH lookup
            sig_s1 = _get_sig(s1_norm_name)
            if sig_s1 is not None:
                for b_idx in range(num_bands):
                    b_key = (b_idx, tuple(sig_s1[b_idx * rows_per_band : (b_idx + 1) * rows_per_band]))
                    if b_key in idx_minhash_pruned:
                        cand_union.update(idx_minhash_pruned[b_key])

            # S1 Address Tokens lookup
            s1_norm_addr = normalize_text(addr)
            s1_addr_toks = extract_address_tokens(s1_norm_addr, min_len=2)
            for tok in set(s1_addr_toks):
                if tok in idx_addr_tok_pruned:
                    cand_union.update(idx_addr_tok_pruned[tok])

            if not cand_union:
                fused_candidates[s1_id] = []
                continue

            # Country Filter & Cheap Scoring
            scored: List[Tuple[str, float]] = []
            for cand_id in cand_union:
                cand_c_norm = s23_country_map.get(cand_id, "")
                if s1_c_norm and cand_c_norm and s1_c_norm != cand_c_norm:
                    continue
                c_prof = cand_profiles.get(cand_id)
                if c_prof is None:
                    continue
                scored.append((cand_id, _score_profiles(s1_prof, c_prof)))

            if not scored:
                fused_candidates[s1_id] = []
                continue

            scored.sort(key=lambda x: x[1], reverse=True)
            fused_candidates[s1_id] = [cid for cid, _ in scored[: args.top_k]]

            processed += 1
            if processed % 100_000 == 0:
                elapsed_fuse = time.time() - start_fuse
                rate = processed / max(elapsed_fuse, 0.001)
                logging.info(
                    f"  [Fused Stream] Processed {processed:,}/{len(s1_df):,} S1 entities "
                    f"({processed / len(s1_df) * 100:.1f}%) [{rate:,.0f} S1/sec]"
                )

        combined_candidates = fused_candidates
        logging.info(
            f"Fused combine+prune complete. {len(fused_candidates):,} S1 entities, "
            f"{sum(len(v) for v in fused_candidates.values()):,} total candidate pairs."
        )

    else:
        # SAMPLE/ANALYSIS PATH: Materialize full unpruned candidates (use only with --sample-s1)
        b1_name_tokens = block_by_name_tokens(s1_df, s2_df, s3_df)
        b2_name_ngrams = block_by_name_ngrams(s1_df, s2_df, s3_df, n=3)
        b3_addr_tokens = block_by_address_tokens(s1_df, s2_df, s3_df)
        combined_candidates = combine_blocks(
            b1_name_tokens,
            b2_name_ngrams,
            b3_addr_tokens,
            s1_country_map=s1_country_map,
            s23_country_map=s23_country_map,
        )

    # Measure quality against ground truth (only meaningful when --sample-s1 is used,
    # since loading full GT for 2M+ entities takes several minutes)
    if os.path.exists(args.ground_truth) and args.sample_s1 is not None:
        logging.info(f"Loading ground truth from {args.ground_truth}...")
        full_gt = load_ground_truth(args.ground_truth)
        eval_s1_set = set(s1_df["entity_id"])
        eval_gt = {k: v for k, v in full_gt.items() if k in eval_s1_set}

        combined_metrics = measure_candidate_recall(
            combined_candidates, eval_gt, strategy_name="Combined All Strategies"
        )

        # Ablation only runs in sample mode (requires the individual block dicts to still be in memory)
        if args.top_k is None:
            print("\n" + "=" * 65)
            print(" ABLATION & STRATEGY CONTRIBUTION BREAKDOWN")
            print("=" * 65)
            print(f"{'Strategy Evaluated':<32} | {'Recall':<8} | {'Recall Drop if Removed':<22}")
            print("-" * 65)

            base_recall = combined_metrics["candidate_recall"]
            strategies = [
                ("Strategy 1 (Name Tokens)", b1_name_tokens, [b2_name_ngrams, b3_addr_tokens]),
                ("Strategy 2 (Name 3-grams)", b2_name_ngrams, [b1_name_tokens, b3_addr_tokens]),
                ("Strategy 3 (Address Tokens)", b3_addr_tokens, [b1_name_tokens, b2_name_ngrams]),
            ]
            for name, single_dict, others in strategies:
                single_comb = combine_blocks(
                    single_dict, s1_country_map=s1_country_map, s23_country_map=s23_country_map
                )
                single_m = measure_candidate_recall(single_comb, eval_gt, strategy_name=f"Standalone: {name}")
                without_this = combine_blocks(
                    *others, s1_country_map=s1_country_map, s23_country_map=s23_country_map
                )
                ablation_m = measure_candidate_recall(
                    without_this, eval_gt, strategy_name=f"Ablation (Without {name})"
                )
                drop = base_recall - ablation_m["candidate_recall"]
                print(f"{name:<32} | {single_m['candidate_recall']*100:>6.2f}% | {drop*100:>+6.2f}% lost if removed")
            print("=" * 65 + "\n")
    elif not os.path.exists(args.ground_truth):
        logging.warning(f"Ground truth file not found at {args.ground_truth}. Skipping recall measurement.")

    # Save candidates to output JSON
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    logging.info(f"Saving combined candidates to {args.output}...")
    json_candidates = {
        s1_id: sorted(list(cands)) if isinstance(cands, set) else cands
        for s1_id, cands in combined_candidates.items()
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(json_candidates, f)
    logging.info(f"Successfully saved candidates for {len(json_candidates):,} S1 entities to {args.output}.")


if __name__ == "__main__":
    main()

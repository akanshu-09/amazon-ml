"""
Candidate Pre-Scoring and Top-K Pruning Module.

This module provides a fast, lightweight pre-scoring and pruning step between
candidate blocking and expensive downstream feature engineering (e.g. TF-IDF,
Levenshtein, Transformer embeddings).

By scoring candidate pairs using quick set-based Jaccard similarities (name tokens,
name character 3-grams, address tokens), we can reduce candidate volume from ~500+
per entity down to a compact top-K (e.g. 20-30) while preserving >95% of blocked recall.

Note: Top-K is a precision-vs-compute tradeoff, not a hard correctness rule.
The optimal top-K should be selected empirically based on the tradeoff table.
"""

import argparse
import collections
import json
import logging
import os
import sys
import time
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

# Add parent directory to sys.path to enable direct module execution
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from candidates.blocking import (
        extract_address_tokens,
        extract_char_ngrams,
        extract_name_tokens,
        load_ground_truth,
        normalize_business_name,
        normalize_text,
    )
except ImportError:
    # Direct import fallback if executed from within src/candidates/
    from blocking import (
        extract_address_tokens,
        extract_char_ngrams,
        extract_name_tokens,
        load_ground_truth,
        normalize_business_name,
        normalize_text,
    )

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ==============================================================================
# Tunable Weights for Cheap Similarity Scoring
# ==============================================================================
# Reasoning for default weights:
# - WEIGHT_NAME_TOKENS (0.45): Primary signal. Exact token matches indicate identical/core business names.
# - WEIGHT_NAME_NGRAMS (0.35): Essential for catching typos, minor spelling differences, and transliteration noise.
# - WEIGHT_ADDR_TOKENS (0.20): Differentiates identically/similarly named businesses at different locations,
#   or supports matches when business names have severe noise.
WEIGHT_NAME_TOKENS = 0.45
WEIGHT_NAME_NGRAMS = 0.35
WEIGHT_ADDR_TOKENS = 0.20


def jaccard_similarity(set_a: Set[str], set_b: Set[str]) -> float:
    """Computes Jaccard similarity |A ∩ B| / |A ∪ B| between two sets."""
    if not set_a and not set_b:
        return 0.0
    intersection_len = len(set_a.intersection(set_b))
    if intersection_len == 0:
        return 0.0
    union_len = len(set_a.union(set_b))
    return intersection_len / union_len if union_len > 0 else 0.0


def cheap_similarity_score(
    s1_name: str,
    s1_addr: str,
    cand_name: str,
    cand_addr: str,
    w_name_tokens: float = WEIGHT_NAME_TOKENS,
    w_name_ngrams: float = WEIGHT_NAME_NGRAMS,
    w_addr_tokens: float = WEIGHT_ADDR_TOKENS,
) -> float:
    """
    Computes a fast similarity score between S1 and candidate entity without
    expensive string algorithms (like full Levenshtein or TF-IDF models).
    
    Combines:
    1. Name token Jaccard similarity (WEIGHT_NAME_TOKENS = 0.45)
    2. Name character 3-gram Jaccard similarity (WEIGHT_NAME_NGRAMS = 0.35)
    3. Address token Jaccard similarity (WEIGHT_ADDR_TOKENS = 0.20)
    
    Returns:
        float: weighted similarity score in range [0.0, 1.0]
    """
    # 1. Name tokens
    s1_norm_name = normalize_business_name(s1_name)
    cand_norm_name = normalize_business_name(cand_name)
    s1_name_toks = set(extract_name_tokens(s1_norm_name, min_len=2))
    cand_name_toks = set(extract_name_tokens(cand_norm_name, min_len=2))
    sim_name_tok = jaccard_similarity(s1_name_toks, cand_name_toks)

    # 2. Name character 3-grams
    s1_name_ngrams = set(extract_char_ngrams(s1_norm_name, n=3))
    cand_name_ngrams = set(extract_char_ngrams(cand_norm_name, n=3))
    sim_name_ngram = jaccard_similarity(s1_name_ngrams, cand_name_ngrams)

    # 3. Address tokens
    s1_norm_addr = normalize_text(s1_addr)
    cand_norm_addr = normalize_text(cand_addr)
    s1_addr_toks = set(extract_address_tokens(s1_norm_addr, min_len=2))
    cand_addr_toks = set(extract_address_tokens(cand_norm_addr, min_len=2))
    sim_addr_tok = jaccard_similarity(s1_addr_toks, cand_addr_toks)

    score = (
        (w_name_tokens * sim_name_tok)
        + (w_name_ngrams * sim_name_ngram)
        + (w_addr_tokens * sim_addr_tok)
    )
    return float(score)


# ==============================================================================
# High-Throughput Precomputed Entity Profile
# ==============================================================================

class EntityProfile:
    """Pre-tokenized and indexed entity profile for maximum scoring speed."""
    __slots__ = ("name_tokens", "name_ngrams", "addr_tokens")

    def __init__(self, name: str, address: str):
        norm_name = normalize_business_name(name)
        norm_addr = normalize_text(address)
        self.name_tokens: Set[str] = set(extract_name_tokens(norm_name, min_len=2))
        self.name_ngrams: Set[str] = set(extract_char_ngrams(norm_name, n=3))
        self.addr_tokens: Set[str] = set(extract_address_tokens(norm_addr, min_len=2))


def _score_profiles(
    p1: EntityProfile,
    p2: EntityProfile,
    w_name_tokens: float = WEIGHT_NAME_TOKENS,
    w_name_ngrams: float = WEIGHT_NAME_NGRAMS,
    w_addr_tokens: float = WEIGHT_ADDR_TOKENS,
) -> float:
    """Blazing fast set-based similarity between two pre-tokenized profiles."""
    sim_name_tok = jaccard_similarity(p1.name_tokens, p2.name_tokens)
    sim_name_ngram = jaccard_similarity(p1.name_ngrams, p2.name_ngrams)
    sim_addr_tok = jaccard_similarity(p1.addr_tokens, p2.addr_tokens)

    return (
        (w_name_tokens * sim_name_tok)
        + (w_name_ngrams * sim_name_ngram)
        + (w_addr_tokens * sim_addr_tok)
    )


# ==============================================================================
# Top-K Pruning Function
# ==============================================================================

def prune_candidates(
    candidates_dict: Dict[str, Iterable[str]],
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    top_k: int = 30,
    min_score_floor: float = 0.0,
    w_name_tokens: float = WEIGHT_NAME_TOKENS,
    w_name_ngrams: float = WEIGHT_NAME_NGRAMS,
    w_addr_tokens: float = WEIGHT_ADDR_TOKENS,
) -> Dict[str, List[str]]:
    """
    For each S1 entity, scores all its candidate matches using cheap_similarity_score,
    sorts descending, and retains only the top_k highest-scoring candidates above min_score_floor.
    
    Optimized for memory and speed: builds pre-tokenized lookup profiles for referenced
    entities once to avoid dataframe lookups and repeated string normalization.
    
    Args:
        candidates_dict: dict mapping s1_id -> collection of candidate S2/S3 entity IDs
        s1_df: DataFrame of S1 entities (columns: entity_id, business_name, business_address)
        s2_df: DataFrame of S2 entities
        s3_df: DataFrame of S3 entities
        top_k: Maximum number of candidates to keep per S1 entity
        min_score_floor: Minimum similarity threshold (candidates below this are discarded)
        
    Returns:
        dict: s1_id -> list of top-K pruned candidate IDs (sorted by score descending)
    """
    logging.info(f"--- Pruning Candidates (top_k={top_k}, min_score_floor={min_score_floor}) ---")
    start_time = time.time()

    # Determine which S1 and candidate IDs are actually referenced to save memory
    referenced_s1_ids = set(candidates_dict.keys())
    referenced_cand_ids: Set[str] = set()
    for cands in candidates_dict.values():
        referenced_cand_ids.update(cands)

    logging.info(
        f"Building profiles for {len(referenced_s1_ids):,} S1 entities and {len(referenced_cand_ids):,} candidate entities..."
    )

    # 1. Build profile lookup for S1
    s1_profiles: Dict[str, EntityProfile] = {}
    for eid, name, addr in zip(s1_df["entity_id"], s1_df["business_name"], s1_df["business_address"]):
        if eid in referenced_s1_ids:
            s1_profiles[eid] = EntityProfile(name, addr)

    # 2. Build profile lookup for S2 and S3
    cand_profiles: Dict[str, EntityProfile] = {}
    for df in (s2_df, s3_df):
        for eid, name, addr in zip(df["entity_id"], df["business_name"], df["business_address"]):
            if eid in referenced_cand_ids:
                cand_profiles[eid] = EntityProfile(name, addr)

    logging.info("Scoring and pruning candidate sets...")
    pruned_dict: Dict[str, List[str]] = {}
    total_in = 0
    total_out = 0

    for s1_id, cands in candidates_dict.items():
        s1_prof = s1_profiles.get(s1_id)
        if not s1_prof:
            pruned_dict[s1_id] = []
            continue

        cand_list = list(cands)
        total_in += len(cand_list)

        if not cand_list:
            pruned_dict[s1_id] = []
            continue

        # Score all candidates for this S1 entity
        scored_cands: List[Tuple[str, float]] = []
        for cand_id in cand_list:
            c_prof = cand_profiles.get(cand_id)
            if c_prof is None:
                continue
            score = _score_profiles(
                s1_prof,
                c_prof,
                w_name_tokens=w_name_tokens,
                w_name_ngrams=w_name_ngrams,
                w_addr_tokens=w_addr_tokens,
            )
            if score >= min_score_floor:
                scored_cands.append((cand_id, score))

        # Sort descending by score
        scored_cands.sort(key=lambda x: x[1], reverse=True)

        # Retain top-K
        kept_cands = [cand_id for cand_id, _ in scored_cands[:top_k]]
        pruned_dict[s1_id] = kept_cands
        total_out += len(kept_cands)

    elapsed = time.time() - start_time
    reduction_pct = ((total_in - total_out) / total_in * 100) if total_in > 0 else 0.0
    logging.info(
        f"Pruning completed in {elapsed:.2f}s:\n"
        f"  Total pairs before: {total_in:,} (avg {total_in / max(len(candidates_dict), 1):.1f}/S1)\n"
        f"  Total pairs after:  {total_out:,} (avg {total_out / max(len(candidates_dict), 1):.1f}/S1)\n"
        f"  Volume reduction:   {reduction_pct:.2f}%"
    )

    return pruned_dict


# ==============================================================================
# Pruning Quality & Impact Measurement
# ==============================================================================

def measure_pruning_impact(
    original_candidates: Dict[str, Iterable[str]],
    pruned_candidates: Dict[str, Iterable[str]],
    ground_truth_dict: Dict[str, List[str]],
) -> Dict[str, float]:
    """
    Compares candidate recall and volume BEFORE vs AFTER pruning.
    
    Returns:
        dict with before/after recall, recall loss, counts, and reduction ratio.
    """
    def _eval_recall(cands_map: Dict[str, Iterable[str]]) -> Tuple[float, int, int]:
        total_true = 0
        total_hits = 0
        for s1_id, true_matches in ground_truth_dict.items():
            actual_set = set(true_matches)
            if not actual_set:
                continue
            total_true += len(actual_set)
            cand_set = set(cands_map.get(s1_id, []))
            total_hits += len(actual_set.intersection(cand_set))
        recall = (total_hits / total_true) if total_true > 0 else 0.0
        return recall, total_hits, total_true

    recall_before, hits_before, total_gt = _eval_recall(original_candidates)
    recall_after, hits_after, _ = _eval_recall(pruned_candidates)

    pairs_before = sum(len(list(c)) for c in original_candidates.values())
    pairs_after = sum(len(list(c)) for c in pruned_candidates.values())
    num_s1 = max(len(original_candidates), 1)

    avg_cands_before = pairs_before / num_s1
    avg_cands_after = pairs_after / num_s1
    recall_loss = recall_before - recall_after
    recall_retention = (recall_after / recall_before * 100) if recall_before > 0 else 0.0
    reduction_ratio = (pairs_before / pairs_after) if pairs_after > 0 else float("inf")
    volume_reduction_pct = ((pairs_before - pairs_after) / pairs_before * 100) if pairs_before > 0 else 0.0

    return {
        "recall_before": recall_before,
        "recall_after": recall_after,
        "recall_loss": recall_loss,
        "recall_retention_pct": recall_retention,
        "hits_before": hits_before,
        "hits_after": hits_after,
        "total_ground_truth_pairs": total_gt,
        "pairs_before": pairs_before,
        "pairs_after": pairs_after,
        "avg_cands_before": avg_cands_before,
        "avg_cands_after": avg_cands_after,
        "reduction_ratio": reduction_ratio,
        "volume_reduction_pct": volume_reduction_pct,
    }


# ==============================================================================
# Main / CLI Execution & Tradeoff Sweep
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Fast pre-scoring and Top-K candidate pruning for entity matching."
    )
    parser.add_argument(
        "--candidates",
        type=str,
        default="output/train_candidates.json",
        help="Path to unpruned candidates JSON file (default: output/train_candidates.json)",
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
        "--top-k",
        type=int,
        default=30,
        help="Top-K candidates to save in output file (default: 30)",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="Minimum similarity score floor (default: 0.0)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/train_candidates_pruned.json",
        help="Path to save pruned candidates JSON (default: output/train_candidates_pruned.json)",
    )

    args = parser.parse_args()

    # Check candidates file
    if not os.path.exists(args.candidates):
        # Check if sample candidate file exists as convenient fallback
        fallback = "output/sample_train_candidates.json"
        if os.path.exists(fallback):
            logging.warning(f"Candidates file '{args.candidates}' not found. Falling back to '{fallback}'.")
            args.candidates = fallback
        else:
            logging.error(f"Candidates file not found at '{args.candidates}'. Please run blocking.py first.")
            sys.exit(1)

    logging.info(f"Loading candidate pairs from {args.candidates}...")
    with open(args.candidates, "r", encoding="utf-8") as f:
        candidates_dict = json.load(f)

    logging.info(f"Loaded candidates for {len(candidates_dict):,} S1 entities.")

    logging.info("Loading source datasets...")
    s1_df = pd.read_csv(args.s1, sep="\t", dtype=str)
    s2_df = pd.read_csv(args.s2, sep="\t", dtype=str)
    s3_df = pd.read_csv(args.s3, sep="\t", dtype=str)

    ground_truth_dict = {}
    if os.path.exists(args.ground_truth):
        logging.info(f"Loading ground truth from {args.ground_truth}...")
        full_gt = load_ground_truth(args.ground_truth)
        # Filter GT to only S1 entities in candidates
        ground_truth_dict = {k: v for k, v in full_gt.items() if k in candidates_dict}

    # Evaluate Tradeoff Sweep across multiple Top-K values (e.g. 10, 20, 30, 50)
    top_k_candidates_sweep = [10, 20, 30, 50]
    sweep_results = []

    print("\n" + "=" * 80)
    print(" TOP-K CANDIDATE PRUNING TRADEOFF ANALYSIS")
    print("=" * 80)
    print(
        f"{'Top-K':<8} | {'Recall':<9} | {'Recall Loss':<13} | {'Avg Cands/S1':<14} | {'Total Pairs':<12} | {'Reduction':<10}"
    )
    print("-" * 80)

    for k in top_k_candidates_sweep:
        pruned_k = prune_candidates(
            candidates_dict,
            s1_df,
            s2_df,
            s3_df,
            top_k=k,
            min_score_floor=args.min_score,
        )
        impact = measure_pruning_impact(candidates_dict, pruned_k, ground_truth_dict)
        sweep_results.append((k, impact, pruned_k))

        print(
            f"{k:<8} | {impact['recall_after']*100:>6.2f}%  | "
            f"-{impact['recall_loss']*100:>5.2f}% ({impact['recall_retention_pct']:>5.1f}%) | "
            f"{impact['avg_cands_after']:>8.1f}      | "
            f"{impact['pairs_after']:>10,}   | "
            f"{impact['volume_reduction_pct']:>6.1f}%"
        )

    print("=" * 80)
    print("Note: Choose Top-K based on available compute budget vs recall retention.")
    print("=" * 80 + "\n")

    # Generate and save final pruned candidates at chosen args.top_k
    logging.info(f"Generating final output at chosen Top-K = {args.top_k}...")
    final_pruned = prune_candidates(
        candidates_dict,
        s1_df,
        s2_df,
        s3_df,
        top_k=args.top_k,
        min_score_floor=args.min_score,
    )

    if ground_truth_dict:
        final_impact = measure_pruning_impact(candidates_dict, final_pruned, ground_truth_dict)
        print("\n" + "=" * 65)
        print(f" FINAL PRUNING IMPACT REPORT (Top-K = {args.top_k})")
        print("=" * 65)
        print(f" Total S1 Entities:              {len(final_pruned):,}")
        print(f" Candidate Recall BEFORE:        {final_impact['recall_before']*100:.2f}%")
        print(f" Candidate Recall AFTER:         {final_impact['recall_after']*100:.2f}%")
        print(f" Recall Retained:                {final_impact['recall_retention_pct']:.2f}%")
        print(f" Recall Lost:                    -{final_impact['recall_loss']*100:.2f}%")
        print("-" * 65)
        print(f" Candidate Pairs BEFORE:         {final_impact['pairs_before']:,} (Avg {final_impact['avg_cands_before']:.1f}/S1)")
        print(f" Candidate Pairs AFTER:          {final_impact['pairs_after']:,} (Avg {final_impact['avg_cands_after']:.1f}/S1)")
        print(f" Volume Reduction:               {final_impact['volume_reduction_pct']:.2f}% ({final_impact['reduction_ratio']:.1f}x speedup for downstream features)")
        print("=" * 65 + "\n")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    logging.info(f"Saving pruned candidates to {args.output}...")
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(final_pruned, f)
    logging.info(f"Successfully saved {len(final_pruned):,} pruned candidate lists to {args.output}.")


if __name__ == "__main__":
    main()

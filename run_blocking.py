"""
Standalone full-scale blocking + pruning script.
Writes progress incrementally to a JSONL file so nothing is lost if interrupted.
Run directly: python run_blocking.py
"""
from ast import Set
from ast import Dict
import argparse
import collections
import json
import logging
import os
import re
import string
import sys
import time
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

LEGAL_SUFFIXES = re.compile(
    r"\b(inc|incorporated|llc|ltd|limited|pvt|private|corp|corporation|"
    r"co|company|gmbh|sa|plc|pty|bv|llp|lp|holdings|group|services|enterprises)\b",
    re.IGNORECASE,
)
PUNCT = str.maketrans(string.punctuation, " " * len(string.punctuation))


def norm_text(t):
    if t is None or pd.isna(t):
        return ""
    return str(t).lower().translate(PUNCT)


def norm_name(t):
    c = norm_text(t)
    c = LEGAL_SUFFIXES.sub(" ", c)
    return " ".join(c.split())


def name_tokens(norm_n, min_len=3):
    return [w for w in norm_n.split() if len(w) >= min_len]


def addr_tokens(norm_a, min_len=3):
    return [w for w in norm_a.split() if len(w) >= min_len]


def char_trigrams(norm_n):
    compact = "".join(norm_n.split())
    if len(compact) < 3:
        return [compact] if compact else []
    return [compact[i:i + 3] for i in range(len(compact) - 2)]


def norm_country(c):
    if c is None or pd.isna(c):
        return ""
    return str(c).strip().lower()


def jaccard(a, b):
    if not a and not b:
        return 0.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def build_stoplist(token_lists, max_doc_freq_ratio=0.005, min_count=500):
    doc_freq = collections.defaultdict(int)
    n = 0
    for toks in token_lists:
        n += 1
        for t in set(toks):
            doc_freq[t] += 1
    max_count = max(int(n * max_doc_freq_ratio), min_count)
    return {t for t, f in doc_freq.items() if f > max_count}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s1", default="dataset/train/train_source1.tsv")
    ap.add_argument("--s2", default="dataset/train/train_source2.tsv")
    ap.add_argument("--s3", default="dataset/train/train_source3.tsv")
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--max-bucket", type=int, default=1000)
    ap.add_argument("--batch-log-every", type=int, default=25000)
    ap.add_argument("--output-jsonl", default="output/candidates_progress.jsonl")
    ap.add_argument("--output-json", default="output/train_candidates_pruned_full.json")
    ap.add_argument("--sample-s1", type=int, default=None)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.output_jsonl), exist_ok=True)

    logging.info("Loading S1/S2/S3 TSVs...")
    s1_df = pd.read_csv(args.s1, sep="\t", dtype=str)
    if args.sample_s1:
        s1_df = s1_df.iloc[:args.sample_s1]
        logging.info(f"Sampling first {args.sample_s1:,} S1 entities.")
    s2_df = pd.read_csv(args.s2, sep="\t", dtype=str)
    s3_df = pd.read_csv(args.s3, sep="\t", dtype=str)
    logging.info(f"S1: {len(s1_df):,} | S2: {len(s2_df):,} | S3: {len(s3_df):,}")

    # ---- Build lookup dicts for S2/S3 (id -> name, addr, country) ----
    cand_name = {}
    cand_addr = {}
    cand_country = {}
    all_name_tok_lists = []
    all_addr_tok_lists = []

    logging.info("Building candidate lookup dicts and token lists...")
    for df in (s2_df, s3_df):
        for eid, name, addr, country in zip(
            df["entity_id"], df["business_name"], df["business_address"], df["country"]
        ):
            nn = norm_name(name)
            na = norm_text(addr)
            cand_name[eid] = nn
            cand_addr[eid] = na
            cand_country[eid] = norm_country(country)
            all_name_tok_lists.append(name_tokens(nn))
            all_addr_tok_lists.append(addr_tokens(na))

    total_cands = len(cand_name)
    logging.info(f"Loaded {total_cands:,} candidate profiles.")

    # ---- Derive dynamic stoplists ----
    logging.info("Deriving stoplists...")
    name_stop = build_stoplist(all_name_tok_lists, max_doc_freq_ratio=0.005)
    addr_stop = build_stoplist(all_addr_tok_lists, max_doc_freq_ratio=0.003)
    logging.info(f"Name stoplist: {len(name_stop)} tokens | Addr stoplist: {len(addr_stop)} tokens")
    del all_name_tok_lists, all_addr_tok_lists  # free memory

    # ---- Build three inverted indices in one pass ----
    logging.info("Building inverted indices (name tokens, trigrams, addr tokens)...")
    name_tok_idx = collections.defaultdict(list)
    trigram_idx = collections.defaultdict(list)
    addr_tok_idx = collections.defaultdict(list)

    t0 = time.time()
    processed = 0
    for df in (s2_df, s3_df):
        for eid, name, addr in zip(df["entity_id"], df["business_name"], df["business_address"]):
            nn = cand_name[eid]
            na = cand_addr[eid]

            for tok in set(name_tokens(nn)):
                if tok not in name_stop:
                    name_tok_idx[tok].append(eid)

            for tg in set(char_trigrams(nn)):
                trigram_idx[tg].append(eid)

            for tok in set(addr_tokens(na)):
                if tok not in addr_stop:
                    addr_tok_idx[tok].append(eid)

            processed += 1
            if processed % 200000 == 0:
                logging.info(f"  Indexed {processed:,}/{total_cands:,} candidates "
                             f"({time.time()-t0:.1f}s elapsed)")

    # Prune oversized buckets to bound memory & avoid useless huge blocks
    logging.info(f"Pruning oversized buckets (cap={args.max_bucket})...")
    name_tok_idx = {k: v for k, v in name_tok_idx.items() if len(v) <= args.max_bucket}
    trigram_idx = {k: v for k, v in trigram_idx.items() if len(v) <= args.max_bucket}
    addr_tok_idx = {k: v for k, v in addr_tok_idx.items() if len(v) <= args.max_bucket}
    logging.info(f"Index sizes -> name_tok: {len(name_tok_idx):,} | "
                 f"trigram: {len(trigram_idx):,} | addr_tok: {len(addr_tok_idx):,}")
    logging.info(f"Indexing complete in {time.time()-t0:.1f}s total.")

    # ---- S1 country map ----
    s1_country = {eid: norm_country(c) for eid, c in zip(s1_df["entity_id"], s1_df["country"])}

    # ---- Lazy Memoization Caches for Candidate Feature Sets ----
    cand_toks_cache: Dict[str, Set[str]] = {}
    cand_tris_cache: Dict[str, Set[str]] = {}
    cand_atoks_cache: Dict[str, Set[str]] = {}

    def get_cand_name_toks(c: str) -> Set[str]:
        if c not in cand_toks_cache:
            cand_toks_cache[c] = set(name_tokens(cand_name.get(c, "")))
        return cand_toks_cache[c]

    def get_cand_tris(c: str) -> Set[str]:
        if c not in cand_tris_cache:
            cand_tris_cache[c] = set(char_trigrams(cand_name.get(c, "")))
        return cand_tris_cache[c]

    def get_cand_addr_toks(c: str) -> Set[str]:
        if c not in cand_atoks_cache:
            cand_atoks_cache[c] = set(addr_tokens(cand_addr.get(c, "")))
        return cand_atoks_cache[c]

    # ---- Stream through S1 entities, block + score + prune, write incrementally ----
    logging.info(f"Processing {len(s1_df):,} S1 entities (writing to {args.output_jsonl})...")
    t1 = time.time()
    n_s1 = len(s1_df)

    already_done = set()
    if os.path.exists(args.output_jsonl):
        with open(args.output_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    already_done.add(json.loads(line)["s1_id"])
                except Exception:
                    pass
        logging.info(f"Resuming: {len(already_done):,} S1 entities already processed, skipping them.")

    with open(args.output_jsonl, "a", encoding="utf-8") as out_f:
        for i, (eid, name, addr, country) in enumerate(
            zip(s1_df["entity_id"], s1_df["business_name"], s1_df["business_address"], s1_df["country"])
        ):
            if eid in already_done:
                continue
            nn = norm_name(name)
            na = norm_text(addr)
            s1_c = norm_country(country)

            n_toks = set(name_tokens(nn))
            n_tris = set(char_trigrams(nn))
            a_toks = set(addr_tokens(na))

            cand_ids = set()
            for tok in n_toks:
                if tok in name_tok_idx:
                    cand_ids.update(name_tok_idx[tok])
            for tg in n_tris:
                if tg in trigram_idx:
                    cand_ids.update(trigram_idx[tg])
            for tok in a_toks:
                if tok in addr_tok_idx:
                    cand_ids.update(addr_tok_idx[tok])

            if i % 500 == 0:
                logging.info(f"  [debug] S1 {i}: raw cand_ids size = {len(cand_ids)}")

            # Country filter (lenient: only drop if both known and different)
            if s1_c:
                cand_ids = {
                    c for c in cand_ids
                    if not cand_country.get(c) or cand_country.get(c) == s1_c
                }

            # If the union is huge, cheaply pre-filter using raw token overlap count
            # (much cheaper than full Jaccard scoring) before running expensive scoring.
            MAX_TO_SCORE = 300
            if len(cand_ids) > MAX_TO_SCORE:
                cheap_prefilter = []
                for c in cand_ids:
                    c_toks = get_cand_name_toks(c)
                    overlap = len(n_toks & c_toks)
                    cheap_prefilter.append((overlap, c))
                cheap_prefilter.sort(key=lambda x: -x[0])
                cand_ids = {c for _, c in cheap_prefilter[:MAX_TO_SCORE]}

            # Cheap scoring for top-K pruning
            scored = []
            for c in cand_ids:
                c_toks = get_cand_name_toks(c)
                c_tris = get_cand_tris(c)
                c_atoks = get_cand_addr_toks(c)

                s_tok = jaccard(n_toks, c_toks)
                s_tri = jaccard(n_tris, c_tris)
                s_addr = jaccard(a_toks, c_atoks)
                score = 0.45 * s_tok + 0.35 * s_tri + 0.20 * s_addr
                scored.append((score, c))

            scored.sort(key=lambda x: -x[0])
            top = [c for _, c in scored[:args.top_k]]

            out_f.write(json.dumps({"s1_id": eid, "candidates": top}) + "\n")

            if (i + 1) % args.batch_log_every == 0:
                elapsed = time.time() - t1
                rate = (i + 1) / elapsed
                remaining = (n_s1 - (i + 1)) / rate if rate > 0 else 0
                out_f.flush()
                logging.info(
                    f"  Processed {i+1:,}/{n_s1:,} S1 entities "
                    f"({rate:.1f}/sec, ~{remaining/60:.1f} min remaining)"
                )

    logging.info(f"Done streaming all S1 entities in {(time.time()-t1)/60:.1f} min.")

    # ---- Convert JSONL to final JSON dict ----
    logging.info("Converting JSONL to final JSON dict...")
    final = {}
    with open(args.output_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            final[row["s1_id"]] = row["candidates"]

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(final, f)

    logging.info(f"Saved final candidates for {len(final):,} S1 entities to {args.output_json}")


if __name__ == "__main__":
    main()
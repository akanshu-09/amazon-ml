"""
Score a partial validation sample (e.g. 3K of the full 441,365 val_s1_ids) using
the trained pairwise model, scoped ONLY to the sampled entities so the other
~438K un-sampled val entities don't get incorrectly counted as wrong empty
predictions.

Uses the existing verified src/evaluation/metrics.py harness directly -- no
new scoring math, just wiring model probabilities + candidates into the
predictions dict format compute_f0_5_scores expects.

Run: python -m src.evaluation.score_val_sample \
        --features output/val_features_3k.parquet \
        --candidates output/candidates_val_3k.json \
        --model output/lgbm_pairwise_model.txt \
        --ground-truth dataset/train/train_ground_truth.tsv \
        --match-thresholds 0.8 0.85 0.9 \
        --singleton-thresholds 0.9 0.93 0.95
"""
import argparse
import itertools
import json
import logging

import lightgbm as lgb
import pandas as pd

from src.evaluation.metrics import compute_f0_5_scores, load_ground_truth

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, help="Parquet feature matrix for the val sample.")
    ap.add_argument("--candidates", required=True, help="Candidates JSON for the val sample (s1_id -> [cand_ids]).")
    ap.add_argument("--model", required=True, help="Path to trained LightGBM model (.txt).")
    ap.add_argument("--ground-truth", default="dataset/train/train_ground_truth.tsv")
    ap.add_argument("--s1-id-col", default="s1_id")
    ap.add_argument("--cand-id-col", default="candidate_id")

    # Threshold options
    ap.add_argument("--match-threshold", type=float, help="Single match threshold (candidate kept if prob >= match_threshold).")
    ap.add_argument("--singleton-threshold", type=float, help="Single singleton threshold (predict empty if max entity prob < singleton_threshold).")
    ap.add_argument("--match-thresholds", type=float, nargs="+", help="List of match thresholds to sweep over.")
    ap.add_argument("--singleton-thresholds", type=float, nargs="+", help="List of singleton thresholds to sweep over.")
    ap.add_argument("--thresholds", type=float, nargs="+", help="Legacy threshold list (sets both match and singleton threshold lists).")
    args = ap.parse_args()

    # Resolve threshold lists
    match_thresholds = args.match_thresholds
    if match_thresholds is None:
        if args.match_threshold is not None:
            match_thresholds = [args.match_threshold]
        elif args.thresholds is not None:
            match_thresholds = args.thresholds
        else:
            match_thresholds = [0.3, 0.5, 0.7, 0.8, 0.85, 0.9, 0.95]

    singleton_thresholds = args.singleton_thresholds
    if singleton_thresholds is None:
        if args.singleton_threshold is not None:
            singleton_thresholds = [args.singleton_threshold]
        elif args.thresholds is not None:
            singleton_thresholds = args.thresholds
        else:
            singleton_thresholds = match_thresholds

    logging.info(f"Loading candidates from {args.candidates}...")
    with open(args.candidates, "r", encoding="utf-8") as f:
        all_candidates = json.load(f)
    sample_s1_ids = set(all_candidates.keys())
    logging.info(f"Sample scope: {len(sample_s1_ids):,} S1 entities (this is our full evaluation universe).")

    logging.info(f"Loading feature matrix from {args.features}...")
    df = pd.read_parquet(args.features)
    feature_cols = [
        c for c in df.columns
        if c not in {args.s1_id_col, args.cand_id_col, "label"}
    ]
    logging.info(f"Using {len(feature_cols)} feature columns for prediction.")

    logging.info(f"Loading model from {args.model}...")
    model = lgb.Booster(model_file=args.model)

    logging.info("Scoring all pairs...")
    df["prob"] = model.predict(df[feature_cols].astype(float).values)

    logging.info(f"Loading ground truth from {args.ground_truth}...")
    full_gt = load_ground_truth(args.ground_truth)
    gt_scoped = {sid: full_gt.get(sid, []) for sid in sample_s1_ids}
    logging.info(f"Ground truth scoped to {len(gt_scoped):,} sampled entities.")

    # Pre-group candidate data by entity for efficient threshold sweeping
    logging.info("Grouping entity candidate predictions...")
    entity_candidates = {}
    for sid, group in df.groupby(args.s1_id_col):
        probs = group["prob"].to_numpy()
        cand_ids = group[args.cand_id_col].to_numpy()
        max_p = float(probs.max()) if len(probs) > 0 else 0.0
        entity_candidates[sid] = (cand_ids, probs, max_p)

    logging.info("=" * 90)
    logging.info(f" THRESHOLD SWEEP -- {len(match_thresholds)} match thresh x {len(singleton_thresholds)} singleton thresh = {len(match_thresholds)*len(singleton_thresholds)} combinations")
    logging.info("=" * 90)

    results = []
    for match_thresh, singleton_thresh in itertools.product(match_thresholds, singleton_thresholds):
        predictions = {}
        for sid in sample_s1_ids:
            if sid not in entity_candidates:
                predictions[sid] = []
                continue
            cand_ids, probs, max_p = entity_candidates[sid]
            if max_p < singleton_thresh:
                predictions[sid] = []
            else:
                predictions[sid] = cand_ids[probs >= match_thresh].tolist()

        per_entity_df, summary = compute_f0_5_scores(predictions, gt_scoped)
        n_predicted_nonempty = sum(1 for v in predictions.values() if v)
        results.append({
            "match_thresh": match_thresh,
            "singleton_thresh": singleton_thresh,
            "macro_f0.5": summary["macro_f0.5"],
            "macro_precision": summary["macro_precision"],
            "macro_recall": summary["macro_recall"],
            "singleton_accuracy": summary["singleton_accuracy"],
            "entities_with_predictions": n_predicted_nonempty,
            "summary": summary
        })

    # Sort best-first by macro_f0.5
    results.sort(key=lambda x: x["macro_f0.5"], reverse=True)

    logging.info("=" * 90)
    logging.info(" TWO-THRESHOLD SWEEP RESULTS (Sorted best-first by macro F0.5)")
    logging.info("=" * 90)
    logging.info(f"{'Rank':<5} | {'Match Thresh':<12} | {'Singleton Thresh':<16} | {'Macro F0.5':<10} | {'Macro Prec':<10} | {'Macro Rec':<10} | {'Sing Acc':<9} | {'With Preds'}")
    logging.info("-" * 90)
    for i, res in enumerate(results, 1):
        logging.info(
            f"{i:<5} | {res['match_thresh']:<12.4f} | {res['singleton_thresh']:<16.4f} | "
            f"{res['macro_f0.5']:<10.4f} | {res['macro_precision']:<10.4f} | {res['macro_recall']:<10.4f} | "
            f"{res['singleton_accuracy']:<9.4f} | {res['entities_with_predictions']}/{len(sample_s1_ids)}"
        )
    logging.info("=" * 90)
    best = results[0]
    logging.info(
        f"BEST COMBINATION: match_threshold={best['match_thresh']:.4f}, singleton_threshold={best['singleton_thresh']:.4f} -> macro_f0.5={best['macro_f0.5']:.4f}"
    )


if __name__ == "__main__":
    main()


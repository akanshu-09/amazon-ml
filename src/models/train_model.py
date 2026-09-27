"""
Train a pairwise LightGBM classifier on the feature matrix produced by
src/features/build_features.py.

Per project notes: ONE pairwise classifier is trained here. Match-vs-singleton
decisioning is handled downstream via two separate probability thresholds
(one for "accept as match", one stricter one for "accept as singleton/empty"),
tuned later against the F0.5 eval harness — NOT via a second model.

This script's internal train/holdout split is a DEV split for sanity-checking
training (grouped by S1 entity so no entity's pairs leak across the split).
It is NOT the official validation_split.json val set -- that evaluation happens
separately via src/evaluation/evaluate.py once candidates+features exist for
val_s1_ids too.

Run directly: python train_model.py --features output/train_features_10k_clean.parquet
"""
import argparse
import json
import logging

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, fbeta_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, help="Path to feature matrix parquet file.")
    ap.add_argument("--s1-id-col", default="source1_entity_id")
    ap.add_argument("--cand-id-col", default="candidate_entity_id")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--test-size", type=float, default=0.2, help="Fraction of S1 groups held out for dev metrics.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-model", default="output/lgbm_pairwise_model.txt")
    ap.add_argument("--output-importance", default="output/feature_importance.csv")
    args = ap.parse_args()

    logging.info(f"Loading feature matrix from {args.features}...")
    df = pd.read_parquet(args.features)
    logging.info(f"Loaded {len(df):,} rows. Columns: {list(df.columns)}")

    if args.s1_id_col not in df.columns or args.cand_id_col not in df.columns:
        raise SystemExit(
            f"Expected id columns '{args.s1_id_col}' / '{args.cand_id_col}' not found in "
            f"{list(df.columns)}. Re-run with --s1-id-col / --cand-id-col set correctly."
        )
    if args.label_col not in df.columns:
        raise SystemExit(f"Label column '{args.label_col}' not found in {list(df.columns)}.")

    n_before = len(df)
    df = df.dropna(subset=[args.label_col])
    n_after = len(df)
    if n_after < n_before:
        logging.info(f"Dropped {n_before - n_after:,} rows with missing label (no ground-truth entry).")

    feature_cols = [
        c for c in df.columns
        if c not in {args.s1_id_col, args.cand_id_col, args.label_col}
    ]
    logging.info(f"Using {len(feature_cols)} feature columns: {feature_cols}")

    X = df[feature_cols].astype(float).values
    y = df[args.label_col].astype(int).values
    groups = df[args.s1_id_col].values

    pos_rate = y.mean()
    logging.info(f"Overall positive rate: {pos_rate:.4%} ({y.sum():,}/{len(y):,})")

    gss = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
    train_idx, dev_idx = next(gss.split(X, y, groups=groups))
    logging.info(
        f"Dev split: {len(train_idx):,} train pairs ({len(set(groups[train_idx])):,} S1 entities) | "
        f"{len(dev_idx):,} dev pairs ({len(set(groups[dev_idx])):,} S1 entities)"
    )

    X_train, y_train = X[train_idx], y[train_idx]
    X_dev, y_dev = X[dev_idx], y[dev_idx]

    train_set = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)
    dev_set = lgb.Dataset(X_dev, label=y_dev, feature_name=feature_cols, reference=train_set)

    params = {
        "objective": "binary",
        "metric": ["auc", "average_precision"],
        "is_unbalance": True,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 5,
        "seed": args.seed,
        "verbosity": -1,
    }

    logging.info("Training LightGBM model...")
    model = lgb.train(
        params,
        train_set,
        num_boost_round=1000,
        valid_sets=[train_set, dev_set],
        valid_names=["train", "dev"],
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=50)],
    )

    logging.info(f"Best iteration: {model.best_iteration}")

    dev_probs = model.predict(X_dev, num_iteration=model.best_iteration)
    auc = roc_auc_score(y_dev, dev_probs)
    pr_auc = average_precision_score(y_dev, dev_probs)

    logging.info("=" * 70)
    logging.info(" DEV SET METRICS (sanity check -- NOT the official val_s1_ids score)")
    logging.info("=" * 70)
    logging.info(f" ROC-AUC:  {auc:.4f}")
    logging.info(f" PR-AUC:   {pr_auc:.4f}")

    for thresh in [0.3, 0.5, 0.7, 0.9]:
        preds = (dev_probs >= thresh).astype(int)
        p = precision_score(y_dev, preds, zero_division=0)
        r = recall_score(y_dev, preds, zero_division=0)
        f05 = fbeta_score(y_dev, preds, beta=0.5, zero_division=0)
        logging.info(f" @ threshold {thresh:.1f} -> precision={p:.4f} recall={r:.4f} pair-level F0.5={f05:.4f}")

    logging.info("=" * 70)
    logging.info(" NOTE: pair-level F0.5 above is NOT the same as the competition's")
    logging.info(" entity-level macro F0.5 (which also scores singletons). Use")
    logging.info(" src/evaluation/evaluate.py with real candidate probabilities for that.")
    logging.info("=" * 70)

    importance = pd.DataFrame({
        "feature": feature_cols,
        "gain": model.feature_importance(importance_type="gain"),
        "split": model.feature_importance(importance_type="split"),
    }).sort_values("gain", ascending=False)
    importance.to_csv(args.output_importance, index=False)
    logging.info(f"Feature importance saved to {args.output_importance}")
    logging.info("Top 10 features by gain:\n" + importance.head(10).to_string(index=False))

    model.save_model(args.output_model)
    logging.info(f"Model saved to {args.output_model}")


if __name__ == "__main__":
    main()

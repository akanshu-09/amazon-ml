import argparse
import json
import os
import sys
from datetime import datetime

import pandas as pd

# Add the directory containing this script to sys.path to allow importing metrics
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from metrics import compute_f0_5_scores, load_ground_truth

def evaluate_predictions(predictions_tsv_path: str, ground_truth_tsv_path: str, split_json_path: str, notes: str = ""):
    print("Starting evaluation...")
    
    # 1. Load the validation split JSON (val_s1_ids field)
    with open(split_json_path, 'r') as f:
        split_data = json.load(f)
    val_s1_ids = set(split_data.get('val_s1_ids', []))
    
    if not val_s1_ids:
        print("Warning: validation split contains no S1 entities (val_s1_ids is empty).")
        
    # 2. Load ground truth, filter to val_s1_ids
    print(f"Loading ground truth from {ground_truth_tsv_path}...")
    full_ground_truth = load_ground_truth(ground_truth_tsv_path)
    val_ground_truth = {k: v for k, v in full_ground_truth.items() if k in val_s1_ids}
    
    # Check if there are s1_ids in val_s1_ids that are missing from full_ground_truth
    missing_gt = val_s1_ids - set(val_ground_truth.keys())
    if missing_gt:
        print(f"Warning: {len(missing_gt)} IDs in the validation split were NOT found in the ground truth file.")
        for s1_id in missing_gt:
            val_ground_truth[s1_id] = [] # Fallback
            
    # 3. Load predictions, filter to val_s1_ids
    #    Since predictions might have the same format as ground truth (source1_entity_id, matched_entity_ids)
    #    we can reuse load_ground_truth to parse it.
    print(f"Loading predictions from {predictions_tsv_path}...")
    try:
        full_predictions = load_ground_truth(predictions_tsv_path)
    except FileNotFoundError:
        print(f"Warning: predictions file not found at {predictions_tsv_path}. Treating all predictions as empty.")
        full_predictions = {}
        
    # 4. Fill missing predictions for val_s1_ids with empty lists (must be scored, not crashed)
    val_predictions = {}
    for s1_id in val_s1_ids:
        val_predictions[s1_id] = full_predictions.get(s1_id, [])
        
    # 5. Call compute_f0_5_scores
    print("Computing metrics...")
    df, summary = compute_f0_5_scores(val_predictions, val_ground_truth)
    
    # 6. Print a readable summary
    print("\n" + "="*50)
    print(" EVALUATION SUMMARY")
    print("="*50)
    print(f" Entities evaluated:   {summary['num_entities']}")
    print(f" Macro F0.5:           {summary['macro_f0.5']:.4f}")
    print(f" Macro Precision:      {summary['macro_precision']:.4f}")
    print(f" Macro Recall:         {summary['macro_recall']:.4f}")
    print(f" Singleton Accuracy:   {summary['singleton_accuracy']:.4f}")
    print(f" Actual Singletons:    {summary['num_actual_singletons']}")
    print(f" Predicted Singletons: {summary['num_predicted_singletons']}")
    print("="*50 + "\n")
    
    # 7. Save per-entity DataFrame
    os.makedirs('output', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    details_csv_path = f"output/eval_details_{timestamp}.csv"
    df.to_csv(details_csv_path, index=False)
    print(f"Detailed per-entity metrics saved to {details_csv_path}")
    
    # 8. Append to experiment_log.csv
    log_csv_path = "output/experiment_log.csv"
    log_row = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'predictions_file': predictions_tsv_path,
        'macro_f0.5': summary['macro_f0.5'],
        'macro_precision': summary['macro_precision'],
        'macro_recall': summary['macro_recall'],
        'singleton_accuracy': summary['singleton_accuracy'],
        'num_entities': summary['num_entities'],
        'notes': notes
    }
    
    log_df = pd.DataFrame([log_row])
    
    # Create headers if file doesn't exist, else append without headers
    if os.path.exists(log_csv_path):
        log_df.to_csv(log_csv_path, mode='a', header=False, index=False)
    else:
        log_df.to_csv(log_csv_path, mode='w', header=True, index=False)
        
    print(f"Evaluation summary appended to {log_csv_path}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate entity matching predictions based on F0.5 score.")
    parser.add_argument('--predictions', required=True, help="Path to predictions TSV file")
    parser.add_argument('--ground-truth', required=True, help="Path to ground truth TSV file")
    parser.add_argument('--split', required=True, help="Path to validation split JSON file")
    parser.add_argument('--notes', type=str, default="", help="Notes to append in the experiment log")
    
    args = parser.parse_args()
    
    evaluate_predictions(
        predictions_tsv_path=args.predictions,
        ground_truth_tsv_path=args.ground_truth,
        split_json_path=args.split,
        notes=args.notes
    )

if __name__ == "__main__":
    main()

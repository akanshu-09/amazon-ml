import pandas as pd
import numpy as np

def compute_f0_5_scores(predictions: dict, ground_truth: dict):
    """
    Computes precision, recall, and F0.5 scores per entity by comparing predicted 
    sets vs actual sets.
    
    Args:
        predictions: dict mapping s1_id -> list of predicted matched entity IDs
        ground_truth: dict mapping s1_id -> list of actual matched entity IDs
        
    Returns:
        tuple: (per_entity_df, summary_dict)
    """
    results = []
    
    for s1_id, actual_list in ground_truth.items():
        actual_set = set(actual_list)
        # Missing predictions are treated as an empty list (no matches predicted)
        pred_list = predictions.get(s1_id, [])
        pred_set = set(pred_list)
        
        num_actual = len(actual_set)
        num_pred = len(pred_set)
        
        correct_set = actual_set.intersection(pred_set)
        num_correct = len(correct_set)
        
        is_singleton_actual = (num_actual == 0)
        is_singleton_predicted = (num_pred == 0)
        
        # Explicit edge cases for F0.5 logic:
        if is_singleton_actual and is_singleton_predicted:
            # actual empty AND predicted empty
            precision = 1.0
            recall = 1.0
            f0_5 = 1.0
        elif is_singleton_actual and not is_singleton_predicted:
            # actual empty AND predicted non-empty -> precision=0, recall=0 (undefined)
            precision = 0.0
            recall = 0.0
            f0_5 = 0.0
        elif not is_singleton_actual and is_singleton_predicted:
            # actual non-empty AND predicted empty -> precision=0 (undefined), recall=0
            precision = 0.0
            recall = 0.0
            f0_5 = 0.0
        elif num_correct == 0:
            # actual non-empty AND predicted non-empty with zero overlap
            precision = 0.0
            recall = 0.0
            f0_5 = 0.0
        else:
            # Standard calculation
            precision = num_correct / num_pred
            recall = num_correct / num_actual
            # F0.5 formula: (1.25 * precision * recall) / (0.25 * precision + recall)
            f0_5 = (1.25 * precision * recall) / (0.25 * precision + recall)
            
        results.append({
            's1_id': s1_id,
            'num_predicted': num_pred,
            'num_actual': num_actual,
            'num_correct': num_correct,
            'precision': precision,
            'recall': recall,
            'f0.5': f0_5,
            'is_singleton_actual': is_singleton_actual,
            'is_singleton_predicted': is_singleton_predicted
        })
        
    df = pd.DataFrame(results)
    
    # Calculate summary dictionary (macro-averaging)
    macro_f0_5 = df['f0.5'].mean() if not df.empty else 0.0
    macro_precision = df['precision'].mean() if not df.empty else 0.0
    macro_recall = df['recall'].mean() if not df.empty else 0.0
    num_entities = len(df)
    
    num_actual_singletons = df['is_singleton_actual'].sum()
    num_predicted_singletons = df['is_singleton_predicted'].sum()
    
    # Fraction of actual-singleton entities where prediction was also empty
    actual_singletons_df = df[df['is_singleton_actual']]
    if len(actual_singletons_df) > 0:
        singleton_accuracy = actual_singletons_df['is_singleton_predicted'].mean()
    else:
        singleton_accuracy = 0.0
        
    summary = {
        'macro_f0.5': float(macro_f0_5),
        'macro_precision': float(macro_precision),
        'macro_recall': float(macro_recall),
        'num_entities': int(num_entities),
        'singleton_accuracy': float(singleton_accuracy),
        'num_actual_singletons': int(num_actual_singletons),
        'num_predicted_singletons': int(num_predicted_singletons)
    }
    
    return df, summary

def load_ground_truth(tsv_path: str):
    """
    Reads a TSV and returns a dict mapping source1_entity_id -> list of matched_entity_ids.
    Handles empty/NaN matched_entity_ids as an empty list (not ["nan"]).
    """
    df = pd.read_csv(tsv_path, sep='\t', dtype=str)
    result = {}
    
    for _, row in df.iterrows():
        s1_id = row['source1_entity_id']
        matches_str = row.get('matched_entity_ids', '')
        
        if s1_id in result:
            print(f"WARNING: duplicate source1_entity_id '{s1_id}' found in {tsv_path} — overwriting previous entry.")
        
        # Handle nan explicitly and empty whitespace strings
        if pd.isna(matches_str) or not str(matches_str).strip() or str(matches_str).lower() == 'nan':
            result[s1_id] = []
        else:
            # Comma-separated matching entity IDs
            result[s1_id] = [m.strip() for m in str(matches_str).split(',') if m.strip()]
            
    return result

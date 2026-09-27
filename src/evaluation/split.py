import argparse
import json
import logging
import os
import random
import sys

import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def create_split(input_csv: str, output_json: str, random_seed: int = 42, train_size: float = 0.8):
    """
    Reads a CSV file, extracts unique S1 entity IDs, and splits them into
    train and validation sets, saving the result to a JSON file.
    
    Args:
        input_csv (str): Path to the input CSV file.
        output_json (str): Path to the output JSON file to save the split.
        random_seed (int): Seed for deterministic random shuffling.
        train_size (float): Proportion of entities to put in the training set (0.0 to 1.0).
    """
    logging.info(f"Loading data from {input_csv}")
    
    try:
        # We try to read with headers first.
        sep = '\t' if str(input_csv).lower().endswith('.tsv') else ','
        df = pd.read_csv(input_csv, sep=sep)
        
        # Check if 'entity_id' column exists. If not, assume it might be headerless
        if 'entity_id' not in df.columns:
            logging.warning("'entity_id' column not found. Trying to read without headers...")
            df = pd.read_csv(input_csv, header=None, sep=sep)
            # Assuming first column is entity_id based on typical data structure
            df.rename(columns={0: 'entity_id'}, inplace=True)
            
        if 'entity_id' not in df.columns:
            logging.error("Could not find 'entity_id' column in the CSV.")
            sys.exit(1)
            
        # Extract unique entity IDs to ensure no entity is present in both splits (no leakage)
        unique_entities = df['entity_id'].dropna().unique()
        
        # Filter specifically for S1 entities (assuming format 'S1-...')
        s1_entities = [str(eid) for eid in unique_entities if str(eid).startswith('S1-')]
        
        if not s1_entities:
            logging.error("No S1 entity IDs found in the data.")
            sys.exit(1)
            
        total_entities = len(s1_entities)
        logging.info(f"Found {total_entities} unique S1 entities.")
        
        # Sort the entities first to ensure deterministic order across different environments 
        # before shuffling
        s1_entities.sort()
        
        # Set the random seed for reproducibility
        random.seed(random_seed)
        random.shuffle(s1_entities)
        
        # Calculate split index
        split_idx = int(total_entities * train_size)
        
        train_ids = s1_entities[:split_idx]
        val_ids = s1_entities[split_idx:]
        
        # Calculate percentages for summary
        train_count = len(train_ids)
        val_count = len(val_ids)
        train_pct = (train_count / total_entities) * 100
        val_pct = (val_count / total_entities) * 100
        
        logging.info(f"Split Summary:")
        logging.info(f"  Train: {train_count} entities ({train_pct:.1f}%)")
        logging.info(f"  Val:   {val_count} entities ({val_pct:.1f}%)")
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(os.path.abspath(output_json))
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Save to JSON
        output_data = {
            "train_s1_ids": train_ids,
            "val_s1_ids": val_ids
        }
        
        with open(output_json, 'w') as f:
            json.dump(output_data, f, indent=2)
            
        logging.info(f"Successfully saved validation split to {output_json}")
        
    except FileNotFoundError:
        logging.error(f"Input file not found: {input_csv}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Create a non-overlapping train/val split for S1 entities.")
    parser.add_argument(
        'input_csv', 
        type=str, 
        help="Path to the input CSV or TSV file containing entity data."
    )
    parser.add_argument(
        '--output_json', 
        type=str, 
        default="output/validation_split.json",
        help="Path to save the output JSON file. Default: output/validation_split.json"
    )
    parser.add_argument(
        '--seed', 
        type=int, 
        default=42,
        help="Random seed for splitting. Default: 42"
    )
    parser.add_argument(
        '--train_size', 
        type=float, 
        default=0.8,
        help="Proportion of the dataset to include in the train split. Default: 0.8"
    )
    
    args = parser.parse_args()
    
    create_split(
        input_csv=args.input_csv,
        output_json=args.output_json,
        random_seed=args.seed,
        train_size=args.train_size
    )


if __name__ == "__main__":
    main()

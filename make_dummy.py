import json, pandas as pd

with open('output/validation_split.json') as f:
    split = json.load(f)

pd.DataFrame({
    'source1_entity_id': split['val_s1_ids'],
    'matched_entity_ids': [''] * len(split['val_s1_ids'])
}).to_csv('output/dummy_all_empty.tsv', sep='\t', index=False)
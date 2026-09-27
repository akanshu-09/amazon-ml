import pandas as pd
import os

os.makedirs('output', exist_ok=True)

test_s1 = pd.read_csv('dataset/test/test_source1.tsv', sep='\t', dtype=str)

pd.DataFrame({
    'source1_entity_id': test_s1['entity_id'],
    'matched_entity_ids': [''] * len(test_s1)
}).to_csv('output/matching_results.tsv', sep='\t', index=False)

pd.DataFrame({
    'source1_entity_id': test_s1['entity_id'],
    'candidate_entity_ids': [''] * len(test_s1)
}).to_csv('output/candidate_pairs.tsv', sep='\t', index=False)

print(f"Done. Wrote {len(test_s1):,} rows to output/matching_results.tsv and output/candidate_pairs.tsv")
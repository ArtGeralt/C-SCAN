import pandas as pd
import re

# Load results
df = pd.read_csv('docking_results/best_scores_only_compounds.csv')

# Load blind set with SMILES and MOL_ID mapping
blind_set = pd.read_csv('docking_results/blind_set.csv', sep='\t', header=None, names=['SMILES', 'MOL_ID'])
print(f"Loaded {len(blind_set)} compounds from blind_set.csv")

print("="*60)
print("DOCKING RESULTS SUMMARY")
print("="*60)

# File already contains best scores only, no need to filter by conf_id
best_scores = df[['file_name', 'score']].copy()

# Extract compound index from file_name (compounds_N -> N)
best_scores['compound_idx'] = best_scores['file_name'].str.extract(r'compounds_(\d+)').astype(int)

# Map compounds_N to MOL_N where N are equal
# compounds_1 -> MOL_1 (row 0 in blind_set), compounds_8 -> MOL_8 (row 7), etc.
best_scores['MOL_ID'] = best_scores['compound_idx'].map(lambda x: blind_set.iloc[x-1]['MOL_ID'] if (x >= 1 and x <= len(blind_set)) else f'MOL_{x}')
best_scores['SMILES'] = best_scores['compound_idx'].map(lambda x: blind_set.iloc[x-1]['SMILES'] if (x >= 1 and x <= len(blind_set)) else 'N/A')

# Reorder columns and sort by score
best_scores = best_scores[['MOL_ID', 'SMILES', 'score', 'file_name']].sort_values('score')

print(f"\nTotal compounds docked: {len(best_scores)}")
print(f"Average best score: {best_scores['score'].mean():.2f} kcal/mol")
print(f"Best score overall: {best_scores['score'].min():.2f} kcal/mol")
print(f"Worst score: {best_scores['score'].max():.2f} kcal/mol")

# Count compounds by score threshold
thresholds = [-7.0, -6.5, -6.0, -5.5, -5.0]
print("\nCompounds by score threshold:")
for thresh in thresholds:
    count = (best_scores['score'] < thresh).sum()
    print(f"  Score < {thresh}: {count} compounds")

print("\n" + "="*60)
print("TOP 20 COMPOUNDS (Best pose only)")
print("="*60)
for idx, (_, row) in enumerate(best_scores.head(20).iterrows(), 1):
    smiles_preview = row['SMILES'][:40] + "..." if len(row['SMILES']) > 40 else row['SMILES']
    print(f"{idx:2d}. {row['MOL_ID']:10s} Score: {row['score']:6.2f} kcal/mol")
    print(f"    {smiles_preview}")

# Save results
output_columns = ['MOL_ID', 'SMILES', 'score', 'file_name']
best_scores[output_columns].to_csv('docking_results/best_docking_scores_from_BS.csv', index=False)
print(f"\n✓ Saved best scores with MOL_ID and SMILES to: docking_results/best_docking_scores_from_BS.csv")
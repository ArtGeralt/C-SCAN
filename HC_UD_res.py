import pandas as pd

# Load results
df = pd.read_csv('docking_results/results_BS.csv')

print("="*60)
print("DOCKING RESULTS SUMMARY")
print("="*60)

# Get ONLY the best score (conf_id 0) for each compound
best_scores = df[df['conf_id'] == 0][['file_name', 'score']].copy()
best_scores = best_scores.sort_values('score')

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
    print(f"{idx:2d}. {row['file_name']:20s} Score: {row['score']:6.2f} kcal/mol")

# Save results
best_scores.to_csv('best_scores_only_compounds.csv', index=False)
print(f"\n✓ Saved best scores to: best_scores_only_compounds.csv")
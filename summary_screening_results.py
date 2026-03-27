import pandas as pd

print('='*70)
print('BLIND SET SCREENING RESULTS SUMMARY')
print('='*70)

# Load results
all_df = pd.read_csv('predictions/20260130/blind_screening/all_predictions.csv')
high_df = pd.read_csv('predictions/20260130/blind_screening/high_efficiency_compounds.csv')

print(f'\nTotal compounds screened: {len(all_df)}')

print(f'\nClass predictions:')
print(f'  Class 0 (inactive): {sum(all_df["Predicted_Class"]==0)} compounds ({100*sum(all_df["Predicted_Class"]==0)/len(all_df):.1f}%)')
print(f'  Class 1 (active): {sum(all_df["Predicted_Class"]==1)} compounds ({100*sum(all_df["Predicted_Class"]==1)/len(all_df):.1f}%)')

print(f'\nHigh-confidence active compounds (Prob >= 0.7): {len(high_df)}')

print(f'\nTop 15 candidates for experimental testing:')
print(high_df.head(15)[['SMILES', 'Prob_1']])

print(f'\n\nOutput files:')
print('  Location: predictions/20260130/blind_screening/')
print('  - all_predictions.csv (all 3,364 compounds with predictions)')
print('  - high_efficiency_compounds.csv (240 high-confidence actives)')
print('  - probability_distribution.png (distribution plot)')
print('  - class_distribution.png (pie chart)')
print('  - top_high_efficiency_molecules.png (top 20 structures)')

print('\n'+'='*70)
print('Screening complete! Review the files above for detailed results.')
print('='*70)

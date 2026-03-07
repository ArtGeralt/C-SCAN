"""
Example script showing how to use your trained models to screen a blind set
"""

from classification_prediction_tool import load_classifier
import pandas as pd

print("="*70)
print("  Screening Blind Set with Trained Classification Models")
print("="*70)

# Load your trained XGBoost model (best performing: 98.7% accuracy)
print("\nLoading trained XGBoost classifier...")
classifier = load_classifier("models/20260130/classification", model_type='xgb')

print("\n" + "="*70)
print("  Model Successfully Loaded!")
print("="*70)
print("Classes: 0 (inactive), 1 (active)")
print("Performance: 98.7% Accuracy, 99.9% AUC-ROC")

# Example: Screen a few test compounds
print("\n" + "="*70)
print("  Example 1: Predicting Individual Compounds")
print("="*70)

test_compounds = [
    "CCN(CC)C(=O)Nc1ccc(C)c(Nc2nccc(n2)c3cccnc3)c1",  # Imatinib
    "CCN(CC)CCNC(=O)c1c(C)[nH]c(C=C2C(=O)Nc3ccc(F)cc23)c1C",  # Sunitinib
    "Cc1ccc(cc1)NC(=O)c2ccc(CN3CCN(C)CC3)cc2",  # Example compound
]

results, mols = classifier.predict(test_compounds, return_probabilities=True)

print("\nPredictions:")
if results is not None:
    print(results[['SMILES', 'Predicted_Class', 'Prob_0', 'Prob_1']])
else:
    print("No valid predictions returned.")

# Example: Screen from a file
print("\n" + "="*70)
print("  Example 2: Screening Blind Set from File")
print("="*70)

# If you have a blind set file, use this format:
blind_set_file = "blind_set/blind_set.csv"  # Your file with SMILES

# Check if file exists
import os
if os.path.exists(blind_set_file):
    print(f"\nScreening compounds from {blind_set_file}...")
    
    results = classifier.screen_blind_set(
        blind_set_file,
        output_dir="predictions/20260130/blind_screening",
        target_class='1',  # Looking for active compounds (class 1)
        confidence_threshold=0.7
    )
    
    print("\n✓ Screening complete!")
    print(f"Check predictions/20260130/blind_screening/ for results")
    
else:
    print(f"\nBlind set file '{blind_set_file}' not found.")
    print("\nTo screen your compounds:")
    print("  1. Create a CSV file with a 'SMILES' column")
    print("  2. Update blind_set_file = 'your_file.csv' in this script")
    print("  3. Run this script again")
    
    print("\nAlternatively, you can screen a list of SMILES directly:")
    print("  results = classifier.screen_blind_set(")
    print("      your_smiles_list,")
    print("      output_dir='predictions/screening',")
    print("      target_class=1,")
    print("      confidence_threshold=0.7")
    print("  )")

print("\n" + "="*70)
print("  Usage Complete!")
print("="*70)
print("\nYour models are ready to screen blind sets!")
print("Model location: models/20260130/classification/")
print("\nKey files:")
print("  - xgb_classifier.pkl (Best: 98.7% accuracy)")
print("  - rf_classifier.pkl (Good: 97.4% accuracy)")
print("  - *_confusion_matrix.png (Performance visualization)")
print("  - *_roc_curve.png (Model discrimination)")
print("  - *_feature_importance.png (Important molecular features)")

# Quick Start Guide - Training on Known Compounds

## Your Data Structure

You have these files in `known_compounds/` folder:
- **chembl_all_class.csv** - Main file with activity classes (0 or 1)
- **chembl_IC50.csv** - IC50 activity measurements
- **chembl_Kd.csv** - Kd binding measurements  
- **chembl_Ki.csv** - Ki inhibition constants
- **chembl_Inhibition.csv** - Inhibition percentages

## Single-Command Training

Run the complete pipeline with one command:

```bash
python prepare_and_train_classification.py
```

This will:
1. ✅ Load all your ChEMBL data files
2. ✅ Merge activity information (IC50, Kd, Ki, Inhibition)
3. ✅ Generate molecular descriptors
4. ✅ Train Random Forest and XGBoost classifiers
5. ✅ Generate evaluation plots and metrics

**Output location**: `models/YYYYMMDD/classification/`

## Configuration Options

Edit these settings in `prepare_and_train_classification.py`:

```python
# Use fingerprints instead of descriptors
USE_FINGERPRINTS = False  # Change to True for fingerprints

# Enable hyperparameter optimization (slower but better)
OPTIMIZE_HYPERPARAMETERS = False  # Change to True for best performance
```

## Expected Output

### Files Generated

**Data folder** (`data/processed/YYYYMMDD/`):
- `chembl_merged_all_data.csv` - All data merged together
- `chembl_classification_descriptors.csv` - Ready for training

**Models folder** (`models/YYYYMMDD/classification/`):
- `rf_classifier.pkl` - Random Forest model
- `xgb_classifier.pkl` - XGBoost model
- `*_scaler.pkl` - Feature scalers
- `*_label_encoder.pkl` - Label encoders
- `*_confusion_matrix.png` - Performance visualization
- `*_roc_curve.png` - ROC curves
- `*_feature_importance.png` - Important features
- `model_comparison.csv` - Model comparison

### Console Output

```
Activity class distribution:
  Class 0: 950 compounds (50.2%)
  Class 1: 943 compounds (49.8%)

Activity data coverage:
  IC50_pActivity: 747 compounds (39.5%)
  Kd_pActivity: 43 compounds (2.3%)
  Ki_pActivity: 227 compounds (12.0%)
  Inhibition_percent: 587 compounds (31.0%)

Random Forest Model Performance:
  Accuracy: 0.892
  F1 Score: 0.890
  AUC-ROC: 0.956
```

## Screening Your Blind Set

Once models are trained, screen new compounds:

```python
from classification_prediction_tool import load_classifier

# Load the trained model
classifier = load_classifier("models/20260130/classification", model_type='xgb')

# Screen your blind set
results = classifier.screen_blind_set(
    "path/to/your/blind_set.csv",  # CSV with SMILES column
    output_dir="predictions/20260130/blind_screening",
    target_class=1,  # Target active class (1 = active)
    confidence_threshold=0.7
)

# Results saved automatically:
# - all_predictions.csv
# - high_efficiency_compounds.csv
# - Visualization plots
```

## Blind Set Format

Your blind set file should be CSV with SMILES:

**Option 1 - Simple format:**
```csv
SMILES
CN1CCN(Cc2ccc(NC(=O)c3cccc(C)c3)cc2)CC1
Cc1ccc(cc1)NC(=O)c2ccc(CN3CCN(C)CC3)cc2
...
```

**Option 2 - With IDs:**
```csv
ID,SMILES
COMP001,CN1CCN(Cc2ccc(NC(=O)c3cccc(C)c3)cc2)CC1
COMP002,Cc1ccc(cc1)NC(=O)c2ccc(CN3CCN(C)CC3)cc2
...
```

## Troubleshooting

### "No module named 'xgboost'"
```bash
pip install xgboost
```

### "No module named 'seaborn'"
```bash
pip install seaborn
```

### Low Model Performance
- Enable hyperparameter optimization: `OPTIMIZE_HYPERPARAMETERS = True`
- Try fingerprints instead of descriptors: `USE_FINGERPRINTS = True`
- Check class balance in output (should be ~50/50)

### "File not found" errors
Make sure you run from the C-SCAN directory and `known_compounds/` folder exists with all CSV files.

## Understanding the Results

### Activity Classes
- **Class 0**: Inactive/low activity compounds
- **Class 1**: Active/high activity compounds

### Performance Metrics
- **Accuracy**: Overall correctness (target: >0.85)
- **F1 Score**: Balance of precision and recall (target: >0.80)
- **AUC-ROC**: Discrimination ability (target: >0.90)

### Confusion Matrix
```
           Predicted
           0    1
Actual 0  TN   FP
       1  FN   TP
```
- **TN** (True Negative): Correctly identified inactive
- **FP** (False Positive): Wrongly predicted as active
- **FN** (False Negative): Missed active compounds
- **TP** (True Positive): Correctly identified active

### ROC Curve
- Shows trade-off between sensitivity and specificity
- Higher AUC = better model discrimination
- AUC = 0.5 is random guessing
- AUC > 0.9 is excellent

## Next Steps After Screening

1. **Review high-confidence predictions**
   ```python
   import pandas as pd
   hits = pd.read_csv("predictions/20260130/blind_screening/high_efficiency_compounds.csv")
   print(f"Found {len(hits)} high-confidence active compounds")
   ```

2. **Export for testing**
   ```python
   # Get top 50 for experimental validation
   top_50 = hits.sort_values('Prob_1', ascending=False).head(50)
   top_50.to_csv("compounds_for_testing.csv", index=False)
   ```

3. **Visualize chemical space**
   - Check similarity to known actives
   - Identify novel scaffolds
   - Assess diversity

## Complete Workflow Summary

```
known_compounds/
  ├── chembl_all_class.csv    → Load training data
  ├── chembl_IC50.csv         ↓
  ├── chembl_Kd.csv           ↓
  ├── chembl_Ki.csv           ↓
  └── chembl_Inhibition.csv   ↓
                              ↓
          [Merge & Process]   ↓
                              ↓
    data/processed/YYYYMMDD/  ↓
      └── chembl_classification_descriptors.csv
                              ↓
      [Generate Descriptors]  ↓
                              ↓
         [Train Models]       ↓
                              ↓
    models/YYYYMMDD/classification/
      ├── rf_classifier.pkl
      ├── xgb_classifier.pkl
      └── evaluation plots
                              ↓
      [Screen Blind Set]      ↓
                              ↓
    blind_set.csv             ↓
                              ↓
    predictions/YYYYMMDD/blind_screening/
      ├── all_predictions.csv
      └── high_efficiency_compounds.csv
```

## Questions?

Check the detailed documentation in [CLASSIFICATION_WORKFLOW.md](CLASSIFICATION_WORKFLOW.md)

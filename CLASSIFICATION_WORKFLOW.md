# Classification Model Training and Prediction Workflow

This guide explains how to train a classification model on known ChEMBL molecules with activity classes and use it to screen a blind set for high-efficiency compounds.

## Overview

You have two new scripts:
1. **`classification_model.py`** - Train classification models
2. **`classification_prediction_tool.py`** - Screen blind sets for high-efficiency compounds

## Workflow

### Step 1: Prepare Your Training Data

Your training data CSV should contain:
- `canonical_smiles` - SMILES strings of your molecules
- `molecule_chembl_id` - ChEMBL IDs
- `activity_class` - Activity labels (e.g., 'active'/'inactive' or 'high'/'moderate'/'low')
- Either molecular **descriptors** (from descriptors.py) OR **fingerprints** (from generating_fingerprints.py)

#### Option A: Add Activity Classes to Existing Data

If you have pIC50 data, convert it to activity classes:

```python
import pandas as pd

# Load your data
df = pd.read_csv("data/processed/20251030/kit_descriptors_selected.csv")

# Define activity class based on pIC50 thresholds
def classify_activity(pic50):
    if pic50 < 6.0:
        return 'inactive'
    elif pic50 < 7.0:
        return 'moderate'
    else:
        return 'active'

# Or for binary classification:
def classify_binary(pic50):
    return 'active' if pic50 >= 7.0 else 'inactive'

# Apply classification
df['activity_class'] = df['pIC50'].apply(classify_binary)

# Check distribution
print(df['activity_class'].value_counts())

# Save
df.to_csv("data/processed/20251030/kit_descriptors_with_class.csv", index=False)
```

#### Option B: Use ChEMBL Activity Comments

If your ChEMBL data has activity comments:

```python
df = pd.read_csv("data/processed/20251030/kit_pic50_20251030.csv")

# Extract activity from comments if available
# Then generate descriptors or fingerprints
```

### Step 2: Train Classification Models

Run the classification model training script:

```bash
python classification_model.py
```

This will:
- Load your data with activity classes
- Train multiple models (Random Forest, XGBoost)
- Perform evaluation with confusion matrix, ROC curves, etc.
- Save trained models to `models/YYYYMMDD/`

**Important parameters in the script:**
```python
# In classification_model.py, adjust these:
input_file = "data/processed/20251030/kit_descriptors_with_class.csv"
activity_column = 'activity_class'  # Name of your activity column

# For better performance (slower training):
build_classification_model(
    input_file,
    activity_column='activity_class',
    model_type='xgb',  # or 'rf'
    optimize=True,     # Enable hyperparameter optimization
)
```

**Output files in `models/YYYYMMDD/`:**
- `rf_classifier.pkl` / `xgb_classifier.pkl` - Trained models
- `rf_scaler.pkl` / `xgb_scaler.pkl` - Feature scalers
- `rf_label_encoder.pkl` / `xgb_label_encoder.pkl` - Label encoders
- `rf_feature_names.pkl` / `xgb_feature_names.pkl` - Feature names
- `*_confusion_matrix.png` - Model evaluation plots
- `*_roc_curve.png` - ROC curves
- `*_feature_importance.png` - Important features
- `model_comparison.csv` - Comparison of all models

### Step 3: Screen Your Blind Set

Prepare your blind set as a CSV file with SMILES:

**blind_set.csv:**
```csv
SMILES
CC(C)Nc1ncnc2[nH]ccc12
CN1CCN(Cc2ccc(NC(=O)c3cccc(C)c3)cc2)CC1
...
```

Or a text file with one SMILES per line.

Run the screening:

```bash
python classification_prediction_tool.py
```

Or customize the script:

```python
from classification_prediction_tool import load_classifier

# Load your trained model
classifier = load_classifier("models/20251030", model_type='xgb')

# Screen blind set
results = classifier.screen_blind_set(
    "data/blind_set.csv",
    output_dir="predictions/20251030/blind_screening",
    target_class='active',      # Target class to identify
    confidence_threshold=0.7    # Minimum probability threshold
)

# Results saved to:
# - all_predictions.csv - All compounds with predictions
# - high_efficiency_compounds.csv - High-confidence active compounds
# - Visualization plots
```

### Step 4: Analyze Results

Check the screening results:

```python
import pandas as pd

# Load high-efficiency compounds
high_eff = pd.read_csv("predictions/20251030/blind_screening/high_efficiency_compounds.csv")

print(f"Found {len(high_eff)} high-efficiency compounds")
print(high_eff.head(10))

# Sort by confidence
high_eff_sorted = high_eff.sort_values('Prob_active', ascending=False)

# Get top 20 for further testing
top_20 = high_eff_sorted.head(20)
top_20.to_csv("predictions/20251030/top_20_for_testing.csv", index=False)
```

## Advanced Usage

### Compare Specific Compounds

```python
from classification_prediction_tool import load_classifier

classifier = load_classifier("models/20251030", model_type='xgb')

# Compare specific compounds
test_smiles = [
    "CCN(CC)C(=O)Nc1ccc(C)c(Nc2nccc(n2)c3cccnc3)c1",  # Imatinib
    "CCN(CC)CCNC(=O)c1c(C)[nH]c(C=C2C(=O)Nc3ccc(F)cc23)c1C",  # Sunitinib
]

results, mols = classifier.predict(test_smiles, return_probabilities=True)
print(results)

# Visualize comparison
classifier.compare_compounds(
    test_smiles,
    labels=["Imatinib", "Sunitinib"],
    output_path="compound_comparison.png"
)
```

### Batch Processing Large Libraries

```python
from classification_prediction_tool import load_classifier
import pandas as pd

classifier = load_classifier("models/20251030", model_type='xgb')

# Process in batches for very large libraries
library = pd.read_csv("data/large_library.csv")
batch_size = 1000

all_results = []
for i in range(0, len(library), batch_size):
    batch = library.iloc[i:i+batch_size]['SMILES'].tolist()
    results, _ = classifier.predict(batch)
    all_results.append(results)
    print(f"Processed {i+batch_size}/{len(library)}")

# Combine results
final_results = pd.concat(all_results, ignore_index=True)
final_results.to_csv("predictions/large_library_results.csv", index=False)
```

### Adjust Activity Class Definitions

If you want different activity thresholds:

```python
# Multi-class classification
def classify_activity(pic50):
    if pic50 < 5.5:
        return 'very_low'
    elif pic50 < 6.5:
        return 'low'
    elif pic50 < 7.5:
        return 'moderate'
    elif pic50 < 8.5:
        return 'high'
    else:
        return 'very_high'

# Then retrain the model with these classes
```

## Expected Results

### Model Performance Metrics
- **Accuracy**: 0.80-0.95 (depending on data quality)
- **F1 Score**: 0.75-0.93
- **AUC-ROC**: 0.85-0.98 for binary classification

### Screening Output
- Distribution of predicted classes across blind set
- High-confidence predictions (probability ≥ 0.7)
- Visualizations of top candidates

## Troubleshooting

### Issue: "activity_class column not found"
**Solution**: Make sure your CSV has a column named `activity_class` or adjust the `activity_column` parameter.

### Issue: "Feature mismatch"
**Solution**: Ensure your blind set uses the same feature generation method (descriptors or fingerprints) as your training data.

### Issue: Poor model performance
**Solutions**:
1. Enable hyperparameter optimization: `optimize=True`
2. Collect more training data
3. Try different activity class thresholds
4. Use ensemble of multiple models

### Issue: Imbalanced classes
**Solution**: The models use `class_weight='balanced'` by default, but you can also:
```python
# Oversample minority class
from imblearn.over_sampling import SMOTE
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X, y)
```

## Files Summary

### Input Files Needed
- Training data CSV with activity classes and features
- Blind set CSV or text file with SMILES

### Output Files Generated

**Training:**
- `models/YYYYMMDD/` - All model files and evaluation plots

**Prediction:**
- `predictions/YYYYMMDD/blind_screening/all_predictions.csv`
- `predictions/YYYYMMDD/blind_screening/high_efficiency_compounds.csv`
- Various visualization plots

## Next Steps

1. **Validate predictions**: Test top predictions experimentally
2. **Iterate**: Refine activity class definitions based on results
3. **Expand**: Add more diverse training data
4. **Deploy**: Integrate into your screening pipeline

## Questions?

Check the inline documentation in:
- `classification_model.py` - Model training details
- `classification_prediction_tool.py` - Prediction and screening details

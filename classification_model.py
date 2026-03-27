import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                             roc_auc_score, accuracy_score, f1_score,
                             precision_recall_curve, roc_curve)
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from xgboost import XGBClassifier
from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d")

def prepare_data(input_csv, activity_column='activity_class'):
    """
    Load and prepare data for classification
    
    Parameters:
    -----------
    input_csv : str
        Path to CSV file with descriptors/fingerprints and activity class
    activity_column : str
        Name of the column containing activity classes
        (e.g., 'active', 'inactive' or 'high', 'moderate', 'low')
    
    Returns:
    --------
    X : array
        Feature matrix
    y : array
        Target labels (encoded)
    feature_names : list
        Names of features
    label_encoder : LabelEncoder
        Fitted label encoder for inverse transform
    df : DataFrame
        Original dataframe
    """
    print(f"Loading data from {input_csv}")
    df = pd.read_csv(input_csv)

    # Check if activity column exists
    if activity_column not in df.columns:
        raise ValueError(
            f"Column '{activity_column}' not found in dataset.\n"
            f"Available columns: {df.columns.tolist()}\n"
            f"Hint: this file may be a regression dataset (pIC50). "
            f"Use the Classification Pipeline page to generate a classification dataset."
        )

    # Normalize labels and remove invalid placeholders.
    activity_series = df[activity_column].astype(str).str.strip()
    invalid_tokens = {"", "nan", "none", "null"}
    valid_mask = ~activity_series.str.lower().isin(invalid_tokens)
    if not bool(valid_mask.all()):
        removed = int((~valid_mask).sum())
        print(f"Dropping {removed} rows with invalid '{activity_column}' labels")
        df = df.loc[valid_mask].copy()
        activity_series = activity_series.loc[valid_mask]

    # Drop classes that occur once; stratified split/CV requires at least 2.
    class_counts = activity_series.value_counts()
    rare_classes = class_counts[class_counts < 2].index.tolist()
    if rare_classes:
        rare_mask = activity_series.isin(rare_classes)
        removed_rare = int(rare_mask.sum())
        print(f"Dropping {removed_rare} rows from rare classes (<2 samples): {rare_classes}")
        keep_mask = ~rare_mask
        df = df.loc[keep_mask].copy()
        activity_series = activity_series.loc[keep_mask]

    if activity_series.nunique() < 2:
        raise ValueError(
            "Need at least 2 classes with >=2 samples each after cleaning activity labels."
        )

    # Encode activity classes
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(activity_series)

    print(f"\nActivity class distribution:")
    for i, class_name in enumerate(label_encoder.classes_):
        count = int(np.sum(y == i))
        print(f"  {class_name}: {count} ({100*count/y.shape[0]:.1f}%)")

    # Separate features
    exclude_cols = [
        'molecule_chembl_id', 'canonical_smiles', 'SMILES', 'smiles', 'Smiles', 'pIC50',
        activity_column, 'Activity_Level',
        'IC50_pActivity', 'Kd_pActivity', 'Ki_pActivity', 'Inhibition_percent'
    ]

    # Candidate features before numeric coercion
    candidate_cols = [col for col in df.columns if col not in exclude_cols]
    if not candidate_cols:
        raise ValueError("No candidate feature columns found after excluding metadata/target columns.")

    # Coerce to numeric so string ID columns (e.g., ZINC IDs) are filtered out safely.
    feature_df = df[candidate_cols].apply(pd.to_numeric, errors='coerce')

    # Drop columns that are entirely non-numeric after coercion.
    dropped_cols = [col for col in feature_df.columns if feature_df[col].notna().sum() == 0]
    if dropped_cols:
        print(f"Dropping {len(dropped_cols)} non-numeric feature columns: {dropped_cols[:10]}")
        if len(dropped_cols) > 10:
            print(f"... and {len(dropped_cols) - 10} more")
        feature_df = feature_df.drop(columns=dropped_cols)

    if feature_df.shape[1] == 0:
        raise ValueError(
            "No numeric feature columns remained after parsing. "
            "Provide a descriptor/fingerprint table with numeric features plus activity_class."
        )

    feature_names = feature_df.columns.tolist()
    X = feature_df.values

    # Clean data - replace inf and nan values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"\nDataset shape: {X.shape[0]} compounds, {X.shape[1]} features")

    return X, y, feature_names, label_encoder, df


def train_classification_model(X_train, X_test, y_train, y_test, 
                               model_type='rf', optimize=True):
    """
    Train and evaluate a classification model
    
    Parameters:
    -----------
    X_train, X_test : arrays
        Training and test features
    y_train, y_test : arrays
        Training and test labels
    model_type : str
        'rf' for Random Forest, 'xgb' for XGBoost, 'gb' for Gradient Boosting
    optimize : bool
        Whether to perform hyperparameter optimization
    
    Returns:
    --------
    model : trained model
    metrics : dict of performance metrics
    """
    
    class_counts_train = pd.Series(y_train).value_counts()
    min_class_count_train = int(class_counts_train.min()) if not class_counts_train.empty else 0
    cv_folds = max(2, min(5, min_class_count_train)) if min_class_count_train >= 2 else 2

    if model_type == 'rf':
        if optimize:
            print("Optimizing Random Forest hyperparameters...")
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'class_weight': ['balanced', None]
            }
            model = RandomForestClassifier(random_state=42, n_jobs=-1)
            grid_search = GridSearchCV(model, param_grid, cv=cv_folds, 
                                      scoring='f1_weighted', n_jobs=-1, verbose=1)
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_
            print(f"Best parameters: {grid_search.best_params_}")
        else:
            model = RandomForestClassifier(
                n_estimators=200, 
                class_weight='balanced',
                random_state=42, 
                n_jobs=-1
            )
            model.fit(X_train, y_train)
        model_name = "Random Forest"
        
    elif model_type == 'xgb':
        if optimize:
            print("Optimizing XGBoost hyperparameters...")
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.3],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            }
            model = XGBClassifier(random_state=42, n_jobs=-1, eval_metric='logloss')
            grid_search = GridSearchCV(model, param_grid, cv=cv_folds, 
                                      scoring='f1_weighted', n_jobs=-1, verbose=1)
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_
            print(f"Best parameters: {grid_search.best_params_}")
        else:
            model = XGBClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1,
                eval_metric='logloss'
            )
            model.fit(X_train, y_train)
        model_name = "XGBoost"
        
    elif model_type == 'gb':
        model = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        model.fit(X_train, y_train)
        model_name = "Gradient Boosting"
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # For binary classification, calculate AUC-ROC
    if len(np.unique(y_test)) == 2:
        auc_roc = roc_auc_score(y_test, y_pred_proba[:, 1])
    else:
        auc_roc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
    
    metrics = {
        'model_name': model_name,
        'accuracy': accuracy,
        'f1_score': f1,
        'auc_roc': auc_roc
    }
    
    print(f"\n{model_name} Model Performance:")
    print(f"  Accuracy: {accuracy:.3f}")
    print(f"  F1 Score: {f1:.3f}")
    print(f"  AUC-ROC: {auc_roc:.3f}")
    
    return model, metrics


def evaluate_and_visualize(model, X_test, y_test, label_encoder, 
                          feature_names, model_type='rf', output_dir='models'):
    """Generate comprehensive evaluation plots and reports"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # 1. Classification Report
    print("\nClassification Report:")
    
    # Convert label encoder classes to strings to avoid issues
    class_names = [str(c) for c in label_encoder.classes_]
    
    report = classification_report(y_test, y_pred, 
                                   target_names=class_names,
                                   output_dict=True)
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    # Save classification report
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(f"{output_dir}/{model_type}_classification_report.csv")
    
    # 2. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title(f'Confusion Matrix - {model_type.upper()}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{model_type}_confusion_matrix.png", dpi=300)
    plt.close()
    
    # 3. ROC Curve (for binary classification)
    if len(label_encoder.classes_) == 2:
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
        auc_score = roc_auc_score(y_test, y_pred_proba[:, 1])
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'AUC = {auc_score:.3f}', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_type.upper()}')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{model_type}_roc_curve.png", dpi=300)
        plt.close()
        
        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba[:, 1])
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, linewidth=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {model_type.upper()}')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{model_type}_precision_recall.png", dpi=300)
        plt.close()
    
    # 4. Feature Importance
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[-20:]  # Top 20 features
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(indices)), importances[indices])
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Feature Importance')
        plt.title(f'Top 20 Most Important Features - {model_type.upper()}')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{model_type}_feature_importance.png", dpi=300)
        plt.close()
        
        # Save feature importance to CSV
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        importance_df.to_csv(f"{output_dir}/{model_type}_feature_importance.csv", index=False)
    
    print(f"\nEvaluation plots saved to {output_dir}/")
    
    return report_df


def build_classification_model(input_csv, activity_column='activity_class', 
                               test_size=0.2, model_type='rf', 
                               optimize=False, output_dir=None):
    """
    Complete workflow for building a classification model
    
    Parameters:
    -----------
    input_csv : str
        Path to CSV with features and activity classes
    activity_column : str
        Name of column containing activity classes
    test_size : float
        Fraction of data to use for testing
    model_type : str
        'rf', 'xgb', or 'gb'
    optimize : bool
        Whether to perform hyperparameter optimization (slower but better)
    output_dir : str
        Directory to save model and results
    
    Returns:
    --------
    model : trained model
    scaler : fitted scaler
    label_encoder : fitted label encoder
    metrics : performance metrics
    """
    
    if output_dir is None:
        output_dir = f"models/{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and prepare data
    X, y, feature_names, label_encoder, df = prepare_data(input_csv, activity_column)
    
    if X is None:
        return None, None, None, None
    assert feature_names is not None
    assert label_encoder is not None

    # Split data with stratification-safe test size.
    y_n = int(np.asarray(y).shape[0])
    n_classes = int(np.unique(y).shape[0])
    min_test_fraction = n_classes / max(y_n, 1)
    effective_test_size = max(float(test_size), min_test_fraction)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=effective_test_size, random_state=42, stratify=y
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\nTraining set: {len(X_train)} compounds")
    print(f"Test set: {len(X_test)} compounds")
    
    # Train model
    model, metrics = train_classification_model(
        X_train_scaled, X_test_scaled, y_train, y_test,
        model_type=model_type, optimize=optimize
    )
    
    # Evaluate and visualize
    report_df = evaluate_and_visualize(
        model, X_test_scaled, y_test, label_encoder, 
        feature_names, model_type, output_dir
    )
    
    # Save model, scaler, and label encoder
    with open(f"{output_dir}/{model_type}_classifier.pkl", 'wb') as f:
        pickle.dump(model, f)
    with open(f"{output_dir}/{model_type}_scaler.pkl", 'wb') as f:
        pickle.dump(scaler, f)
    with open(f"{output_dir}/{model_type}_label_encoder.pkl", 'wb') as f:
        pickle.dump(label_encoder, f)
    
    # Save feature names
    with open(f"{output_dir}/{model_type}_feature_names.pkl", 'wb') as f:
        pickle.dump(feature_names, f)
    
    # Save summary
    summary = pd.DataFrame({
        'model_type': [model_type],
        'accuracy': [metrics['accuracy']],
        'f1_score': [metrics['f1_score']],
        'auc_roc': [metrics['auc_roc']],
        'n_features': [len(feature_names)],
        'n_train': [len(X_train)],
        'n_test': [len(X_test)],
        'classes': [', '.join([str(c) for c in label_encoder.classes_])]
    })
    summary.to_csv(f"{output_dir}/{model_type}_model_summary.csv", index=False)
    
    print(f"\nModel and results saved to {output_dir}/")
    print(f"  - {model_type}_classifier.pkl")
    print(f"  - {model_type}_scaler.pkl")
    print(f"  - {model_type}_label_encoder.pkl")
    
    return model, scaler, label_encoder, metrics


if __name__ == "__main__":
    # Example usage
    
    # Option 1: If you have descriptors with activity class
    input_file = f"data/processed/{timestamp}/kit_descriptors_with_class.csv"
    
    # Option 2: If you have fingerprints with activity class
    # input_file = f"data/processed/{timestamp}/kit_fingerprints_with_class.csv"
    
    # Check if file exists
    if not os.path.exists(input_file):
        print(f"File not found: {input_file}")
        print("\nPlease provide a CSV file with:")
        print("  - Molecular descriptors or fingerprints")
        print("  - An 'activity_class' column (e.g., 'active'/'inactive' or 'high'/'moderate'/'low')")
        print("  - 'canonical_smiles' column")
        print("  - 'molecule_chembl_id' column")
    else:
        # Train multiple models and compare
        models = {}
        
        # Random Forest (quick)
        print("\n" + "="*60)
        print("Training Random Forest Classifier")
        print("="*60)
        rf_model, rf_scaler, rf_encoder, rf_metrics = build_classification_model(
            input_file, 
            activity_column='activity_class',
            model_type='rf',
            optimize=False  # Set to True for better performance (slower)
        )
        models['rf'] = rf_metrics
        
        # XGBoost (usually best performance)
        print("\n" + "="*60)
        print("Training XGBoost Classifier")
        print("="*60)
        xgb_model, xgb_scaler, xgb_encoder, xgb_metrics = build_classification_model(
            input_file,
            activity_column='activity_class',
            model_type='xgb',
            optimize=False  # Set to True for better performance (slower)
        )
        models['xgb'] = xgb_metrics
        
        # Compare models
        print("\n" + "="*60)
        print("Model Comparison Summary")
        print("="*60)
        comparison_df = pd.DataFrame(models).T
        print(comparison_df)
        comparison_df.to_csv(f"models/{timestamp}/model_comparison.csv")
        
        print("\n✓ All models trained successfully!")
        print(f"Check models/{timestamp}/ for results")

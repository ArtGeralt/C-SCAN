import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pickle
import os
from xgboost import XGBRegressor  # Add XGBoost import

timestamp = ""

def build_model(input_csv, test_size=0.2, random_state=42, model_type='rf'):
    """Build a QSAR model using molecular descriptors"""
    print(f"Loading data from {input_csv}")
    df = pd.read_csv(input_csv)
    
    # Separate features (descriptors) and target (pIC50)
    y = df['pIC50'].values
    
    # Get only the descriptor columns (exclude identifier columns and target)
    exclude_cols = ['molecule_chembl_id', 'canonical_smiles', 'pIC50']
    X = df.drop(columns=exclude_cols).values
    
    # Split the dataset first (to avoid data leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Standardize features (fit only on training data)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Training set: {X_train.shape[0]} compounds")
    print(f"Test set: {X_test.shape[0]} compounds")
    
    # Choose model based on model_type
    if model_type == 'xgb':
        print("Training XGBoost model...")
        model = XGBRegressor(
            n_estimators=200, 
            learning_rate=0.1,
            max_depth=6,
            min_child_weight=1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            n_jobs=-1
        )
        model_name = "XGBoost"
    else:  # Default to RandomForest
        print("Training Random Forest model...")
        model = RandomForestRegressor(
            n_estimators=100, 
            random_state=random_state, 
            n_jobs=-1
        )
        model_name = "RandomForest"
    
    # Train model
    model.fit(X_train_scaled, y_train)
    
    # Evaluate on test set
    y_pred = model.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"Test set RMSE: {rmse:.3f}")
    print(f"Test set R²: {r2:.3f}")
    
    # Plot predicted vs actual
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--')
    plt.xlabel('Actual pIC50')
    plt.ylabel('Predicted pIC50')
    plt.title(f'{model_name} Model (R²={r2:.3f}, RMSE={rmse:.3f})')
    
    # Save plot
    os.makedirs("models", exist_ok=True)
    plt.savefig(f"models/{model_type}_prediction_plot.png")
    plt.close()
    
    # Save model and scaler
    with open(f"models/{model_type}_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(f"models/{model_type}_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    
    print(f"Model saved to models/{model_type}_model.pkl")
    
    # Feature importance
    feature_names = df.drop(columns=exclude_cols).columns
    
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif model_type == 'xgb':
        # For XGBoost
        importance_type = 'weight'  # or 'gain', 'cover', 'total_gain', 'total_cover'
        importances = model.get_booster().get_score(importance_type=importance_type)
        # Convert to array matching feature order
        imp_array = np.zeros(len(feature_names))
        for key, value in importances.items():
            try:
                idx = int(key.replace('f', ''))
                if idx < len(imp_array):
                    imp_array[idx] = value
            except:
                continue
        importances = imp_array
    
    # Get top 15 features
    indices = np.argsort(importances)[-15:]
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(indices)), importances[indices], align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.title(f'Top 15 Most Important Features ({model_name})')
    plt.tight_layout()
    plt.savefig(f"models/{model_type}_feature_importance.png")
    plt.close()
    
    return model, scaler, (rmse, r2)

if __name__ == "__main__":
    input_file = "data/processed/kit_descriptors_selected.csv"
    
    # Run Random Forest model
    rf_model, rf_scaler, rf_metrics = build_model(input_file, model_type='rf')
    
    # Run XGBoost model
    xgb_model, xgb_scaler, xgb_metrics = build_model(input_file, model_type='xgb')
    
    # Compare results
    print("\nModel Comparison:")
    print(f"RandomForest - R²: {rf_metrics[1]:.3f}, RMSE: {rf_metrics[0]:.3f}")
    print(f"XGBoost - R²: {xgb_metrics[1]:.3f}, RMSE: {xgb_metrics[0]:.3f}")
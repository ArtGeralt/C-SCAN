import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor
from sklearn.inspection import permutation_importance
import os
import pickle

timestamp = ""

def build_subcluster_model(subcluster_file, subcluster_id=0, min_pic50=5.0, max_pic50=10.0):
    """
    Build a regression model specifically for a subcluster with filtered pIC50 range
    """
    print(f"Loading data from {subcluster_file}")
    df = pd.read_csv(subcluster_file)
    
    # Filter by subcluster and pIC50 range
    filtered_df = df[(df['subcluster'] == subcluster_id) & 
                     (df['pIC50'] >= min_pic50) & 
                     (df['pIC50'] < max_pic50)].copy()
    
    print(f"Selected {len(filtered_df)} compounds from subcluster {subcluster_id} "
          f"with pIC50 between {min_pic50} and {max_pic50}")
    
    # Get fingerprint columns
    exclude_cols = ['molecule_chembl_id', 'canonical_smiles', 'pIC50', 'cluster', 'subcluster', 'scaffold_smiles']
    feature_cols = [col for col in filtered_df.columns if col not in exclude_cols]
    
    # Check if we have enough data
    if len(filtered_df) < 50:
        print("Warning: Very small dataset size. Results may not be reliable.")
    
    # Get features and target
    X = filtered_df[feature_cols].values
    y = filtered_df['pIC50'].values
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training set: {X_train.shape[0]} compounds")
    print(f"Test set: {X_test.shape[0]} compounds")
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create output directories
    os.makedirs("models/subcluster_models", exist_ok=True)
    
    # Try both RandomForest and XGBoost
    models = {
        'rf': RandomForestRegressor(n_estimators=100, random_state=42),
        'xgb': XGBRegressor(n_estimators=100, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name.upper()} model...")
        
        # First try with default parameters
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        
        # Evaluate
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        results[name] = {
            'model': model,
            'rmse': rmse,
            'r2': r2,
            'mae': mae
        }
        
        print(f"{name.upper()} Results:")
        print(f"  RMSE: {rmse:.3f}")
        print(f"  R²: {r2:.3f}")
        print(f"  MAE: {mae:.3f}")
        
        # Plot actual vs predicted
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, y_pred, alpha=0.7)
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--')
        plt.xlabel('Actual pIC50')
        plt.ylabel('Predicted pIC50')
        plt.title(f'Subcluster {subcluster_id} - {name.upper()} Model\nR² = {r2:.3f}, RMSE = {rmse:.3f}')
        plt.tight_layout()
        plt.savefig(f"models/subcluster_models/subcluster_{subcluster_id}_{name}_performance.png")
        plt.close()
        
        # Save the model
        with open(f"models/subcluster_models/subcluster_{subcluster_id}_{name}_model.pkl", 'wb') as f:
            pickle.dump(model, f)
        
        # Also save the scaler
        with open(f"models/subcluster_models/subcluster_{subcluster_id}_scaler.pkl", 'wb') as f:
            pickle.dump(scaler, f)
    
    # Identify the best model
    best_model_name = min(results, key=lambda x: results[x]['rmse'])
    print(f"\nBest model: {best_model_name.upper()} with R² = {results[best_model_name]['r2']:.3f}")
    
    # Perform hyperparameter optimization for the best model
    print("\nPerforming hyperparameter optimization...")
    if best_model_name == 'rf':
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        base_model = RandomForestRegressor(random_state=42)
    else:  # xgb
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }
        base_model = XGBRegressor(random_state=42)
    
    # Perform grid search
    grid_search = GridSearchCV(
        base_model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1
    )
    grid_search.fit(X_train_scaled, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    best_model = grid_search.best_estimator_
    
    # Evaluate optimized model
    y_pred_optimized = best_model.predict(X_test_scaled)
    rmse_optimized = np.sqrt(mean_squared_error(y_test, y_pred_optimized))
    r2_optimized = r2_score(y_test, y_pred_optimized)
    mae_optimized = mean_absolute_error(y_test, y_pred_optimized)
    
    print("\nOptimized model performance:")
    print(f"  RMSE: {rmse_optimized:.3f} (improvement: {results[best_model_name]['rmse'] - rmse_optimized:.3f})")
    print(f"  R²: {r2_optimized:.3f} (improvement: {r2_optimized - results[best_model_name]['r2']:.3f})")
    print(f"  MAE: {mae_optimized:.3f}")
    
    # Plot optimized model results
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred_optimized, alpha=0.7)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--')
    plt.xlabel('Actual pIC50')
    plt.ylabel('Predicted pIC50')
    plt.title(f'Subcluster {subcluster_id} - Optimized {best_model_name.upper()} Model\n'
             f'R² = {r2_optimized:.3f}, RMSE = {rmse_optimized:.3f}')
    plt.tight_layout()
    plt.savefig(f"models/subcluster_models/subcluster_{subcluster_id}_optimized_performance.png")
    plt.close()
    
    # Save optimized model
    with open(f"models/subcluster_models/subcluster_{subcluster_id}_optimized_model.pkl", 'wb') as f:
        pickle.dump(best_model, f)
    
    # Feature importance analysis
    if hasattr(best_model, 'feature_importances_'):
        # Get feature importance
        importances = best_model.feature_importances_
        
        # Sort features by importance
        indices = np.argsort(importances)[-20:]  # Top 20 features
        
        # Plot feature importances
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(indices)), importances[indices], align='center')
        plt.yticks(range(len(indices)), [feature_cols[i] for i in indices])
        plt.title('Top 20 Most Important Features')
        plt.tight_layout()
        plt.savefig(f"models/subcluster_models/subcluster_{subcluster_id}_feature_importance.png")
        plt.close()
    
    # Save a summary of results
    summary = {
        'subcluster_id': subcluster_id,
        'min_pic50': min_pic50,
        'max_pic50': max_pic50,
        'n_compounds': len(filtered_df),
        'train_size': X_train.shape[0],
        'test_size': X_test.shape[0],
        'rf_r2': results['rf']['r2'],
        'rf_rmse': results['rf']['rmse'],
        'xgb_r2': results['xgb']['r2'],
        'xgb_rmse': results['xgb']['rmse'],
        'best_model': best_model_name,
        'optimized_r2': r2_optimized,
        'optimized_rmse': rmse_optimized,
        'best_params': grid_search.best_params_
    }
    
    # Save as CSV
    pd.DataFrame([summary]).to_csv(
        f"models/subcluster_models/subcluster_{subcluster_id}_summary.csv", index=False
    )
    
    print("\nModel building completed. Results saved to models/subcluster_models/")
    
    return best_model, scaler, (r2_optimized, rmse_optimized)

if __name__ == "__main__":
    # Path to subcluster data file
    subcluster_file = "data/subclusters/cluster_1_subclustered.csv"
    
    # Build model for subcluster 0 with pIC50 between 5 and 10
    print("=" * 50)
    print("BUILDING MODEL FOR SUBCLUSTER 0 (pIC50: 5-10)")
    print("=" * 50)
    model, scaler, metrics = build_subcluster_model(
        subcluster_file, subcluster_id=0, min_pic50=5.0, max_pic50=10.0
    )
    
    print(f"\nFinal model performance: R² = {metrics[0]:.3f}, RMSE = {metrics[1]:.3f}")
    
    """
    This script builds a QSAR regression model specifically for compounds in Subcluster 0
    with pIC50 values between 5 and 10. The process includes:
    
    Filters the Data: Extracts only compounds from Subcluster 0 with pIC50 values between 5 and 10
    Prepares Features: Uses Morgan fingerprints as molecular descriptors
    Trains Multiple Models:
    Builds both RandomForest and XGBoost models with default parameters
    Compares their performance
    Hyperparameter Optimization: Performs grid search to find the best parameters for the better-performing model
    Visualizations:
    Creates scatter plots of predicted vs. actual values
    Generates feature importance plots
    Model Evaluation: Calculates RMSE, R², and MAE on the test set
    Saves Results: Stores models, performance metrics, and visualizations
    
    Expected Benefits:
    Higher Accuracy: By focusing on a specific subcluster and activity range, the model should perform better than the global model
    More Reliable Predictions: The focused approach accounts for structure-activity patterns specific to that chemical subclass
    Actionable Insights: Feature importance analysis reveals which molecular features drive activity within this subcluster
    Practical Application Range: The 5-10 pIC50 range focuses on compounds with meaningful activity
    
    """
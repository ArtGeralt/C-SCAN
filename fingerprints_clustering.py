import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
import os
from datetime import datetime

timestamp = ""

def cluster_fingerprints(input_csv, n_clusters=5, random_state=42, plot_results=True):
    """
    Perform k-means clustering on molecular fingerprints to identify chemical clusters
    """
    print(f"Loading fingerprint data from {input_csv}")
    df = pd.read_csv(input_csv)
    
    # Get only the fingerprint columns
    exclude_cols = ['molecule_chembl_id', 'canonical_smiles', 'pIC50']
    fingerprint_cols = [col for col in df.columns if col not in exclude_cols]
    X = df[fingerprint_cols].values
    
    print(f"Clustering {len(df)} compounds with {len(fingerprint_cols)} fingerprint bits")
    
    # Standardize features for better clustering
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply k-means clustering
    print(f"Performing k-means clustering with {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Add cluster assignments to the dataframe
    df_clustered = df.copy()
    df_clustered['cluster'] = clusters
    
    # Analyze clusters
    print("\nCluster distribution:")
    cluster_counts = df_clustered['cluster'].value_counts().sort_index()
    for cluster_id, count in cluster_counts.items():
        print(f"Cluster {cluster_id}: {count} compounds")
    
    # Analyze pIC50 distribution across clusters
    print("\npIC50 distribution by cluster:")
    cluster_stats = df_clustered.groupby('cluster')['pIC50'].agg(['mean', 'std', 'min', 'max'])
    print(cluster_stats)
    
    # Create output directory with timestamp
    output_dir = f"data/processed/{timestamp}/clusters"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save clustered data
    output_csv = f"{output_dir}/kit_fingerprints_clustered.csv"
    df_clustered.to_csv(output_csv, index=False)
    print(f"Saved clustered data to {output_csv}")
    
    if plot_results:
        # Dimensionality reduction for visualization
        print("Performing dimensionality reduction for visualization...")
        # First reduce to 50 dimensions with PCA (for speed)
        pca = PCA(n_components=50)
        X_pca = pca.fit_transform(X_scaled)
        
        # Then reduce to 2D with t-SNE
        tsne = TSNE(n_components=2, random_state=random_state)
        X_tsne = tsne.fit_transform(X_pca)
        
        # Create visualization dataframe
        vis_df = pd.DataFrame({
            'TSNE1': X_tsne[:, 0],
            'TSNE2': X_tsne[:, 1],
            'cluster': clusters,
            'pIC50': df['pIC50']
        })
        
        # Plot clusters
        plt.figure(figsize=(12, 10))
        
        # Plot 1: Clusters
        plt.subplot(1, 2, 1)
        sns.scatterplot(
            x='TSNE1', y='TSNE2', 
            hue='cluster', 
            palette='viridis',
            data=vis_df
        )
        plt.title('Chemical Space Clustering')
        plt.legend(title='Cluster')
        
        # Plot 2: pIC50 values
        plt.subplot(1, 2, 2)
        scatter = sns.scatterplot(
            x='TSNE1', y='TSNE2', 
            hue='pIC50', 
            palette='coolwarm',
            data=vis_df
        )
        plt.title('pIC50 Distribution')
        plt.colorbar(scatter.collections[0], label='pIC50')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/fingerprint_clusters.png")
        print(f"Visualization saved to {output_dir}/fingerprint_clusters.png")
        
        # Generate cluster representatives
        visualize_cluster_representatives(df_clustered, timestamp, n_per_cluster=3)
    
    return df_clustered, kmeans

def visualize_cluster_representatives(df_clustered, timestamp, n_per_cluster=3):
    """
    Generate representative molecule images for each cluster
    """
    # Create output directory for molecule images
    output_dir = f"data/processed/{timestamp}/clusters/representatives"
    os.makedirs(output_dir, exist_ok=True)
    
    clusters = df_clustered['cluster'].unique()
    
    for cluster_id in clusters:
        # Get compounds from this cluster
        cluster_df = df_clustered[df_clustered['cluster'] == cluster_id]
        
        # Sort by pIC50 value (most active first)
        cluster_df = cluster_df.sort_values('pIC50', ascending=False)
        
        # Take top n compounds as representatives
        representatives = cluster_df.head(n_per_cluster)
        
        # Generate molecule images
        mols = []
        legends = []
        for _, row in representatives.iterrows():
            smiles = row['canonical_smiles']
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                mols.append(mol)
                legends.append(f"pIC50: {row['pIC50']:.2f}")
        
        if mols:
            # Create grid image
            img = Draw.MolsToGridImage(mols, molsPerRow=3, subImgSize=(200, 200), legends=legends)
            img.save(f"{output_dir}/cluster_{cluster_id}.png")
    
    print(f"Generated representative molecule images for {len(clusters)} clusters")

def train_cluster_specific_models(df_clustered, timestamp, model_type='rf', test_size=0.2, random_state=42):
    """
    Train separate models for each cluster
    """
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from xgboost import XGBRegressor
    from sklearn.metrics import r2_score, mean_squared_error
    
    # Create output directory
    output_dir = f"data/processed/{timestamp}/models/cluster_models"
    os.makedirs(output_dir, exist_ok=True)
    
    # Get unique clusters
    clusters = sorted(df_clustered['cluster'].unique())
    
    # Results storage
    results = []
    
    # Train a model for each cluster with sufficient data
    for cluster_id in clusters:
        # Get data for this cluster
        cluster_data = df_clustered[df_clustered['cluster'] == cluster_id]
        
        # Skip if insufficient data
        if len(cluster_data) < 20:  # Minimum size for meaningful split
            print(f"Cluster {cluster_id} has only {len(cluster_data)} compounds - skipping model training")
            continue
        
        print(f"\nTraining model for Cluster {cluster_id} with {len(cluster_data)} compounds")
        
        # Prepare data
        exclude_cols = ['molecule_chembl_id', 'canonical_smiles', 'pIC50', 'cluster']
        X = cluster_data.drop(columns=exclude_cols).values
        y = cluster_data['pIC50'].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Standardize
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Select model
        if model_type == 'xgb':
            model = XGBRegressor(n_estimators=100, random_state=random_state)
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=random_state)
        
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test_scaled)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        print(f"Cluster {cluster_id} model - R²: {r2:.3f}, RMSE: {rmse:.3f}")
        
        # Plot predictions
        plt.figure(figsize=(6, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--')
        plt.xlabel('Actual pIC50')
        plt.ylabel('Predicted pIC50')
        plt.title(f'Cluster {cluster_id} (R²={r2:.3f}, RMSE={rmse:.3f})')
        plt.savefig(f"{output_dir}/cluster_{cluster_id}_performance.png")
        plt.close()
        
        # Store results
        results.append({
            'cluster': cluster_id,
            'n_compounds': len(cluster_data),
            'r2': r2,
            'rmse': rmse
        })
    
    # Create results summary
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        print("\nCluster-specific model performance:")
        print(results_df)
        results_df.to_csv(f"{output_dir}/performance_summary.csv", index=False)
    
    return results_df

if __name__ == "__main__":
    # Path to fingerprint data file with timestamp
    input_file = f"data/processed/{timestamp}/kit_fingerprints.csv"
    
    # Perform clustering
    df_clustered, kmeans = cluster_fingerprints(input_file, n_clusters=5)
    
    # Train cluster-specific models
    print("\n" + "="*50)
    print("TRAINING CLUSTER-SPECIFIC MODELS")
    print("="*50)
    results = train_cluster_specific_models(df_clustered, timestamp, model_type='rf')
    
"""
Main Functions
1. Chemical Clustering
    Takes molecular fingerprints (likely Morgan/ECFP) generated previously
    Uses k-means clustering to group compounds into 5 chemical clusters
    Assigns each molecule to its respective cluster
    
2. Cluster Visualization
    Creates a 2D chemical space map using dimensionality reduction (PCA + t-SNE)
    Visualizes clusters with color coding in one plot
    Shows activity distribution (pIC50) in the chemical space in a second plot
    
3. Cluster Analysis
    Reports cluster size distribution (number of compounds per cluster)
    Calculates activity statistics (mean, std, min, max pIC50) for each cluster
    Identifies which clusters contain the most active compounds
    
4. Representative Molecule Visualization
    For each cluster, selects the 3 most active compounds
    Generates chemical structure images of these representatives
    Provides visual examples of what compounds in each cluster look like
    
5. Cluster-Specific QSAR Models
    Trains separate Random Forest regression models for each cluster
    Only processes clusters with at least 20 compounds
    Evaluates model performance using R² and RMSE metrics
    Visualizes actual vs. predicted pIC50 for each cluster model
    
Workflow Steps

    1. Reads fingerprint data from your timestamp folder
    2. Applies k-means clustering to group similar compounds
    3. Saves clustered data to CSV for future use
    4. Generates visualizations of chemical space and clusters
    5. Creates representative molecule images for each cluster
    6. Builds and evaluates cluster-specific QSAR models
    7. Saves performance metrics and visualizations
    
Purpose
    This script implements the "chemical space analysis" approach to QSAR modeling, which:

    1. Groups similar compounds based on structural features
    2. Identifies if activity patterns differ between chemical classes
    3. Builds specialized models that may better predict within chemical families
    4. Provides insights into which structural classes are most promising for KIT inhibition

"""
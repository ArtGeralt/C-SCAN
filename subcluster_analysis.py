import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from rdkit import Chem
from rdkit.Chem import Draw, AllChem, PandasTools, Descriptors, MACCSkeys
from rdkit.Chem.Scaffolds import MurckoScaffold
import os
from datetime import datetime
from sklearn.metrics import silhouette_score

def determine_optimal_subclusters(clustered_data_file, target_cluster=1, max_clusters=10):
    """Determine optimal number of subclusters using multiple methods"""
    print(f"Loading clustered data from {clustered_data_file}")
    df = pd.read_csv(clustered_data_file)
    
    # Extract the target cluster
    cluster_df = df[df['cluster'] == target_cluster].copy()
    print(f"Analyzing Cluster {target_cluster} with {len(cluster_df)} compounds")
    
    # Get fingerprint columns
    exclude_cols = ['molecule_chembl_id', 'canonical_smiles', 'pIC50', 'cluster']
    fingerprint_cols = [col for col in cluster_df.columns if col not in exclude_cols]
    
    # Extract fingerprints
    X = cluster_df[fingerprint_cols].values
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d")
    output_dir = f"data/processed/{timestamp}/subclusters/optimization"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Elbow Method
    print("\nPerforming elbow method analysis...")
    inertia = []
    
    for k in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertia.append(kmeans.inertia_)
    
    # 2. Silhouette Analysis
    print("\nPerforming silhouette analysis...")
    silhouette_scores = []
    
    for k in range(2, max_clusters + 1):  # Silhouette requires at least 2 clusters
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)
        silhouette_avg = silhouette_score(X_scaled, cluster_labels)
        silhouette_scores.append(silhouette_avg)
        print(f"For n_clusters = {k}, the silhouette score is {silhouette_avg:.3f}")
    
    # 3. Practical recommendation
    n_compounds = len(cluster_df)
    
    if n_compounds < 60:
        recommended = 2
    elif n_compounds < 150:
        recommended = 3
    elif n_compounds < 300:
        recommended = 4
    elif n_compounds < 600:
        recommended = 5
    else:
        recommended = 6
    
    # Find elbow point (approximate)
    elbow_detected = 0
    for i in range(1, len(inertia)-1):
        prev_slope = inertia[i-1] - inertia[i]
        next_slope = inertia[i] - inertia[i+1]
        if prev_slope > 2 * next_slope:
            elbow_detected = i + 1
            break
    
    # Find best silhouette score
    best_silhouette = np.argmax(silhouette_scores) + 2
    
    print("\n" + "="*60)
    print("CLUSTER ANALYSIS SUMMARY")
    print("="*60)
    print(f"Dataset size: {n_compounds} compounds")
    print(f"Elbow method suggests: {elbow_detected} subclusters")
    print(f"Best silhouette score at: {best_silhouette} subclusters")
    print(f"Rule of thumb recommendation: {recommended} subclusters")
    print("="*60)
    
    return recommended

def subcluster_analysis(timestamp, target_cluster, n_subclusters):
    """Perform subclustering on a specific cluster and analyze chemical properties"""
    # Define input path with timestamp
    clustered_data_file = f"data/processed/{timestamp}/clusters/kit_fingerprints_clustered.csv"
    
    if not os.path.exists(clustered_data_file):
        print(f"Error: Clustered data file not found at {clustered_data_file}")
        print("Please ensure the clustering analysis has been run first.")
        return None
    
    print(f"Loading clustered data from {clustered_data_file}")
    df = pd.read_csv(clustered_data_file)
    
    # Extract the target cluster
    cluster_df = df[df['cluster'] == target_cluster].copy()
    print(f"Analyzing Cluster {target_cluster} with {len(cluster_df)} compounds")
    
    if len(cluster_df) == 0:
        print(f"Error: No compounds found in Cluster {target_cluster}")
        print(f"Available clusters: {sorted(df['cluster'].unique())}")
        return None
    
    # Get fingerprint columns
    exclude_cols = ['molecule_chembl_id', 'canonical_smiles', 'pIC50', 'cluster']
    fingerprint_cols = [col for col in cluster_df.columns if col not in exclude_cols]
    
    # Extract fingerprints
    X = cluster_df[fingerprint_cols].values
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform subclustering
    print(f"Performing subclustering with {n_subclusters} subclusters...")
    kmeans = KMeans(n_clusters=n_subclusters, random_state=42, n_init=10)
    subclusters = kmeans.fit_predict(X_scaled)
    
    # Add subcluster assignments
    cluster_df['subcluster'] = subclusters
    
    # Analyze subcluster distribution
    print("\nSubcluster distribution:")
    subcluster_counts = cluster_df['subcluster'].value_counts().sort_index()
    for subcluster_id, count in subcluster_counts.items():
        print(f"Subcluster {subcluster_id}: {count} compounds")
    
    # Analyze pIC50 distribution across subclusters
    print("\npIC50 distribution by subcluster:")
    subcluster_stats = cluster_df.groupby('subcluster')['pIC50'].agg(['count', 'mean', 'std', 'min', 'max'])
    print(subcluster_stats)
    
    # Create output directory with timestamp
    output_dir = f"data/processed/{timestamp}/subclusters"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save subcluster data
    output_csv = f"{output_dir}/cluster_{target_cluster}_subclustered.csv"
    cluster_df.to_csv(output_csv, index=False)
    print(f"Saved subcluster data to {output_csv}")
    
    # Visualize subclusters
    visualize_subclusters(cluster_df, fingerprint_cols, target_cluster, timestamp)
    
    # Find top molecules in each subcluster
    find_top_molecules(cluster_df, target_cluster, timestamp)
    
    # Extract and analyze common scaffolds
    analyze_scaffolds(cluster_df, target_cluster, timestamp)
    
    return cluster_df

def visualize_subclusters(cluster_df, fingerprint_cols, target_cluster, timestamp):
    """Visualize subclusters using dimensionality reduction"""
    X = cluster_df[fingerprint_cols].values
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # First reduce to 50 dimensions with PCA (for speed)
    print("Performing dimensionality reduction for visualization...")
    pca = PCA(n_components=min(50, X_scaled.shape[1]))
    X_pca = pca.fit_transform(X_scaled)
    
    # Then reduce to 2D with t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X_pca)
    
    # Create visualization dataframe
    vis_df = pd.DataFrame({
        'TSNE1': X_tsne[:, 0],
        'TSNE2': X_tsne[:, 1],
        'subcluster': cluster_df['subcluster'],
        'pIC50': cluster_df['pIC50']
    })
    
    # Plot subclusters
    plt.figure(figsize=(16, 7))
    
    # Plot 1: Subclusters
    plt.subplot(1, 2, 1)
    sns.scatterplot(
        x='TSNE1', y='TSNE2', 
        hue='subcluster', 
        palette='viridis',
        data=vis_df
    )
    plt.title(f'Subclusters within Cluster {target_cluster}')
    plt.legend(title='Subcluster')
    
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
    
    # Save to timestamp directory
    output_dir = f"data/processed/{timestamp}/subclusters"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/cluster_{target_cluster}_subclusters.png")
    print(f"Visualization saved to {output_dir}/cluster_{target_cluster}_subclusters.png")

def find_top_molecules(cluster_df, target_cluster, timestamp, top_n=20):
    """Find and visualize top molecules by pIC50 in each subcluster"""
    subclusters = sorted(cluster_df['subcluster'].unique())
    
    # Create output directories with timestamp
    output_dir = f"data/processed/{timestamp}/subclusters/top_molecules"
    os.makedirs(output_dir, exist_ok=True)
    
    # Find overall top molecules
    top_overall = cluster_df.sort_values('pIC50', ascending=False).head(top_n)
    print(f"\nTop {top_n} molecules overall by pIC50:")
    print(top_overall[['molecule_chembl_id', 'pIC50', 'subcluster']].to_string(index=False))
    
    # Save top molecules to CSV
    top_overall.to_csv(f"{output_dir}/cluster_{target_cluster}_top_{top_n}.csv", index=False)
    
    # Generate image with top molecule structures
    mols = []
    legends = []
    for _, row in top_overall.iterrows():
        mol = Chem.MolFromSmiles(row['canonical_smiles'])
        if mol:
            mols.append(mol)
            legends.append(f"pIC50: {row['pIC50']:.2f} | SC: {row['subcluster']}")
    
    if mols:
        img = Draw.MolsToGridImage(mols, molsPerRow=4, subImgSize=(250, 250), legends=legends)
        img.save(f"{output_dir}/cluster_{target_cluster}_top_{top_n}.png")
    
    # Find top molecules by subcluster
    for subcluster_id in subclusters:
        subcluster_df = cluster_df[cluster_df['subcluster'] == subcluster_id]
        top_in_subcluster = subcluster_df.sort_values('pIC50', ascending=False).head(top_n)
        
        # Save subcluster top molecules
        top_in_subcluster.to_csv(
            f"{output_dir}/cluster_{target_cluster}_subcluster_{subcluster_id}_top_{top_n}.csv", 
            index=False
        )
        
        # Generate images
        mols = []
        legends = []
        for _, row in top_in_subcluster.iterrows():
            mol = Chem.MolFromSmiles(row['canonical_smiles'])
            if mol:
                mols.append(mol)
                legends.append(f"pIC50: {row['pIC50']:.2f}")
        
        if mols:
            img = Draw.MolsToGridImage(mols, molsPerRow=4, subImgSize=(200, 200), legends=legends)
            img.save(f"{output_dir}/cluster_{target_cluster}_subcluster_{subcluster_id}_top_{top_n}.png")

def analyze_scaffolds(cluster_df, target_cluster, timestamp):
    """Extract and analyze common scaffolds in each subcluster"""
    # Create output directory with timestamp
    output_dir = f"data/processed/{timestamp}/subclusters/scaffolds"
    os.makedirs(output_dir, exist_ok=True)
    
    # Add Murcko scaffolds to the dataframe
    cluster_df['scaffold_smiles'] = cluster_df['canonical_smiles'].apply(
        lambda x: MurckoScaffold.MurckoScaffoldSmiles(
            smiles=x, includeChirality=False
        ) if Chem.MolFromSmiles(x) else ''
    )
    
    # Find common scaffolds overall
    scaffold_counts = cluster_df['scaffold_smiles'].value_counts().head(10)
    print(f"\nMost common scaffolds in Cluster {target_cluster}:")
    for scaffold, count in scaffold_counts.items():
        if scaffold and scaffold != '':
            print(f"- {scaffold}: {count} compounds")
    
    # Create scaffold summary by subcluster
    subclusters = sorted(cluster_df['subcluster'].unique())
    scaffold_summary = []
    
    for subcluster_id in subclusters:
        subcluster_df = cluster_df[cluster_df['subcluster'] == subcluster_id]
        scaffold_counts = subcluster_df['scaffold_smiles'].value_counts().head(5)
        
        # Draw the top scaffolds for this subcluster
        mols = []
        legends = []
        for scaffold, count in scaffold_counts.items():
            if scaffold and scaffold != '':
                mol = Chem.MolFromSmiles(scaffold)
                if mol:
                    mols.append(mol)
                    legends.append(f"Count: {count}")
                    
                    # Add to summary
                    scaffold_summary.append({
                        'subcluster': subcluster_id,
                        'scaffold': scaffold,
                        'count': count,
                        'percent': 100 * count / len(subcluster_df)
                    })
        
        if mols:
            img = Draw.MolsToGridImage(mols, molsPerRow=3, subImgSize=(250, 250), legends=legends)
            img.save(f"{output_dir}/cluster_{target_cluster}_subcluster_{subcluster_id}_scaffolds.png")
    
    # Create a summary dataframe and save it
    scaffold_df = pd.DataFrame(scaffold_summary)
    scaffold_df.to_csv(f"{output_dir}/scaffold_summary.csv", index=False)
    print(f"Scaffold analysis completed and saved to {output_dir}")

def main():
    """Main function to run the subclustering analysis"""
    # ===== EDIT THESE VALUES =====
    # Set the timestamp folder to use (format: YYYYMMDD)
    timestamp = "20251006"  # EDIT THIS to match your data folder timestamp
    
    # Set the cluster ID you want to analyze
    target_cluster = 3      # EDIT THIS to change which cluster to analyze
    
    # Set number of subclusters
    # Set to None to auto-determine the optimal number
    n_subclusters = 6       # EDIT THIS or set to None for automatic detection
    # ============================
    
    print("\n" + "="*60)
    print(f"SUBCLUSTER ANALYSIS (Timestamp: {timestamp})")
    print("="*60)
    
    # Check that required directories exist
    clustered_data_path = f"data/processed/{timestamp}/clusters/kit_fingerprints_clustered.csv"
    if not os.path.exists(clustered_data_path):
        print(f"Error: Could not find clustered data at {clustered_data_path}")
        print("Please run fingerprints_clustering.py first.")
        return
    
    # Auto-determine the number of subclusters if not specified
    if n_subclusters is None:
        print("\n" + "="*60)
        print(f"DETERMINING OPTIMAL NUMBER OF SUBCLUSTERS FOR CLUSTER {target_cluster}")
        print("="*60)
        n_subclusters = determine_optimal_subclusters(clustered_data_path, target_cluster)
        print(f"\nUsing recommended number of subclusters: {n_subclusters}")
    
    # Perform the subclustering analysis
    print("\n" + "="*60)
    print(f"SUBCLUSTERING ANALYSIS OF CLUSTER {target_cluster} WITH {n_subclusters} SUBCLUSTERS")
    print("="*60)
    
    # Run the analysis
    subcluster_df = subcluster_analysis(timestamp, target_cluster, n_subclusters)
    
    if subcluster_df is not None:
        print("\n" + "="*60)
        print("SUBCLUSTERING ANALYSIS COMPLETE")
        print("="*60)
        print(f"Results saved to data/processed/{timestamp}/subclusters/")

if __name__ == "__main__":
    main()
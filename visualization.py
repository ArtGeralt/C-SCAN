import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from datetime import datetime

timestamp = ""

def plot_distribution(data_path, output_dir=None):
    """Plot the distribution of pIC50 values"""
    if output_dir is None:
        output_dir = f"data/processed/{timestamp}/visualizations"
    
    os.makedirs(output_dir, exist_ok=True)
    
    df = pd.read_csv(data_path)
    
    # Basic statistics
    print(f"Dataset size: {len(df)} compounds")
    print(f"pIC50 range: {df['pIC50'].min():.2f} - {df['pIC50'].max():.2f}")
    print(f"pIC50 mean ± std: {df['pIC50'].mean():.2f} ± {df['pIC50'].std():.2f}")

    # Visualize distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['pIC50'], bins=20, kde=True)
    plt.title("Distribution of KIT Inhibitor pIC50 Values")
    plt.xlabel("pIC50")
    plt.ylabel("Count")
    
    # Add vertical lines for activity thresholds
    plt.axvline(x=6, color='orange', linestyle='--', alpha=0.7, label='Moderate activity (6)')
    plt.axvline(x=7, color='green', linestyle='--', alpha=0.7, label='High activity (7)')
    plt.axvline(x=8, color='red', linestyle='--', alpha=0.7, label='Very high activity (8)')
    plt.legend()
    
    # Save plot
    output_path = os.path.join(output_dir, "pIC50_distribution.png")
    plt.savefig(output_path)
    print(f"Saved distribution plot to {output_path}")
    plt.close()
    
    # Add activity classification summary
    low = sum(df['pIC50'] < 6)
    moderate = sum((df['pIC50'] >= 6) & (df['pIC50'] < 7))
    high = sum((df['pIC50'] >= 7) & (df['pIC50'] < 8))
    very_high = sum(df['pIC50'] >= 8)
    
    print("\nActivity distribution:")
    print(f"Low activity (pIC50 < 6): {low} compounds ({100*low/len(df):.1f}%)")
    print(f"Moderate activity (6 ≤ pIC50 < 7): {moderate} compounds ({100*moderate/len(df):.1f}%)")
    print(f"High activity (7 ≤ pIC50 < 8): {high} compounds ({100*high/len(df):.1f}%)")
    print(f"Very high activity (pIC50 ≥ 8): {very_high} compounds ({100*very_high/len(df):.1f}%)")

def plot_model_performance(y_true, y_pred, output_dir=None):
    """Plot predicted vs actual values"""
    if output_dir is None:
        output_dir = f"data/processed/{timestamp}/models"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate metrics
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    r2 = 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2)
    
    # Create plot
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'k--')
    plt.xlabel('Actual pIC50')
    plt.ylabel('Predicted pIC50')
    plt.title(f'Model Performance (R²={r2:.3f}, RMSE={rmse:.3f})')
    
    # Save plot
    output_path = os.path.join(output_dir, "model_performance.png")
    plt.savefig(output_path)
    print(f"Saved performance plot to {output_path}")
    plt.close()

def plot_property_distributions(data_path, output_dir=None):
    """Plot distributions of key molecular properties"""
    if output_dir is None:
        output_dir = f"data/processed/{timestamp}/visualizations"
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Print debug info
    print(f"Saving property distributions to {output_dir}")
    
    # Load data
    if isinstance(data_path, pd.DataFrame):
        df = data_path
    else:
        df = pd.read_csv(data_path)
    
    # Check if we have molecular descriptors
    exclude_cols = ['molecule_chembl_id', 'canonical_smiles', 'pIC50', 'cluster', 'subcluster']
    descriptor_cols = [col for col in df.columns if col not in exclude_cols]
    
    # If no descriptors, calculate some basic ones using RDKit
    if len(descriptor_cols) < 5:
        print("Calculating basic molecular descriptors...")
        from rdkit import Chem
        from rdkit.Chem import Descriptors, Lipinski
        
        # Calculate basic properties
        molecules = [Chem.MolFromSmiles(s) for s in df['canonical_smiles']]
        valid_mols = [m for m in molecules if m is not None]
        
        # Calculate properties
        prop_data = []
        for m in valid_mols:
            props = {
                'MW': Descriptors.MolWt(m),
                'LogP': Descriptors.MolLogP(m),
                'HBD': Descriptors.NumHDonors(m),
                'HBA': Descriptors.NumHAcceptors(m),
                'TPSA': Descriptors.TPSA(m),
                'RotBonds': Descriptors.NumRotatableBonds(m),
                'ArRings': Lipinski.NumAromaticRings(m)  # Using the correct Lipinski method
            }
            prop_data.append(props)
        
        # Convert to DataFrame columns
        prop_df = pd.DataFrame(prop_data)
        
        # Join with original DataFrame (keep only valid molecules)
        df = df.iloc[:len(valid_mols)].reset_index(drop=True)
        df = pd.concat([df, prop_df], axis=1)
        
        descriptor_cols = ['MW', 'LogP', 'HBD', 'HBA', 'TPSA', 'RotBonds', 'ArRings']
    
    # Plot distributions of individual properties
    for i, prop in enumerate(descriptor_cols[:12]):  # Limit to 12 properties for readability
        if prop in df.columns and (df[prop].dtypes == np.float64 or df[prop].dtypes == np.int64):
            plt.figure(figsize=(10, 6))
            sns.histplot(df[prop], kde=True)
            plt.title(f"Distribution of {prop}")
            plt.xlabel(prop)
            plt.ylabel("Count")
            plt.savefig(os.path.join(output_dir, f"{prop}_distribution.png"))
            plt.close()
            print(f"Saved {prop} distribution plot")
    
    # IMPORTANT: Create the property summary grid (2x2) - This creates property_distributions.png
    # First, identify key properties to visualize
    default_key_props = ['MW', 'LogP', 'TPSA', 'RotBonds']
    alt_key_props = ['MolWt', 'MolLogP', 'TPSA', 'NumRotatableBonds']
    
    # Try default names first, then alternatives, then whatever is available
    available_props = [p for p in default_key_props if p in df.columns]
    if len(available_props) < 4:
        available_props = [p for p in alt_key_props if p in df.columns]
    if len(available_props) < 4:
        # Just take the first four numeric descriptors
        available_props = [col for col in descriptor_cols if col in df.columns 
                          and (df[col].dtypes == np.float64 or df[col].dtypes == np.int64)][:4]
    
    # Create the summary grid (THIS FILE MUST BE CREATED FOR APP.PY)
    if len(available_props) >= 1:  # Even if we don't have 4 properties, create the plot with what we have
        n_plots = min(len(available_props), 4)
        n_cols = min(n_plots, 2)
        n_rows = (n_plots + 1) // 2  # Round up to get number of rows needed
        
        plt.figure(figsize=(15, 12))
        for i, prop in enumerate(available_props[:4]):
            if i < n_plots:  # Only create as many subplots as we have properties
                plt.subplot(n_rows, n_cols, i+1)
                if df[prop].dtypes == np.float64 or df[prop].dtypes == np.int64:
                    sns.histplot(df[prop], kde=True)
                    plt.title(f"Distribution of {prop}")
                    plt.xlabel(prop)
        
        plt.tight_layout()
        property_distributions_path = os.path.join(output_dir, "property_distributions.png")
        plt.savefig(property_distributions_path)
        plt.close()
        print(f"Saved property summary grid to {property_distributions_path}")
    else:
        print("WARNING: Could not create property_distributions.png - no suitable properties found")
        # Create a placeholder image to avoid errors in the app
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, "No property data available", 
                 ha='center', va='center', fontsize=14)
        plt.savefig(os.path.join(output_dir, "property_distributions.png"))
        plt.close()
    
    # Generate Lipinski Rule of 5 compliance report if we have the needed properties
    lipinski_props_sets = [
        ['MW', 'LogP', 'HBD', 'HBA'],
        ['MolWt', 'MolLogP', 'NumHDonors', 'NumHAcceptors']
    ]
    
    lipinski_available = False
    for prop_set in lipinski_props_sets:
        if all(p in df.columns for p in prop_set):
            # Use the first available set
            mw_col, logp_col, hbd_col, hba_col = prop_set
            lipinski_available = True
            break
    
    if lipinski_available:
        ro5_violations = pd.DataFrame({
            'MW_over_500': df[mw_col] > 500,
            'LogP_over_5': df[logp_col] > 5,
            'HBD_over_5': df[hbd_col] > 5,
            'HBA_over_10': df[hba_col] > 10
        })
        
        # Count violations per compound
        df['RO5_violations'] = ro5_violations.sum(axis=1)
        
        # Plot distribution of RO5 violations
        plt.figure(figsize=(8, 6))
        violation_counts = df['RO5_violations'].value_counts().sort_index()
        plt.bar(violation_counts.index, violation_counts.values)
        plt.xlabel('Number of Lipinski Rule of 5 Violations')
        plt.ylabel('Count')
        plt.title('Lipinski Rule of 5 Compliance')
        plt.xticks(range(5))
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add count labels
        for i, v in enumerate(violation_counts.values):
            plt.text(violation_counts.index[i], v + 0.1, str(int(v)), ha='center')
            
        plt.savefig(os.path.join(output_dir, "lipinski_violations.png"))
        plt.close()
        print(f"Saved Lipinski violations plot")
        
        print("\nLipinski Rule of 5 compliance:")
        for i in range(5):
            count = sum(df['RO5_violations'] == i)
            print(f"Compounds with {i} violations: {count} ({100*count/len(df):.1f}%)")
    
    print(f"Completed property distribution analysis. All plots saved to {output_dir}")
    
    return df  # Return DataFrame in case properties were calculated

def plot_activity_vs_properties(data_path, output_dir=None):
    """Plot relationships between molecular properties and activity"""
    if output_dir is None:
        output_dir = f"data/processed/{timestamp}/visualizations"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data - use the DataFrame returned by plot_property_distributions if available
    if isinstance(data_path, pd.DataFrame):
        df = data_path
    else:
        df = pd.read_csv(data_path)
    
    # Check for descriptor columns
    exclude_cols = ['molecule_chembl_id', 'canonical_smiles', 'pIC50', 'cluster', 'subcluster', 'RO5_violations']
    descriptor_cols = [col for col in df.columns if col not in exclude_cols]
    
    # If no descriptors, calculate basic ones - should be handled by plot_property_distributions
    if len(descriptor_cols) < 5:
        print("Insufficient descriptor columns for correlation analysis")
        return

    # Select key properties to analyze
    key_props = ['MW', 'LogP', 'TPSA', 'RotBonds', 'HBD', 'HBA'] 
    available_props = [p for p in key_props if p in df.columns]
    
    if not available_props:
        available_props = descriptor_cols[:6]  # Take first 6 descriptors if key ones not found
    
    # Calculate property correlations with activity
    correlations = []
    for prop in available_props:
        if df[prop].dtypes == np.float64 or df[prop].dtypes == np.int64:
            corr = df[prop].corr(df['pIC50'])
            correlations.append({'Property': prop, 'Correlation': corr})
    
    # Sort by absolute correlation
    corr_df = pd.DataFrame(correlations)
    corr_df['AbsCorr'] = corr_df['Correlation'].abs()
    corr_df = corr_df.sort_values('AbsCorr', ascending=False)
    
    print("\nProperty correlations with pIC50:")
    for _, row in corr_df.iterrows():
        print(f"{row['Property']}: {row['Correlation']:.3f}")
    
    # Individual scatter plots for top correlating properties
    for prop in corr_df['Property'][:6]:  # Top 6 correlating properties
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=prop, y='pIC50', data=df, alpha=0.6)
        
        # Add trendline
        sns.regplot(x=prop, y='pIC50', data=df, scatter=False, ci=None, line_kws={'color': 'red'})
        
        # Add correlation in title
        corr = df[prop].corr(df['pIC50'])
        plt.title(f"pIC50 vs {prop} (r = {corr:.3f})")
        plt.savefig(os.path.join(output_dir, f"pIC50_vs_{prop}.png"))
        plt.close()
    
    # Create a correlation heatmap
    all_numeric_props = [col for col in descriptor_cols 
                    if col in df.columns and df[col].dtype in [np.float64, np.int64]][:15]
                    
    if len(all_numeric_props) == 0:
        print("No numeric properties found for correlation matrix!")
    else:
        print(f"Including {len(all_numeric_props)} properties in correlation matrix")

    # Always include pIC50
    corr_cols = all_numeric_props + ['pIC50']

    # 2. Adjust figure size based on number of properties for better readability
    matrix_size = len(corr_cols)
    plt.figure(figsize=(max(12, matrix_size*0.8), max(10, matrix_size*0.8)))

    # 3. Generate correlation matrix with all selected properties
    corr_matrix = df[corr_cols].corr()

    # 4. Create enhanced heatmap with better annotations and formatting
    sns.heatmap(
        corr_matrix, 
        annot=True, 
        cmap='coolwarm', 
        vmin=-1, 
        vmax=1,
        annot_kws={"size": max(7, 12 - matrix_size//4)},  # Adjust text size based on matrix size
        fmt='.2f',  # Show only 2 decimal places
        linewidths=0.5  # Add subtle gridlines
    )

    # 5. Enhance the title and axes
    plt.title('Molecular Property Correlation Matrix', fontsize=14, pad=10)

    # 6. Improve readability for large matrices
    if matrix_size > 12:
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)

    # 7. Save both standard and high-resolution versions
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"), dpi=100)
    plt.savefig(os.path.join(output_dir, "correlation_heatmap_hires.png"), dpi=300)
    plt.close()

    # 8. Also create a focused correlation matrix showing only the strongest correlations
    plt.figure(figsize=(10, 8))
    # Get absolute correlations with pIC50 and select top properties
    pic50_corrs = df[descriptor_cols].corrwith(df['pIC50']).abs().sort_values(ascending=False)
    top_corr_props = list(pic50_corrs.index[:10])  # Top 10 most correlated with activity
    focus_cols = top_corr_props + ['pIC50']
    focus_corr = df[focus_cols].corr()

    sns.heatmap(
        focus_corr, 
        annot=True, 
        cmap='coolwarm', 
        vmin=-1, 
        vmax=1,
        annot_kws={"size": 9}, 
        fmt='.2f',
        linewidths=0.5
    )
    plt.title('Top Activity-Correlated Properties', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "top_activity_correlations.png"))
    plt.close()
    
    # Save correlation data for future exploration rather than forcing a specific pairplot
    corr_df.to_csv(os.path.join(output_dir, "property_activity_correlations.csv"), index=False)
    print(f"Saved property_activity_correlations.csv to {os.path.join(output_dir, 'property_activity_correlations.csv')}")

    # Create a comprehensive property correlation dataset for exploration
    all_numeric_cols = [col for col in descriptor_cols if df[col].dtype in [np.float64, np.int64]]
    all_property_corrs = df[all_numeric_cols + ['pIC50']].corr()
    all_property_corrs.to_csv(os.path.join(output_dir, "all_property_correlations.csv"))
    print(f"Saved all_property_correlations.csv to {os.path.join(output_dir, 'all_property_correlations.csv')}")

    # Generate an example pairplot with the most significant properties
    # But focus on providing the underlying data for custom exploration
    top_props = list(corr_df['Property'][:5])  # Get top 5 properties
    if top_props:
        # Save the top correlating properties list for reference
        with open(os.path.join(output_dir, "top_correlating_properties.txt"), 'w') as f:
            f.write("Top properties correlated with pIC50:\n")
            for i, row in corr_df.iterrows():
                f.write(f"{row['Property']}: r = {row['Correlation']:.3f}\n")
        
        # Create a sample pairplot with top 5 properties, but only show 3 at a time for readability
        plt.figure(figsize=(12, 10))
        plt.text(0.5, 0.5, 
                "Top 5 properties correlated with activity:\n\n" + 
                "\n".join([f"{i+1}. {p} (r = {df[p].corr(df['pIC50']):.3f})" 
                        for i, p in enumerate(top_props)]),
                ha='center', va='center', fontsize=14)
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, "top_correlations_summary.png"))
        plt.close()
        
        # Create a sample pairplot with just top 3 properties for visualization purposes
        # This serves as an example but scientists can create their own with different properties
        sample_cols = top_props[:3] + ['pIC50']
        sns.pairplot(df[sample_cols], diag_kind='kde', corner=True,
                    plot_kws={'alpha': 0.6, 'edgecolor': 'none'})
        plt.suptitle("Sample Pairplot (Top 3 Properties)", y=1.02)
        plt.savefig(os.path.join(output_dir, "sample_pairplot.png"))
        plt.close()
        
        print(f"Saved property correlation data for custom exploration")

    print(f"Found {len(descriptor_cols)} descriptor columns: {descriptor_cols[:10]}...")
    print(f"Available properties for correlation: {available_props}")
    print(f"Correlation dataframe size: {len(corr_df)}")
    
def visualize_clusters(data_path, output_dir=None):
    """Visualize clustering of compounds"""
    if output_dir is None:
        output_dir = f"data/processed/{timestamp}/visualizations"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    if isinstance(data_path, pd.DataFrame):
        df = data_path
    else:
        df = pd.read_csv(data_path)
    
    # Check if we have cluster information
    if 'cluster' not in df.columns:
        print("No clustering information found in dataset")
        return
    
    # Count compounds per cluster
    cluster_counts = df['cluster'].value_counts().sort_index()
    
    # Create bar chart of cluster distribution
    plt.figure(figsize=(12, 6))
    bars = plt.bar(cluster_counts.index, cluster_counts.values)
    
    # Add count labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{int(height)}', ha='center', va='bottom')
    
    plt.xlabel('Cluster')
    plt.ylabel('Number of Compounds')
    plt.title('Distribution of KIT Inhibitors Across Clusters')
    plt.xticks(cluster_counts.index)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, "cluster_distribution.png"))
    plt.close()
    
    # Show pIC50 distribution per cluster
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='cluster', y='pIC50', data=df)
    plt.title('pIC50 Distribution by Cluster')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add cluster statistics
    cluster_stats = df.groupby('cluster')['pIC50'].agg(['mean', 'count'])
    for cluster, stats in cluster_stats.iterrows():
        plt.text(cluster, df['pIC50'].min() - 0.2, 
                f"n={int(stats['count'])}\nμ={stats['mean']:.2f}", 
                ha='center', va='top', fontsize=9)
    
    plt.savefig(os.path.join(output_dir, "pIC50_by_cluster.png"))
    plt.close()
    
    # Statistical analysis
    print("\nCluster pIC50 statistics:")
    cluster_full_stats = df.groupby('cluster')['pIC50'].agg(['count', 'mean', 'std', 'min', 'max'])
    print(cluster_full_stats)
    
    # If we have subcluster information, show that too
    if 'subcluster' in df.columns:
        # For each cluster, analyze its subclusters
        clusters = sorted(df['cluster'].unique())
        
        for cluster_id in clusters:
            cluster_df = df[df['cluster'] == cluster_id]
            
            if len(cluster_df) < 10:
                continue  # Skip very small clusters
            
            # Check if this cluster has subclusters
            if len(cluster_df['subcluster'].unique()) <= 1:
                continue
                
            plt.figure(figsize=(12, 6))
            sns.boxplot(x='subcluster', y='pIC50', data=cluster_df)
            plt.title(f'pIC50 Distribution by Subcluster (Cluster {cluster_id})')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Add count and mean annotations
            subcluster_stats = cluster_df.groupby('subcluster')['pIC50'].agg(['mean', 'count'])
            for subcluster, stats in subcluster_stats.iterrows():
                plt.text(subcluster, cluster_df['pIC50'].min() - 0.2, 
                        f"n={int(stats['count'])}\nμ={stats['mean']:.2f}", 
                        ha='center', va='top', fontsize=9)
            
            plt.savefig(os.path.join(output_dir, f"pIC50_by_subcluster_cluster{cluster_id}.png"))
            plt.close()
            
            print(f"\nSubcluster statistics for Cluster {cluster_id}:")
            subcluster_full_stats = cluster_df.groupby('subcluster')['pIC50'].agg(['count', 'mean', 'std', 'min', 'max'])
            print(subcluster_full_stats)
    
    print(f"Saved cluster visualization to {output_dir}")

def chemical_space_visualization(data_path, output_dir=None):
    """Visualize chemical space using dimensionality reduction"""
    if output_dir is None:
        output_dir = f"data/processed/{timestamp}/visualizations"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    if isinstance(data_path, pd.DataFrame):
        df = data_path
    else:
        df = pd.read_csv(data_path)
    
    # Get fingerprint columns (if present)
    exclude_cols = ['molecule_chembl_id', 'canonical_smiles', 'pIC50', 'cluster', 'subcluster', 'RO5_violations']
    exclude_cols.extend([c for c in df.columns if c.startswith('MW') or c.startswith('LogP') or c.startswith('HB')])
    fingerprint_cols = [col for col in df.columns if col not in exclude_cols]
    
    # If no fingerprints found, try to generate them
    if len(fingerprint_cols) < 10:
        print("Fingerprint data not found in input file, generating MACCS keys...")
        try:
            from rdkit import Chem
            from rdkit.Chem import MACCSkeys
            
            # Generate MACCS fingerprints
            mols = [Chem.MolFromSmiles(s) for s in df['canonical_smiles']]
            valid_mols = [m for m in mols if m is not None]
            fingerprints = [list(MACCSkeys.GenMACCSKeys(m)) for m in valid_mols]
            
            # Convert to DataFrame columns
            fp_df = pd.DataFrame(fingerprints, columns=[f'bit_{i}' for i in range(len(fingerprints[0]))])
            
            # Keep only valid molecules in the original dataframe
            df = df.iloc[:len(valid_mols)].reset_index(drop=True)
            df = pd.concat([df, fp_df], axis=1)
            
            fingerprint_cols = [f'bit_{i}' for i in range(len(fingerprints[0]))]
            
        except Exception as e:
            print(f"Could not generate fingerprints: {e}")
            return
    
    # Reduce dimensions with PCA and t-SNE
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    
    print("Performing dimensionality reduction for chemical space visualization...")
    X = df[fingerprint_cols].values
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA first to reduce dimensions (for speed)
    pca = PCA(n_components=min(50, X_scaled.shape[1], X_scaled.shape[0]))
    X_pca = pca.fit_transform(X_scaled)
    
    # t-SNE for final visualization
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X_pca)
    
    # Create visualization DataFrame
    vis_df = pd.DataFrame({
        'TSNE1': X_tsne[:, 0],
        'TSNE2': X_tsne[:, 1],
        'pIC50': df['pIC50']
    })
    
    if 'cluster' in df.columns:
        vis_df['cluster'] = df['cluster']
    
    # Plot t-SNE visualization
    plt.figure(figsize=(16, 7))
    
    if 'cluster' in vis_df.columns:
        plt.subplot(1, 2, 1)
        sns.scatterplot(x='TSNE1', y='TSNE2', hue='cluster', palette='viridis', data=vis_df)
        plt.title('Chemical Space by Cluster')
        plt.legend(title='Cluster')
        
        plt.subplot(1, 2, 2)
    
    scatter = sns.scatterplot(x='TSNE1', y='TSNE2', hue='pIC50', palette='coolwarm', data=vis_df)
    plt.title('Chemical Space by Activity (pIC50)')
    plt.colorbar(scatter.collections[0], label='pIC50')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "chemical_space.png"))
    plt.close()
    
    # If subclusters are present, visualize one major cluster with subclusters
    if 'cluster' in df.columns and 'subcluster' in df.columns:
        # Find the largest cluster
        largest_cluster = df['cluster'].value_counts().idxmax()
        cluster_df = df[df['cluster'] == largest_cluster]
        
        # Filter visualization dataframe
        cluster_vis_df = vis_df[vis_df['cluster'] == largest_cluster].copy()
        cluster_vis_df['subcluster'] = df[df['cluster'] == largest_cluster]['subcluster'].values
        
        # Plot
        plt.figure(figsize=(16, 7))
        
        plt.subplot(1, 2, 1)
        sns.scatterplot(x='TSNE1', y='TSNE2', hue='subcluster', palette='tab10', data=cluster_vis_df)
        plt.title(f'Cluster {largest_cluster} Subcluster Distribution')
        plt.legend(title='Subcluster')
        
        plt.subplot(1, 2, 2)
        scatter = sns.scatterplot(x='TSNE1', y='TSNE2', hue='pIC50', palette='coolwarm', data=cluster_vis_df)
        plt.title(f'Cluster {largest_cluster} Activity Distribution')
        plt.colorbar(scatter.collections[0], label='pIC50')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"cluster_{largest_cluster}_subclusters.png"))
        plt.close()
    
    print(f"Saved chemical space visualization to {output_dir}")

def activity_landscape_3d(data_path, output_dir=None):
    """Create a 3D visualization of the activity landscape"""
    if output_dir is None:
        output_dir = f"data/processed/{timestamp}/visualizations"
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        import plotly.graph_objects as go
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        print("Plotly is required for 3D visualization. Install with: pip install plotly")
        return
    
    # Load data
    if isinstance(data_path, pd.DataFrame):
        df = data_path
    else:
        df = pd.read_csv(data_path)
    
    # Get fingerprint columns
    exclude_cols = ['molecule_chembl_id', 'canonical_smiles', 'pIC50', 'cluster', 'subcluster', 'RO5_violations']
    exclude_cols.extend([c for c in df.columns if c.startswith('MW') or c.startswith('LogP') or c.startswith('HB')])
    fingerprint_cols = [col for col in df.columns if col not in exclude_cols]
    
    if len(fingerprint_cols) < 10:
        print("Not enough descriptor columns for 3D activity landscape")
        return
    
    # Standardize and reduce dimensions with PCA
    print("Performing dimensionality reduction for 3D landscape...")
    X = df[fingerprint_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=3)
    components = pca.fit_transform(X_scaled)
    
    # Create hover text
    hover_text = []
    for i, row in df.iterrows():
        text = f"pIC50: {row['pIC50']:.2f}"
        if 'molecule_chembl_id' in row:
            text = f"{row['molecule_chembl_id']}<br>" + text
        if 'cluster' in row:
            text += f"<br>Cluster: {row['cluster']}"
        if 'subcluster' in row:
            text += f"<br>Subcluster: {row['subcluster']}"
        hover_text.append(text)
    
    # Create 3D scatter plot
    marker_size = min(8, max(3, 10000 // len(df)))  # Adaptive marker size
    
    fig = go.Figure(data=[go.Scatter3d(
        x=components[:, 0],
        y=components[:, 1],
        z=components[:, 2],
        mode='markers',
        marker=dict(
            size=marker_size,
            color=df['pIC50'],
            colorscale='Viridis',
            colorbar=dict(title='pIC50'),
            opacity=0.8
        ),
        text=hover_text,
        hoverinfo='text'
    )])
    
    # Update layout
    fig.update_layout(
        title="3D KIT Inhibitor Activity Landscape",
        scene=dict(
            xaxis_title="PCA Component 1",
            yaxis_title="PCA Component 2",
            zaxis_title="PCA Component 3"
        ),
        width=900,
        height=800
    )
    
    html_file = os.path.join(output_dir, "activity_landscape_3d.html")
    
    # Save with explicit UTF-8 encoding
    fig.write_html(html_file, 
                   include_plotlyjs='cdn',  # Use CDN instead of inline to reduce file size
                   config={'displayModeBar': True, 'toImageButtonOptions': {'format': 'png'}})
    
    print(f"3D activity landscape saved to {html_file}")

def analyze_all(data_timestamp=None):
    """Run all visualization functions"""
    if data_timestamp is None:
        data_timestamp = timestamp
    
    print("\n" + "="*60)
    print(f"COMPREHENSIVE VISUALIZATION ANALYSIS (Dataset: {data_timestamp})")
    print("="*60)
    
    # Define paths
    base_path = f"data/processed/{data_timestamp}"
    output_dir = f"{base_path}/visualizations"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initial data path - look for main dataset
    data_path = f"{base_path}/kit_pic50.csv"
    if not os.path.exists(data_path):
        # Try alternative with timestamp in filename
        data_path = f"{base_path}/kit_pic50_{data_timestamp}.csv"
        if not os.path.exists(data_path):
            # Look for any CSV files in the directory
            csv_files = [f for f in os.listdir(base_path) if f.endswith('.csv')]
            if csv_files:
                data_path = os.path.join(base_path, csv_files[0])
                print(f"Using data file: {data_path}")
            else:
                print(f"No CSV data files found in {base_path}")
                return
    
    # 1. Basic pIC50 distribution
    print("\n===== BASIC DISTRIBUTION ANALYSIS =====")
    plot_distribution(data_path, output_dir)
    
    # 2. Property analysis (which also calculates properties if needed)
    print("\n===== MOLECULAR PROPERTY ANALYSIS =====")
    df_with_props = plot_property_distributions(data_path, output_dir)
    
    # 3. Property-activity relationships
    print("\n===== PROPERTY-ACTIVITY RELATIONSHIPS =====")
    plot_activity_vs_properties(df_with_props, output_dir)
    
    # 4. Check for clustered data
    clustered_path = f"{base_path}/clusters/kit_fingerprints_clustered.csv"
    if os.path.exists(clustered_path):
        print("\n===== CLUSTER ANALYSIS =====")
        print(f"Found clustered data at: {clustered_path}")
        visualize_clusters(clustered_path, output_dir)
        
        # 5. Chemical space visualization with clusters
        print("\n===== CHEMICAL SPACE VISUALIZATION =====")
        chemical_space_visualization(clustered_path, output_dir)
        
        # 6. 3D landscape with clustered data
        print("\n===== 3D ACTIVITY LANDSCAPE =====")
        try:
            activity_landscape_3d(clustered_path, output_dir)
        except Exception as e:
            print(f"Could not create 3D visualization: {e}")
    else:
        # If no clustered data, try fingerprint data
        fingerprint_path = f"{base_path}/kit_fingerprints.csv"
        if os.path.exists(fingerprint_path):
            print("\n===== CHEMICAL SPACE VISUALIZATION =====")
            print(f"Found fingerprint data at: {fingerprint_path}")
            chemical_space_visualization(fingerprint_path, output_dir)
            
            print("\n===== 3D ACTIVITY LANDSCAPE =====")
            try:
                activity_landscape_3d(fingerprint_path, output_dir)
            except Exception as e:
                print(f"Could not create 3D visualization: {e}")
        else:
            # Use the data with calculated properties
            print("\n===== CHEMICAL SPACE VISUALIZATION =====")
            print("Using property data for dimensionality reduction")
            chemical_space_visualization(df_with_props, output_dir)
    
    print(f"\nAll visualizations saved to: {output_dir}")
    print("="*60)

if __name__ == "__main__":
    # Run all visualizations with automatic file detection
    analyze_all()   
    
"""
Activity Data Analysis
1. pIC50 Distribution
    Histogram with density curve of activity values
    Activity thresholds (moderate, high, very high)
    Statistical breakdown of compounds by activity class
    
Molecular Property Analysis
    
2. Key Physicochemical Properties
    Molecular weight (MW) distribution
    Lipophilicity (LogP) patterns
    Hydrogen bond donors/acceptors (HBD/HBA)
    Topological polar surface area (TPSA)
    Rotatable bond counts
    Aromatic ring counts
    
3. Drug-Likeness Assessment
    Lipinski Rule of 5 compliance
    Visualization of violation counts
    Percentage of compounds meeting drug-like criteria
    
Structure-Activity Relationships

4. Property-Activity Correlations
    Scatter plots of top correlating properties vs. pIC50
    Trendlines showing relationship direction
    Correlation coefficients for quantitative assessment
    
5. Correlation Matrix
    Heatmap of inter-property correlations
    Identification of collinear descriptors
    Relationship strength between properties and activity
    
Chemical Space Analysis

6. Cluster Distribution
    Bar charts showing compound distribution across clusters
    Boxplots of pIC50 distribution by cluster
    Statistical summary of each cluster's activity profile
    
7. Subcluster Analysis
    Activity distribution within subclusters
    Comparative statistics between subclusters
    Identification of high-activity subclusters
    
8. 2D Chemical Space Maps
    t-SNE visualization of compound similarity
    Color-coding by cluster assignment
    Activity gradient visualization
    
9. 3D Activity Landscape
    Interactive 3D visualization of chemical space
    PCA-based dimensionality reduction
    Identification of activity cliffs and SAR patterns
    Hover information for individual compounds

"""
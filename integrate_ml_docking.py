"""
Integration of Machine Learning Predictions and Docking Results
================================================================

This script combines:
1. ML classification predictions (activity probability)
2. Molecular docking scores (binding affinity)
3. Generates consensus ranking for experimental validation

Author: C-SCAN Pipeline
Date: January 30, 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

timestamp = datetime.now().strftime("%Y%m%d")


class MLDockingIntegrator:
    """Integrate ML predictions with docking results for consensus ranking"""
    
    def __init__(self, ml_predictions_path, docking_scores_path):
        """
        Initialize integrator with ML and docking data
        
        Parameters:
        -----------
        ml_predictions_path : str
            Path to ML screening results (all_predictions.csv or high_efficiency_compounds.csv)
        docking_scores_path : str
            Path to docking results (best_docking_scores.csv)
        """
        
        print("="*70)
        print("ML-DOCKING INTEGRATION PIPELINE")
        print("="*70)
        
        # Load ML predictions
        print(f"\nLoading ML predictions from: {ml_predictions_path}")
        self.ml_data = pd.read_csv(ml_predictions_path)
        print(f"  Loaded {len(self.ml_data)} ML predictions")
        
        # Load docking scores
        print(f"\nLoading docking scores from: {docking_scores_path}")
        self.docking_data = pd.read_csv(docking_scores_path)
        print(f"  Loaded {len(self.docking_data)} docking scores")
        
        # Merge datasets
        self._merge_data()
    
    def _detect_id_column(self, df, dataset_name):
        """Auto-detect compound ID column"""
        possible_names = ['MOL_ID', 'Mol_ID', 'mol_id', 'compound_ID', 'Compound_ID', 
                         'compound_id', 'CompoundID', 'ID', 'id', 'molecule_id', 'Molecule_ID']
        
        for col in possible_names:
            if col in df.columns:
                print(f"  Detected {dataset_name} ID column: {col}")
                return col
        
        print(f"  Warning: No ID column found in {dataset_name} data")
        print(f"  Available columns: {df.columns.tolist()}")
        return None
    
    def _detect_score_column(self, df):
        """Auto-detect docking score column"""
        possible_names = ['score', 'Score', 'SCORE', 'docking_score', 'Docking_Score',
                         'binding_affinity', 'Binding_Affinity', 'affinity', 'Affinity']
        
        for col in possible_names:
            if col in df.columns:
                print(f"  Detected docking score column: {col}")
                return col
        
        # If not found, try to find numeric column with 'score' in name
        score_cols = [col for col in df.columns if 'score' in col.lower()]
        if score_cols:
            print(f"  Detected docking score column: {score_cols[0]}")
            return score_cols[0]
        
        print(f"  Warning: No score column found")
        print(f"  Available columns: {df.columns.tolist()}")
        return None
    
    def _merge_data(self):
        """Merge ML and docking data on MOL_ID or compound_ID"""
        
        print("\nMerging ML predictions with docking scores...")
        
        # Auto-detect ID column in ML data
        ml_id_col = self._detect_id_column(self.ml_data, 'ML')
        
        # Auto-detect ID column in docking data
        dock_id_col = self._detect_id_column(self.docking_data, 'docking')
        
        if ml_id_col is None or dock_id_col is None:
            print("  ERROR: Could not find ID columns in both datasets")
            return
        
        # Standardize column names
        if ml_id_col != 'MOL_ID':
            self.ml_data['MOL_ID'] = self.ml_data[ml_id_col]
        if dock_id_col != 'MOL_ID':
            self.docking_data['MOL_ID'] = self.docking_data[dock_id_col]
        
        # Auto-detect score column in docking data
        score_col = self._detect_score_column(self.docking_data)
        
        # Auto-detect score column in docking data
        score_col = self._detect_score_column(self.docking_data)
        
        if score_col is None:
            print("  ERROR: Could not find score column in docking data")
            return
        
        # Merge on MOL_ID (inner join - only compounds with both ML and docking)
        self.merged_data = pd.merge(
            self.ml_data,
            self.docking_data[['MOL_ID', score_col]],
            on='MOL_ID',
            how='inner'
        )
        
        # Rename score column to standard name
        if score_col != 'Docking_Score':
            self.merged_data.rename(columns={score_col: 'Docking_Score'}, inplace=True)
        
        print(f"  Merged dataset: {len(self.merged_data)} compounds with both ML and docking data")
        
        if len(self.merged_data) == 0:
            print("\n  ERROR: No overlapping compounds found!")
            print("  Check that MOL_ID values match between files")
            return
        
        # Add normalized scores for consensus ranking
        self._calculate_consensus_scores()
    
    def _calculate_consensus_scores(self):
        """Calculate normalized scores and consensus ranking"""
        
        print("\nCalculating consensus scores...")
        
        # Normalize ML probability (already 0-1)
        self.merged_data['ML_Score_Norm'] = self.merged_data['Prob_1']
        
        # Normalize docking score (more negative = better)
        # Convert to 0-1 scale where 1 = best (most negative)
        min_dock = self.merged_data['Docking_Score'].min()
        max_dock = self.merged_data['Docking_Score'].max()
        
        # Invert so more negative scores get higher values
        self.merged_data['Docking_Score_Norm'] = (
            (max_dock - self.merged_data['Docking_Score']) / (max_dock - min_dock)
        )
        
        # Calculate consensus score (weighted average)
        # Default: 50% ML, 50% docking
        self.merged_data['Consensus_Score'] = (
            0.5 * self.merged_data['ML_Score_Norm'] + 
            0.5 * self.merged_data['Docking_Score_Norm']
        )
        
        # Sort by consensus score
        self.merged_data = self.merged_data.sort_values('Consensus_Score', ascending=False)
        
        print("  Calculated normalized ML scores, docking scores, and consensus ranking")
    
    def get_consensus_hits(self, ml_threshold=0.7, docking_threshold=-6.0, top_n=50):
        """
        Get consensus hits that pass both ML and docking thresholds
        
        Parameters:
        -----------
        ml_threshold : float
            Minimum ML probability (default: 0.7)
        docking_threshold : float
            Maximum docking score in kcal/mol (default: -6.0)
        top_n : int
            Return top N compounds by consensus score
        
        Returns:
        --------
        consensus_hits : DataFrame
            Top consensus hits passing both thresholds
        """
        
        print("\n" + "="*70)
        print("CONSENSUS HIT IDENTIFICATION")
        print("="*70)
        
        # Filter by thresholds
        consensus = self.merged_data[
            (self.merged_data['Prob_1'] >= ml_threshold) &
            (self.merged_data['Docking_Score'] <= docking_threshold)
        ].head(top_n).copy()
        
        print(f"\nThresholds:")
        print(f"  ML Probability >= {ml_threshold}")
        print(f"  Docking Score <= {docking_threshold} kcal/mol")
        print(f"\nConsensus hits: {len(consensus)} compounds")
        
        return consensus
    
    def analyze_correlation(self):
        """Analyze correlation between ML predictions and docking scores"""
        
        print("\n" + "="*70)
        print("CORRELATION ANALYSIS")
        print("="*70)
        
        # Calculate correlation
        corr = self.merged_data['Prob_1'].corr(self.merged_data['Docking_Score'])
        print(f"\nPearson correlation (ML Prob vs Docking Score): {corr:.3f}")
        
        if abs(corr) < 0.3:
            print("  Weak correlation - ML and docking provide complementary information")
        elif abs(corr) < 0.7:
            print("  Moderate correlation - Some agreement between methods")
        else:
            print("  Strong correlation - High agreement between methods")
        
        return corr
    
    def visualize_results(self, output_dir='docking_results/integrated'):
        """Generate visualizations of integrated results"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n" + "="*70)
        print("GENERATING VISUALIZATIONS")
        print("="*70)
        
        # 1. Scatter plot: ML probability vs Docking score
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(
            self.merged_data['Prob_1'],
            self.merged_data['Docking_Score'],
            c=self.merged_data['Consensus_Score'],
            cmap='RdYlGn',
            s=100,
            alpha=0.6,
            edgecolors='black',
            linewidths=0.5
        )
        plt.colorbar(scatter, label='Consensus Score')
        plt.xlabel('ML Probability (Active)', fontsize=12)
        plt.ylabel('Docking Score (kcal/mol)', fontsize=12)
        plt.title('ML Predictions vs Docking Scores', fontsize=14, fontweight='bold')
        plt.axhline(y=-6.0, color='red', linestyle='--', label='Docking threshold (-6.0)')
        plt.axvline(x=0.7, color='blue', linestyle='--', label='ML threshold (0.7)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/ml_vs_docking_scatter.png', dpi=300)
        plt.close()
        print(f"  Saved: ml_vs_docking_scatter.png")
        
        # 2. Distribution comparison
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # ML probability distribution
        axes[0].hist(self.merged_data['Prob_1'], bins=30, edgecolor='black', alpha=0.7, color='steelblue')
        axes[0].axvline(0.7, color='red', linestyle='--', label='Threshold (0.7)')
        axes[0].set_xlabel('ML Probability (Active)', fontsize=11)
        axes[0].set_ylabel('Count', fontsize=11)
        axes[0].set_title('ML Probability Distribution', fontsize=12, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Docking score distribution
        axes[1].hist(self.merged_data['Docking_Score'], bins=30, edgecolor='black', alpha=0.7, color='coral')
        axes[1].axvline(-6.0, color='red', linestyle='--', label='Threshold (-6.0)')
        axes[1].set_xlabel('Docking Score (kcal/mol)', fontsize=11)
        axes[1].set_ylabel('Count', fontsize=11)
        axes[1].set_title('Docking Score Distribution', fontsize=12, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/score_distributions.png', dpi=300)
        plt.close()
        print(f"  Saved: score_distributions.png")
        
        # 3. Top 20 consensus hits bar chart
        top20 = self.merged_data.head(20).copy()
        
        fig, ax = plt.subplots(figsize=(12, 8))
        x = np.arange(len(top20))
        width = 0.35
        
        bars1 = ax.barh(x - width/2, top20['ML_Score_Norm'], width, label='ML Score (Norm)', color='steelblue')
        bars2 = ax.barh(x + width/2, top20['Docking_Score_Norm'], width, label='Docking Score (Norm)', color='coral')
        
        ax.set_yticks(x)
        ax.set_yticklabels(top20['MOL_ID'])
        ax.set_xlabel('Normalized Score', fontsize=11)
        ax.set_title('Top 20 Consensus Hits - Score Comparison', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/top20_consensus_comparison.png', dpi=300)
        plt.close()
        print(f"  Saved: top20_consensus_comparison.png")
        
        # 4. Consensus score ranking
        plt.figure(figsize=(12, 6))
        plt.plot(range(1, len(self.merged_data)+1), self.merged_data['Consensus_Score'].values, 
                 linewidth=2, color='green')
        plt.xlabel('Compound Rank', fontsize=11)
        plt.ylabel('Consensus Score', fontsize=11)
        plt.title('Consensus Score Ranking', fontsize=13, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/consensus_ranking.png', dpi=300)
        plt.close()
        print(f"  Saved: consensus_ranking.png")
        
        print(f"\n  All visualizations saved to: {output_dir}/")
    
    def save_results(self, output_dir='docking_results/integrated', 
                    ml_threshold=0.7, docking_threshold=-6.0):
        """Save integrated results and consensus hits"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n" + "="*70)
        print("SAVING RESULTS")
        print("="*70)
        
        # Save full merged dataset
        output_cols = ['MOL_ID', 'SMILES', 'Prob_1', 'Docking_Score', 
                      'ML_Score_Norm', 'Docking_Score_Norm', 'Consensus_Score', 'Predicted_Class']
        available_cols = [col for col in output_cols if col in self.merged_data.columns]
        
        self.merged_data[available_cols].to_csv(
            f'{output_dir}/ml_docking_integrated.csv', index=False
        )
        print(f"  Saved: ml_docking_integrated.csv ({len(self.merged_data)} compounds)")
        
        # Save consensus hits
        consensus_hits = self.merged_data[
            (self.merged_data['Prob_1'] >= ml_threshold) &
            (self.merged_data['Docking_Score'] <= docking_threshold)
        ]
        
        consensus_hits[available_cols].to_csv(
            f'{output_dir}/consensus_hits.csv', index=False
        )
        print(f"  Saved: consensus_hits.csv ({len(consensus_hits)} compounds)")
        
        # Save top 50 by consensus score
        top50 = self.merged_data.head(50)
        top50[available_cols].to_csv(
            f'{output_dir}/top50_consensus.csv', index=False
        )
        print(f"  Saved: top50_consensus.csv (50 compounds)")
        
        return consensus_hits
    
    def generate_report(self, output_path='docking_results/integrated/INTEGRATION_REPORT.txt'):
        """Generate comprehensive text report"""
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("ML-DOCKING INTEGRATION REPORT\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*70 + "\n\n")
            
            # Dataset summary
            f.write("DATASET SUMMARY\n")
            f.write("-"*70 + "\n")
            f.write(f"Total compounds with ML predictions: {len(self.ml_data)}\n")
            f.write(f"Total compounds with docking scores: {len(self.docking_data)}\n")
            f.write(f"Compounds with both ML and docking: {len(self.merged_data)}\n\n")
            
            # Score statistics
            f.write("SCORE STATISTICS\n")
            f.write("-"*70 + "\n")
            f.write(f"ML Probability Range: {self.merged_data['Prob_1'].min():.3f} - {self.merged_data['Prob_1'].max():.3f}\n")
            f.write(f"ML Probability Mean: {self.merged_data['Prob_1'].mean():.3f}\n")
            f.write(f"ML Probability Median: {self.merged_data['Prob_1'].median():.3f}\n\n")
            
            f.write(f"Docking Score Range: {self.merged_data['Docking_Score'].min():.2f} - {self.merged_data['Docking_Score'].max():.2f} kcal/mol\n")
            f.write(f"Docking Score Mean: {self.merged_data['Docking_Score'].mean():.2f} kcal/mol\n")
            f.write(f"Docking Score Median: {self.merged_data['Docking_Score'].median():.2f} kcal/mol\n\n")
            
            # Correlation
            corr = self.merged_data['Prob_1'].corr(self.merged_data['Docking_Score'])
            f.write(f"Correlation (ML vs Docking): {corr:.3f}\n\n")
            
            # Threshold analysis
            f.write("THRESHOLD ANALYSIS\n")
            f.write("-"*70 + "\n")
            
            thresholds = [
                (0.7, -6.0),
                (0.8, -6.5),
                (0.9, -7.0)
            ]
            
            for ml_thresh, dock_thresh in thresholds:
                count = len(self.merged_data[
                    (self.merged_data['Prob_1'] >= ml_thresh) &
                    (self.merged_data['Docking_Score'] <= dock_thresh)
                ])
                f.write(f"ML >= {ml_thresh} AND Docking <= {dock_thresh}: {count} compounds\n")
            
            f.write("\n")
            
            # Top 20 compounds
            f.write("TOP 20 CONSENSUS HITS\n")
            f.write("-"*70 + "\n")
            top20 = self.merged_data.head(20)
            
            f.write(f"{'Rank':<6}{'MOL_ID':<12}{'ML_Prob':<12}{'Dock_Score':<14}{'Consensus':<12}\n")
            f.write("-"*70 + "\n")
            
            for idx, (_, row) in enumerate(top20.iterrows(), 1):
                f.write(f"{idx:<6}{row['MOL_ID']:<12}{row['Prob_1']:<12.3f}"
                       f"{row['Docking_Score']:<14.2f}{row['Consensus_Score']:<12.3f}\n")
            
            f.write("\n")
            
            # Recommendations
            f.write("RECOMMENDATIONS FOR EXPERIMENTAL VALIDATION\n")
            f.write("-"*70 + "\n")
            f.write("Priority 1 (Immediate Testing):\n")
            f.write("  Top 5-10 consensus hits with ML Prob >= 0.8 AND Docking <= -6.5\n\n")
            
            f.write("Priority 2 (High Priority):\n")
            f.write("  Next 10-20 compounds with ML Prob >= 0.7 AND Docking <= -6.0\n\n")
            
            f.write("Priority 3 (Extended Panel):\n")
            f.write("  Top 50 consensus hits for broader validation\n\n")
            
            f.write("="*70 + "\n")
        
        print(f"\n  Saved comprehensive report: {output_path}")


def main():
    """Main execution function"""
    
    # Paths to data files
    ml_predictions = "predictions/20260130/blind_screening/all_predictions.csv"
    docking_scores = "docking_results/best_docking_scores_from_BS.csv"
    output_dir = "docking_results/integrated"
    
    # Check if files exist
    if not os.path.exists(ml_predictions):
        print(f"Error: ML predictions file not found: {ml_predictions}")
        return
    
    if not os.path.exists(docking_scores):
        print(f"Error: Docking scores file not found: {docking_scores}")
        return
    
    # Initialize integrator
    integrator = MLDockingIntegrator(ml_predictions, docking_scores)
    
    # Analyze correlation
    integrator.analyze_correlation()
    
    # Get consensus hits
    consensus = integrator.get_consensus_hits(ml_threshold=0.7, docking_threshold=-6.0, top_n=50)
    
    # Display top 10
    print("\n" + "="*70)
    print("TOP 10 CONSENSUS HITS")
    print("="*70)
    print(f"\n{'Rank':<6}{'MOL_ID':<12}{'ML_Prob':<12}{'Dock_Score':<14}{'Consensus':<12}")
    print("-"*70)
    
    for idx, (_, row) in enumerate(consensus.head(10).iterrows(), 1):
        print(f"{idx:<6}{row['MOL_ID']:<12}{row['Prob_1']:<12.3f}"
              f"{row['Docking_Score']:<14.2f}{row['Consensus_Score']:<12.3f}")
    
    # Generate visualizations
    integrator.visualize_results(output_dir)
    
    # Save results
    integrator.save_results(output_dir, ml_threshold=0.7, docking_threshold=-6.0)
    
    # Generate report
    integrator.generate_report(f'{output_dir}/INTEGRATION_REPORT.txt')
    
    print("\n" + "="*70)
    print("INTEGRATION COMPLETE!")
    print("="*70)
    print(f"\nAll results saved to: {output_dir}/")
    print("\nKey files:")
    print(f"  - ml_docking_integrated.csv (all {len(integrator.merged_data)} compounds)")
    print(f"  - consensus_hits.csv (compounds passing both thresholds)")
    print(f"  - top50_consensus.csv (top 50 by consensus score)")
    print(f"  - INTEGRATION_REPORT.txt (comprehensive analysis)")
    print(f"  - Visualization PNGs (4 charts)")
    print("\nNext steps:")
    print("  1. Review consensus_hits.csv for experimental validation")
    print("  2. Check visualizations for agreement between methods")
    print("  3. Use top compounds for wet-lab testing")
    print("  4. Retrain ML model with experimental results when available")


if __name__ == "__main__":
    main()

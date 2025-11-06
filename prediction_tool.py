import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, PandasTools
from rdkit.Chem.Scaffolds import MurckoScaffold

timestamp = ""

class KIT_Inhibitor_Predictor:
    """
    A class for predicting KIT (c-Kit) inhibitor activity (pIC50) using the 
    trained subcluster-specific model.
    
    KIT is a receptor tyrosine kinase involved in cell growth and differentiation.
    Inhibition of KIT is important for treating gastrointestinal stromal tumors (GIST),
    certain leukemias, systemic mastocytosis, and some melanomas.
    """
    
    def __init__(self, subcluster_id=0):
        """Initialize the predictor with the specified subcluster model"""
        self.subcluster_id = subcluster_id
        self.model_path = f"models/subcluster_models/subcluster_{subcluster_id}_optimized_model.pkl"
        self.scaler_path = f"models/subcluster_models/subcluster_{subcluster_id}_scaler.pkl"
        
        # Check if model files exist
        if not os.path.exists(self.model_path) or not os.path.exists(self.scaler_path):
            raise FileNotFoundError(f"Model or scaler file not found for subcluster {subcluster_id}")
        
        # Load model and scaler
        with open(self.model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        with open(self.scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        print(f"Loaded model for subcluster {subcluster_id}")
        
        # Get model summary information if available
        summary_path = f"models/subcluster_models/subcluster_{subcluster_id}_summary.csv"
        if os.path.exists(summary_path):
            self.summary = pd.read_csv(summary_path)
            print(f"Model performance: R² = {self.summary['optimized_r2'].values[0]:.3f}, "
                  f"RMSE = {self.summary['optimized_rmse'].values[0]:.3f}")
        else:
            self.summary = None
    
    def _generate_fingerprints(self, smiles_list, radius=2, nBits=2048):
        """Generate Morgan fingerprints for a list of SMILES strings"""
        fingerprints = []
        valid_smiles = []
        valid_mols = []
        errors = []
        
        for smiles in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
                    fingerprints.append(list(fp))
                    valid_smiles.append(smiles)
                    valid_mols.append(mol)
                else:
                    errors.append((smiles, "Invalid SMILES - could not parse"))
            except Exception as e:
                errors.append((smiles, str(e)))
        
        return fingerprints, valid_smiles, valid_mols, errors
    
    def predict(self, smiles_list, output_csv=None):
        """Predict pIC50 values for a list of compounds defined by SMILES strings"""
        if isinstance(smiles_list, str):
            # If a single SMILES is passed
            smiles_list = [smiles_list]
        
        # Generate fingerprints
        fingerprints, valid_smiles, valid_mols, errors = self._generate_fingerprints(smiles_list)
        
        if not fingerprints:
            print("Error: No valid compounds to predict")
            return None
        
        # Convert fingerprints to array, scale, and predict
        X = np.array(fingerprints)
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        
        # Create results dataframe
        results = pd.DataFrame({
            'SMILES': valid_smiles,
            'Predicted_pIC50': predictions
        })
        
        # Add predicted activity level
        def activity_level(pic50):
            if pic50 < 6:
                return "Low"
            elif pic50 < 7:
                return "Moderate"
            elif pic50 < 8:
                return "High"
            else:
                return "Very High"
        
        results['Activity_Level'] = results['Predicted_pIC50'].apply(activity_level)
        
        # Sort by predicted activity
        results = results.sort_values('Predicted_pIC50', ascending=False)
        
        # Save results if requested
        if output_csv:
            os.makedirs(os.path.dirname(output_csv), exist_ok=True)
            results.to_csv(output_csv, index=False)
            print(f"Results saved to {output_csv}")
        
        # Report any errors
        if errors:
            print(f"\nWarning: {len(errors)} SMILES had errors:")
            for smiles, error in errors[:5]:  # Show first 5 errors
                print(f"  {smiles}: {error}")
            if len(errors) > 5:
                print(f"  ...and {len(errors) - 5} more")
        
        return results, valid_mols
    
    def predict_and_visualize(self, smiles_list, output_dir="predictions", filename_prefix="prediction"):
        """Predict activities and generate visualizations"""
        results, mols = self.predict(smiles_list)
        
        if results is None or len(results) == 0:
            return None
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save results
        csv_path = os.path.join(output_dir, f"{filename_prefix}_results.csv")
        results.to_csv(csv_path, index=False)
        
        # Generate visualization with molecules and predictions
        if mols:
            # Generate molecule images with predictions
            legends = [f"pIC50: {pic50:.2f} ({level})" 
                      for pic50, level in zip(results['Predicted_pIC50'], results['Activity_Level'])]
            
            img = Draw.MolsToGridImage(
                mols, molsPerRow=3, subImgSize=(300, 300),
                legends=legends
            )
            img_path = os.path.join(output_dir, f"{filename_prefix}_molecules.png")
            img.save(img_path)
            print(f"Molecule visualization saved to {img_path}")
            
            # Create a bar plot of predictions
            plt.figure(figsize=(10, 6))
            # Limit to top 15 compounds for readability
            plot_data = results.head(15)
            bars = plt.bar(range(len(plot_data)), plot_data['Predicted_pIC50'], color='skyblue')
            
            # Color bars by activity level
            colors = {'Low': 'lightblue', 'Moderate': 'skyblue', 
                      'High': 'royalblue', 'Very High': 'darkblue'}
            for i, level in enumerate(plot_data['Activity_Level']):
                bars[i].set_color(colors[level])
                
            plt.xlabel('Compound')
            plt.ylabel('Predicted pIC50')
            plt.title('Predicted KIT Inhibitor Activity')
            plt.xticks(range(len(plot_data)), [f"Cpd {i+1}" for i in range(len(plot_data))], rotation=45)
            plt.ylim(bottom=max(4.5, min(plot_data['Predicted_pIC50']) - 0.5))
            plt.tight_layout()
            
            # Add a legend for activity levels
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor=colors[level], label=level)
                              for level in ['Low', 'Moderate', 'High', 'Very High']]
            plt.legend(handles=legend_elements, title="Activity Level")
            
            plot_path = os.path.join(output_dir, f"{filename_prefix}_plot.png")
            plt.savefig(plot_path)
            plt.close()
            print(f"Prediction plot saved to {plot_path}")
        
        print("\nPrediction Results Summary:")
        print(f"Total compounds: {len(results)}")
        for level in ['Low', 'Moderate', 'High', 'Very High']:
            count = sum(results['Activity_Level'] == level)
            print(f"{level} activity compounds: {count} ({100*count/len(results):.1f}%)")
        
        return results
    
    def virtual_screening(self, smiles_file, output_dir="virtual_screening", top_n=50):
        """Screen a virtual library of compounds from a file"""
        print(f"Loading compounds from {smiles_file}...")
        
        # Load SMILES from file
        if smiles_file.endswith('.csv'):
            try:
                df = pd.read_csv(smiles_file)
                # Try to find a column containing SMILES
                smiles_col = next((col for col in df.columns if 'smiles' in col.lower()), df.columns[0])
                smiles_list = df[smiles_col].tolist()
            except Exception as e:
                print(f"Error reading CSV: {e}")
                return None
        else:
            # Assume it's a text file with one SMILES per line
            with open(smiles_file, 'r') as f:
                smiles_list = [line.strip() for line in f if line.strip()]
        
        print(f"Screening {len(smiles_list)} compounds...")
        
        # Make predictions
        results, _ = self.predict(smiles_list)
        
        if results is None:
            return None
            
        # Focus on top compounds
        top_results = results.head(top_n)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save all results
        results.to_csv(os.path.join(output_dir, "screening_all_results.csv"), index=False)
        
        # Save top results
        top_results.to_csv(os.path.join(output_dir, f"screening_top_{top_n}.csv"), index=False)
        
        # Visualize top compounds
        top_mols = [Chem.MolFromSmiles(smiles) for smiles in top_results['SMILES']]
        top_legends = [f"pIC50: {pic50:.2f}" for pic50 in top_results['Predicted_pIC50']]
        
        img = Draw.MolsToGridImage(
            top_mols, molsPerRow=5, subImgSize=(250, 250),
            legends=top_legends
        )
        img.save(os.path.join(output_dir, f"screening_top_{top_n}_compounds.png"))
        
        print(f"\nVirtual screening complete.")
        print(f"Top compound predicted pIC50: {top_results['Predicted_pIC50'].iloc[0]:.2f}")
        print(f"Results saved to {output_dir}")
        
        return results
    
    def applicability_domain(self, smiles_list):
        """Estimate if compounds are within the model's applicability domain"""
        # This is a simplified approach - more sophisticated methods exist
        fingerprints, valid_smiles, valid_mols, _ = self._generate_fingerprints(smiles_list)
        
        if not fingerprints:
            return None
            
        # Get training data fingerprints (you would need to save these during model building)
        # This is just a placeholder - you would need to implement this fully
        print("Note: Applicability domain check is a placeholder and needs training data")
        
        return {"in_domain": True, "similarity_score": 0.8}

# Example usage
if __name__ == "__main__":
    # Initialize the predictor
    predictor = KIT_Inhibitor_Predictor(subcluster_id=0)
    
    # Example compounds - actual KIT inhibitors
    test_compounds = [
        # Imatinib - first-line KIT inhibitor for GIST
        "CCN(CC)C(=O)Nc1ccc(C)c(Nc2nccc(n2)c3cccnc3)c1",
        
        # Sunitinib - second-line KIT inhibitor
        "CCN(CC)CCNC(=O)c1c(C)[nH]c(C=C2C(=O)Nc3ccc(F)cc23)c1C",
        
        # Masitinib - veterinary and investigational KIT inhibitor
        "Cc1ccc(cc1)NC(=O)c2ccc(CN3CCN(C)CC3)cc2",
        
        # Ripretinib - fourth-generation KIT inhibitor
        "CN1CCN(Cc2ccc(NC(=O)c3cccc(C)c3)cc2)CC1"
    ]
    
    # Make predictions with visualization
    results = predictor.predict_and_visualize(
        test_compounds, 
        output_dir="predictions",
        filename_prefix="kit_inhibitors"
    )
    
    print("\nIndividual predictions complete!")
    
    # Example 2: Virtual screening
    # This would be used if you have a file with many compounds to screen
    # Commented out since the file might not exist in your environment
    '''
    screening_results = predictor.virtual_screening(
        "data/external/screening_library.csv",
        output_dir="virtual_screening",
        top_n=50
    )
    '''
    
    print("\nAll prediction tasks completed!")
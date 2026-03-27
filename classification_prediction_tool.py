import pandas as pd
import numpy as np
import pickle
import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Lipinski, MolSurf, GraphDescriptors
from rdkit.Chem import Draw  # type: ignore[attr-defined]
from rdkit.Chem import rdPartialCharges, rdMolDescriptors
from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d")

class ActivityClassifier:
    """
    A class for predicting activity classes of compounds using trained classification models.
    Can be used for virtual screening to identify high-efficiency compounds.
    """
    
    def __init__(self, model_path, scaler_path=None, label_encoder_path=None, 
                 feature_names_path=None, use_fingerprints=False):
        """
        Initialize the classifier with trained model files
        
        Parameters:
        -----------
        model_path : str
            Path to saved classifier model (.pkl)
        scaler_path : str, optional
            Path to saved scaler (.pkl). If missing, raw features are used.
        label_encoder_path : str, optional
            Path to saved label encoder (.pkl). If missing, raw class labels are used.
        feature_names_path : str, optional
            Path to saved feature names (.pkl)
        use_fingerprints : bool
            If True, generate fingerprints; if False, generate descriptors
        """
        
        # Check if files exist
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if scaler_path and not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
        if label_encoder_path and not os.path.exists(label_encoder_path):
            raise FileNotFoundError(f"Label encoder file not found: {label_encoder_path}")
        
        # Load model components
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        if scaler_path and os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
        else:
            self.scaler = None
            print("[WARNING] Scaler file not found. Proceeding without external scaling.")

        if label_encoder_path and os.path.exists(label_encoder_path):
            with open(label_encoder_path, 'rb') as f:
                self.label_encoder = pickle.load(f)
        else:
            self.label_encoder = None
            print("[WARNING] Label encoder file not found. Using raw model class labels.")
        
        # Load feature names if provided
        if feature_names_path and os.path.exists(feature_names_path):
            if str(feature_names_path).lower().endswith('.txt'):
                with open(feature_names_path, 'r', encoding='utf-8', errors='replace') as f:
                    self.feature_names = [line.strip() for line in f if line.strip()]
            else:
                with open(feature_names_path, 'rb') as f:
                    self.feature_names = pickle.load(f)
        else:
            self.feature_names = None
        
        self.use_fingerprints = use_fingerprints
        
        print(f"[OK] Loaded classification model")
        if self.label_encoder is not None and hasattr(self.label_encoder, 'classes_'):
            print(f"  Classes: {', '.join([str(c) for c in self.label_encoder.classes_])}")
        elif hasattr(self.model, 'classes_'):
            print(f"  Classes: {', '.join([str(c) for c in self.model.classes_])}")
        else:
            print("  Classes: unknown")
        print(f"  Feature type: {'Fingerprints' if use_fingerprints else 'Descriptors'}")
    
    def _generate_fingerprints(self, smiles_list, radius=2, nBits=2048):
        """Generate Morgan fingerprints for a list of SMILES"""
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
                    errors.append((smiles, "Invalid SMILES"))
            except Exception as e:
                errors.append((smiles, str(e)))
        
        return fingerprints, valid_smiles, valid_mols, errors
    
    def _compute_max_partial_charge(self, mol):
        """Compute maximum partial charge"""
        try:
            rdPartialCharges.ComputeGasteigerCharges(mol)
            charges = [float(atom.GetProp('_GasteigerCharge')) for atom in mol.GetAtoms()]
            return max(charges) if charges else 0.0
        except:
            return 0.0
    
    def _compute_min_partial_charge(self, mol):
        """Compute minimum partial charge"""
        try:
            rdPartialCharges.ComputeGasteigerCharges(mol)
            charges = [float(atom.GetProp('_GasteigerCharge')) for atom in mol.GetAtoms()]
            return min(charges) if charges else 0.0
        except:
            return 0.0
    
    def _compute_max_abs_partial_charge(self, mol):
        """Compute maximum absolute partial charge"""
        try:
            rdPartialCharges.ComputeGasteigerCharges(mol)
            charges = [abs(float(atom.GetProp('_GasteigerCharge'))) for atom in mol.GetAtoms()]
            return max(charges) if charges else 0.0
        except:
            return 0.0
    
    def _generate_descriptors(self, smiles_list):
        """Generate molecular descriptors matching training features"""
        
        # Define the same descriptors used in training
        descriptor_functions = {
            'MolWt': Descriptors.MolWt,
            'MolLogP': Descriptors.MolLogP,
            'NumHDonors': Lipinski.NumHDonors,
            'NumHAcceptors': Lipinski.NumHAcceptors,
            'TPSA': MolSurf.TPSA,
            'NumRotatableBonds': Descriptors.NumRotatableBonds,
            'NumAromaticRings': Lipinski.NumAromaticRings,
            'NumAliphaticRings': Lipinski.NumAliphaticRings,
            'FractionCSP3': Descriptors.FractionCSP3,
            'NumHeteroatoms': Descriptors.NumHeteroatoms,
            'RingCount': Descriptors.RingCount,
            'LabuteASA': Descriptors.LabuteASA,
            'SlogP_VSA1': MolSurf.SlogP_VSA1,
            'SMR_VSA1': MolSurf.SMR_VSA1,
            'BalabanJ': Descriptors.BalabanJ,
            'BertzCT': Descriptors.BertzCT,
            'PEOE_VSA1': MolSurf.PEOE_VSA1,
            'PEOE_VSA2': MolSurf.PEOE_VSA2,
            'PEOE_VSA3': MolSurf.PEOE_VSA3,
            'SlogP_VSA2': MolSurf.SlogP_VSA2,
            'SlogP_VSA3': MolSurf.SlogP_VSA3,
            'SMR_VSA2': MolSurf.SMR_VSA2,
            'SMR_VSA3': MolSurf.SMR_VSA3,
            'Chi0': GraphDescriptors.Chi0,
            'Chi1': GraphDescriptors.Chi1,
            'Chi0v': GraphDescriptors.Chi0v,
            'Chi1v': GraphDescriptors.Chi1v,
            'Kappa1': GraphDescriptors.Kappa1,
            'Kappa2': GraphDescriptors.Kappa2,
            'Kappa3': GraphDescriptors.Kappa3,
            'MaxPartialCharge': self._compute_max_partial_charge,
            'MinPartialCharge': self._compute_min_partial_charge,
            'MaxAbsPartialCharge': self._compute_max_abs_partial_charge,
            'NumRings': Descriptors.RingCount,
            'AromaticRings': rdMolDescriptors.CalcNumAromaticRings,
            'AromaticHetero': rdMolDescriptors.CalcNumAromaticHeterocycles,
            'AromaticCarbocycles': rdMolDescriptors.CalcNumAromaticCarbocycles
        }
        
        descriptors = []
        valid_smiles = []
        valid_mols = []
        errors = []
        
        for smiles in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    desc_values = []
                    for desc_name, desc_func in descriptor_functions.items():
                        try:
                            desc_values.append(desc_func(mol))
                        except:
                            desc_values.append(0.0)
                    
                    descriptors.append(desc_values)
                    valid_smiles.append(smiles)
                    valid_mols.append(mol)
                else:
                    errors.append((smiles, "Invalid SMILES"))
            except Exception as e:
                errors.append((smiles, str(e)))
        
        return descriptors, valid_smiles, valid_mols, errors
    
    def predict(self, smiles_list, return_probabilities=True, compound_ids=None):
        """
        Predict activity classes for compounds
        
        Parameters:
        -----------
        smiles_list : list or str
            SMILES string(s) to predict
        return_probabilities : bool
            If True, return class probabilities
        compound_ids : list, optional
            List of compound IDs corresponding to SMILES
        
        Returns:
        --------
        results_df : DataFrame
            DataFrame with predictions and probabilities
        valid_mols : list
            List of valid RDKit molecule objects
        """
        
        if isinstance(smiles_list, str):
            smiles_list = [smiles_list]
            if compound_ids and isinstance(compound_ids, str):
                compound_ids = [compound_ids]
        
        # Generate features
        if self.use_fingerprints:
            features, valid_smiles, valid_mols, errors = self._generate_fingerprints(smiles_list)
        else:
            features, valid_smiles, valid_mols, errors = self._generate_descriptors(smiles_list)
        
        if not features:
            print("Error: No valid compounds to predict")
            return None, None
        
        # Track valid compound IDs (filter out those that failed)
        valid_compound_ids = None
        if compound_ids:
            # Build mapping of original indices that are valid
            valid_indices = []
            for i, smiles in enumerate(smiles_list):
                if smiles in valid_smiles:
                    valid_indices.append(i)
            valid_compound_ids = [compound_ids[i] for i in valid_indices]
        
        # Scale features and predict
        X = np.array(features)
        X_in = self.scaler.transform(X) if self.scaler is not None else X

        predictions = self.model.predict(X_in)
        if self.label_encoder is not None:
            predicted_classes = self.label_encoder.inverse_transform(predictions)
        else:
            predicted_classes = predictions
        
        # Create results dataframe
        results_data = {
            'SMILES': valid_smiles,
            'Predicted_Class': predicted_classes
        }
        
        # Add compound IDs if provided
        if valid_compound_ids:
            results_data['MOL_ID'] = valid_compound_ids
            # Reorder columns to have MOL_ID after SMILES
            results = pd.DataFrame(results_data)
            cols = ['SMILES', 'MOL_ID', 'Predicted_Class'] + [col for col in results.columns if col not in ['SMILES', 'MOL_ID', 'Predicted_Class']]
            results = results[cols]
        else:
            results = pd.DataFrame(results_data)
        
        # Add probabilities for each class
        if return_probabilities and hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(X_in)
            if self.label_encoder is not None and hasattr(self.label_encoder, 'classes_'):
                class_labels = [str(c) for c in self.label_encoder.classes_]
            elif hasattr(self.model, 'classes_'):
                class_labels = [str(c) for c in self.model.classes_]
            else:
                class_labels = [str(i) for i in range(probabilities.shape[1])]

            for i, class_name in enumerate(class_labels):
                results[f'Prob_{class_name}'] = probabilities[:, i]
        
        # Report errors
        if errors:
            print(f"\n[WARNING] {len(errors)} SMILES had errors:")
            for smiles, error in errors[:5]:
                print(f"  {smiles}: {error}")
            if len(errors) > 5:
                print(f"  ...and {len(errors) - 5} more")
        
        return results, valid_mols
    
    def screen_blind_set(self, smiles_list, output_dir="blind_screening", 
                        target_class='active', confidence_threshold=0.7):
        """
        Screen a blind set and identify high-efficiency compounds
        
        Parameters:
        -----------
        smiles_list : list or str (path to file)
            Either a list of SMILES or path to CSV/text file
        output_dir : str
            Directory to save results
        target_class : str
            Target activity class to prioritize (e.g., 'active', 'high')
        confidence_threshold : float
            Minimum probability threshold for high-confidence predictions
        
        Returns:
        --------
        results_df : DataFrame
            Screening results sorted by target class probability
        """
        
        # Load SMILES if file path provided
        compound_ids = None
        if isinstance(smiles_list, str) and os.path.isfile(smiles_list):
            print(f"Loading compounds from {smiles_list}...")
            if smiles_list.endswith('.csv'):
                # Try reading with tab separator first, then comma
                try:
                    df = pd.read_csv(smiles_list, sep='\t', header=None)
                    if len(df.columns) >= 2:
                        # Has SMILES and ID columns
                        smiles_list = df[0].tolist()
                        compound_ids = df[1].tolist()
                    else:
                        smiles_list = df[0].tolist()
                except:
                    # Fall back to comma-separated
                    df = pd.read_csv(str(smiles_list))
                    smiles_col = next((col for col in df.columns if 'smiles' in col.lower()), df.columns[0])
                    smiles_list = df[smiles_col].tolist()
                    # Try to get ID column if exists
                    id_cols = [col for col in df.columns if 'id' in col.lower() or 'mol' in col.lower()]
                    if id_cols:
                        compound_ids = df[id_cols[0]].tolist()
            else:
                with open(smiles_list, 'r') as f:
                    smiles_list = [line.strip() for line in f if line.strip()]
        
        print(f"\n{'='*60}")
        print(f"Screening {len(smiles_list)} compounds from blind set")
        print(f"{'='*60}")
        
        # Make predictions with compound IDs
        results, mols = self.predict(smiles_list, return_probabilities=True, compound_ids=compound_ids)
        
        if results is None:
            return None
        
        # Check if target class exists
        prob_col = f'Prob_{target_class}'
        if prob_col not in results.columns:
            available_prob_cols = [c.replace('Prob_', '') for c in results.columns if str(c).startswith('Prob_')]
            available_classes = ', '.join([str(c) for c in available_prob_cols]) if available_prob_cols else 'unknown'
            print(f"Error: Target class '{target_class}' not found.")
            print(f"Available classes: {available_classes}")
            return None
        
        # Sort by target class probability
        results = results.sort_values(prob_col, ascending=False)
        
        # Flag high-confidence predictions
        results['High_Confidence'] = results[prob_col] >= confidence_threshold
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save all results
        results.to_csv(f"{output_dir}/all_predictions.csv", index=False)
        
        # Filter and save high-efficiency compounds
        target_class_str = str(target_class)
        pred_class_str = results['Predicted_Class'].astype(str)
        high_efficiency = results[
            (pred_class_str == target_class_str) &
            (results['High_Confidence'])
        ]
        high_efficiency.to_csv(f"{output_dir}/high_efficiency_compounds.csv", index=False)
        
        # Generate summary statistics
        print(f"\n{'='*60}")
        print("Screening Results Summary")
        print(f"{'='*60}")
        print(f"\nTotal compounds screened: {len(results)}")
        print(f"\nPredicted class distribution:")
        if self.label_encoder is not None and hasattr(self.label_encoder, 'classes_'):
            class_names = [str(c) for c in self.label_encoder.classes_]
        elif hasattr(self.model, 'classes_'):
            class_names = [str(c) for c in self.model.classes_]
        else:
            class_names = sorted(results['Predicted_Class'].astype(str).unique().tolist())

        for class_name in class_names:
            count = int((results['Predicted_Class'].astype(str) == str(class_name)).sum())
            print(f"  {class_name}: {count} ({100*count/len(results):.1f}%)")
        
        print(f"\nHigh-confidence {target_class} compounds: {len(high_efficiency)}")
        print(f"  (Probability ≥ {confidence_threshold})")
        
        if len(high_efficiency) > 0:
            print(f"\nTop {min(5, len(high_efficiency))} high-efficiency compounds:")
            for rank, (_, row) in enumerate(high_efficiency.head(5).iterrows(), start=1):
                print(f"  {rank}. Probability: {row[prob_col]:.3f}")
        
        # Visualize results
        self._visualize_screening_results(results, high_efficiency, target_class, 
                                          prob_col, mols, output_dir)
        
        print(f"\n[OK] Results saved to {output_dir}/")
        
        return results
    
    def _visualize_screening_results(self, results, high_efficiency, target_class, 
                                     prob_col, mols, output_dir):
        """Generate visualizations for screening results"""
        
        # 1. Distribution of predicted probabilities
        plt.figure(figsize=(10, 6))
        plt.hist(results[prob_col], bins=30, edgecolor='black', alpha=0.7)
        plt.axvline(0.7, color='red', linestyle='--', label='Confidence threshold')
        plt.xlabel(f'Predicted Probability (Class {target_class})')
        plt.ylabel('Number of Compounds')
        plt.title(f'Distribution of Class {target_class} Probabilities')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/probability_distribution.png", dpi=300)
        plt.close()
        
        # 2. Class distribution pie chart
        class_counts = results['Predicted_Class'].value_counts()
        plt.figure(figsize=(8, 8))
        plt.pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%',
                startangle=90, colors=plt.cm.Set3.colors)
        plt.title('Predicted Class Distribution')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/class_distribution.png", dpi=300)
        plt.close()
        
        # 3. Visualize top high-efficiency molecules
        if len(high_efficiency) > 0:
            top_n = min(20, len(high_efficiency))
            top_indices = high_efficiency.head(top_n).index.tolist()
            top_mols = [mols[i] for i in range(len(mols)) if i in top_indices]
            top_probs = high_efficiency.head(top_n)[prob_col].tolist()
            
            legends = [f"P(Class {target_class})={prob:.3f}" for prob in top_probs]
            
            img = Draw.MolsToGridImage(
                top_mols, molsPerRow=4, subImgSize=(300, 300),
                legends=legends
            )
            img.save(f"{output_dir}/top_high_efficiency_molecules.png")
            print(f"  - Saved visualization of top {top_n} molecules")
    
    def compare_compounds(self, smiles_list, labels=None, output_path="compound_comparison.png"):
        """
        Compare predictions for a set of compounds side-by-side
        
        Parameters:
        -----------
        smiles_list : list
            List of SMILES to compare
        labels : list, optional
            Custom labels for each compound
        output_path : str
            Path to save comparison image
        """
        
        results, mols = self.predict(smiles_list, return_probabilities=True)
        
        if results is None:
            return None
        
        # Create legends with predictions
        legends = []
        for rank, (_, row) in enumerate(results.iterrows()):
            pred_class = row['Predicted_Class']
            prob_col = f'Prob_{pred_class}'
            prob = row[prob_col]

            if labels and rank < len(labels):
                legend = f"{labels[rank]}\n{pred_class} ({prob:.2f})"
            else:
                legend = f"Compound {rank+1}\n{pred_class} ({prob:.2f})"
            legends.append(legend)
        
        # Generate image
        if mols is None:
            return results
        img = Draw.MolsToGridImage(
            mols, molsPerRow=min(3, len(mols)), subImgSize=(300, 300),
            legends=legends
        )
        img.save(output_path)
        print(f"Comparison saved to {output_path}")
        
        return results


def load_classifier(model_dir, model_type='rf'):
    """
    Convenience function to load a trained classifier
    
    Parameters:
    -----------
    model_dir : str
        Directory containing model files
    model_type : str
        Model type ('rf' or 'xgb')
    
    Returns:
    --------
    classifier : ActivityClassifier
        Loaded classifier ready for prediction
    """
    
    classifier_model_path = f"{model_dir}/{model_type}_classifier.pkl"
    generic_model_path = f"{model_dir}/{model_type}_model.pkl"
    model_path = classifier_model_path if os.path.exists(classifier_model_path) else generic_model_path

    def _first_existing(candidates):
        for p in candidates:
            if p and os.path.exists(p):
                return p
        return None

    scaler_path = _first_existing([
        f"{model_dir}/{model_type}_scaler.pkl",
        f"{model_dir}/scaler.pkl",
    ])
    encoder_path = _first_existing([
        f"{model_dir}/{model_type}_label_encoder.pkl",
        f"{model_dir}/label_encoder.pkl",
    ])
    features_path = _first_existing([
        f"{model_dir}/{model_type}_feature_names.pkl",
        f"{model_dir}/feature_names.pkl",
        f"{model_dir}/selected_features.txt",
    ])
    
    # Determine if using fingerprints or descriptors
    use_fingerprints = 'fingerprints' in model_dir or 'fingerprint' in model_dir
    
    classifier = ActivityClassifier(
        model_path, scaler_path, encoder_path, features_path,
        use_fingerprints=use_fingerprints
    )
    
    return classifier


if __name__ == "__main__":
    # Example usage
    
    print("="*60)
    print("Activity Classification - Blind Set Screening")
    print("="*60)
    
    # 1. Load trained classifier
    # Update these paths to match your trained model location
    model_dir = f"models/{timestamp}/classification"
    
    try:
        classifier = load_classifier(model_dir, model_type='rf')
        
        # 2. Example: Predict activity for a few test compounds
        print("\n" + "="*60)
        print("Example 1: Single compound prediction")
        print("="*60)
        
        test_smiles = [
            "CCN(CC)C(=O)Nc1ccc(C)c(Nc2nccc(n2)c3cccnc3)c1",  # Imatinib
            "CCN(CC)CCNC(=O)c1c(C)[nH]c(C=C2C(=O)Nc3ccc(F)cc23)c1C",  # Sunitinib
        ]
        
        results, mols = classifier.predict(test_smiles)
        print("\nPredictions:")
        print(results)
        
        # 3. Example: Screen a blind set
        print("\n" + "="*60)
        print("Example 2: Blind set screening")
        print("="*60)
        
        # Option A: From a file
        blind_set_file = "data/blind_set_smiles.csv"
        
        if os.path.exists(blind_set_file):
            screening_results = classifier.screen_blind_set(
                blind_set_file,
                output_dir=f"predictions/{timestamp}/blind_screening",
                target_class='active',  # or 'high' depending on your classes
                confidence_threshold=0.7
            )
        else:
            print(f"\nBlind set file not found: {blind_set_file}")
            print("To screen a blind set, create a CSV file with a 'SMILES' column")
            
            # Option B: From a list
            print("\nScreening example compounds instead...")
            example_blind_set = [
                "Cc1ccc(cc1)NC(=O)c2ccc(CN3CCN(C)CC3)cc2",
                "CN1CCN(Cc2ccc(NC(=O)c3cccc(C)c3)cc2)CC1",
                "CCN(CC)C(=O)Nc1ccc(C)c(Nc2nccc(n2)c3cccnc3)c1"
            ]
            
            screening_results = classifier.screen_blind_set(
                example_blind_set,
                output_dir=f"predictions/{timestamp}/example_screening",
                target_class='active',
                confidence_threshold=0.7
            )
        
        print("\n[OK] Screening complete!")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease train a classification model first using classification_model.py")
        print(f"Expected model files in: {model_dir}/")

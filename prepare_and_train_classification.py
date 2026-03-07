"""
Complete pipeline for preparing known compounds data and training classification models
Processes chembl_all_class.csv with additional activity data (IC50, Kd, Ki, Inhibition)
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Lipinski, MolSurf, GraphDescriptors
from rdkit.Chem import rdPartialCharges, rdMolDescriptors
import os
from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d")

# Descriptor calculation functions
def compute_max_partial_charge(mol):
    """Compute the maximum partial charge for a molecule"""
    try:
        rdPartialCharges.ComputeGasteigerCharges(mol)
        charges = [float(atom.GetProp('_GasteigerCharge')) for atom in mol.GetAtoms()]
        return max(charges) if charges else 0.0
    except:
        return 0.0

def compute_min_partial_charge(mol):
    """Compute the minimum partial charge for a molecule"""
    try:
        rdPartialCharges.ComputeGasteigerCharges(mol)
        charges = [float(atom.GetProp('_GasteigerCharge')) for atom in mol.GetAtoms()]
        return min(charges) if charges else 0.0
    except:
        return 0.0

def compute_max_abs_partial_charge(mol):
    """Compute the maximum absolute partial charge for a molecule"""
    try:
        rdPartialCharges.ComputeGasteigerCharges(mol)
        charges = [abs(float(atom.GetProp('_GasteigerCharge'))) for atom in mol.GetAtoms()]
        return max(charges) if charges else 0.0
    except:
        return 0.0


def load_and_merge_data(base_dir='known_compounds'):
    """
    Load all ChEMBL data files and merge them
    
    Returns:
    --------
    merged_df : DataFrame
        Combined dataset with all activity information
    """
    
    print("="*60)
    print("Loading ChEMBL Data Files")
    print("="*60)
    
    # Load main classification data
    all_class_path = os.path.join(base_dir, 'chembl_all_class.csv')
    print(f"\nLoading {all_class_path}...")
    df_all = pd.read_csv(all_class_path, sep='\t')
    
    # Rename columns for consistency
    df_all.columns = ['Smiles', 'molecule_chembl_id', 'activity_class']
    print(f"  Loaded {len(df_all)} compounds with activity classes")
    print(f"  Class distribution: {df_all['activity_class'].value_counts().to_dict()}")
    
    # Load activity data files
    activity_files = {
        'IC50': os.path.join(base_dir, 'chembl_IC50.csv'),
        'Kd': os.path.join(base_dir, 'chembl_Kd.csv'),
        'Ki': os.path.join(base_dir, 'chembl_Ki.csv'),
        'Inhibition': os.path.join(base_dir, 'chembl_Inhibition.csv')
    }
    
    activity_dfs = {}
    for activity_type, filepath in activity_files.items():
        if os.path.exists(filepath):
            print(f"\nLoading {activity_type} data from {filepath}...")
            df = pd.read_csv(filepath, sep='\t')
            df.columns = ['Smiles', 'molecule_chembl_id', 'Standard Type', 
                         'activity', 'Standard Units', 'activity_sd', 'p_activity']
            print(f"  Loaded {len(df)} {activity_type} measurements")
            
            # For p_activity values
            if activity_type != 'Inhibition':
                df[f'{activity_type}_pActivity'] = df['p_activity']
            else:
                df[f'{activity_type}_percent'] = df['activity']
            
            activity_dfs[activity_type] = df
    
    # Merge all dataframes
    print("\n" + "="*60)
    print("Merging datasets...")
    print("="*60)
    
    merged = df_all.copy()
    
    # Merge each activity type
    for activity_type, df in activity_dfs.items():
        merge_cols = ['molecule_chembl_id', 'Smiles']
        
        if activity_type != 'Inhibition':
            # For IC50, Kd, Ki - use p_activity
            value_col = f'{activity_type}_pActivity'
            df_subset = df[merge_cols + ['p_activity']].copy()
            df_subset.rename(columns={'p_activity': value_col}, inplace=True)
        else:
            # For Inhibition - use percent
            value_col = f'{activity_type}_percent'
            df_subset = df[merge_cols + ['activity']].copy()
            df_subset.rename(columns={'activity': value_col}, inplace=True)
        
        # Remove duplicates by taking mean
        df_subset = df_subset.groupby(merge_cols, as_index=False).mean()
        
        merged = merged.merge(df_subset, on=merge_cols, how='left')
        
        # Report merge statistics
        n_matched = merged[value_col].notna().sum()
        print(f"  {activity_type}: {n_matched} compounds with data")
    
    print(f"\nFinal merged dataset: {len(merged)} compounds")
    print(f"Columns: {merged.columns.tolist()}")
    
    return merged


def generate_molecular_descriptors(df, smiles_col='Smiles'):
    """
    Generate molecular descriptors for all compounds
    
    Parameters:
    -----------
    df : DataFrame
        Input dataframe with SMILES
    smiles_col : str
        Name of SMILES column
    
    Returns:
    --------
    df_with_descriptors : DataFrame
        DataFrame with added molecular descriptors
    """
    
    print("\n" + "="*60)
    print("Generating Molecular Descriptors")
    print("="*60)
    
    # Define descriptors
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
        'MaxPartialCharge': compute_max_partial_charge,
        'MinPartialCharge': compute_min_partial_charge,
        'MaxAbsPartialCharge': compute_max_abs_partial_charge,
        'NumRings': Descriptors.RingCount,
        'AromaticRings': rdMolDescriptors.CalcNumAromaticRings,
        'AromaticHetero': rdMolDescriptors.CalcNumAromaticHeterocycles,
        'AromaticCarbocycles': rdMolDescriptors.CalcNumAromaticCarbocycles
    }
    
    desc_names = list(descriptor_functions.keys())
    print(f"Calculating {len(desc_names)} descriptors for {len(df)} compounds...")
    
    # Calculate descriptors
    descriptors_list = []
    valid_indices = []
    
    for idx, smiles in enumerate(df[smiles_col]):
        if idx % 100 == 0:
            print(f"  Progress: {idx}/{len(df)}", end='\r')
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            
            desc_values = []
            for desc_name in desc_names:
                try:
                    func = descriptor_functions[desc_name]
                    desc_values.append(func(mol))
                except:
                    desc_values.append(0.0)
            
            descriptors_list.append(desc_values)
            valid_indices.append(idx)
            
        except Exception as e:
            continue
    
    print(f"  Progress: {len(df)}/{len(df)}")
    print(f"\nSuccessfully calculated descriptors for {len(valid_indices)} compounds")
    print(f"Failed for {len(df) - len(valid_indices)} compounds")
    
    # Create descriptor dataframe
    desc_df = pd.DataFrame(descriptors_list, columns=desc_names)
    
    # Replace infinity and NaN values
    desc_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    desc_df.fillna(0, inplace=True)
    
    # Combine with original data
    result = df.iloc[valid_indices].reset_index(drop=True)
    result = pd.concat([result, desc_df], axis=1)
    
    # Final check for any remaining problematic values
    numeric_cols = desc_df.columns
    for col in numeric_cols:
        if result[col].isnull().any() or np.isinf(result[col]).any():
            print(f"  Warning: Cleaning column {col}")
            result[col].replace([np.inf, -np.inf], 0, inplace=True)
            result[col].fillna(0, inplace=True)
    
    return result


def generate_fingerprints(df, smiles_col='Smiles', radius=2, nBits=2048):
    """
    Generate Morgan fingerprints for all compounds
    
    Parameters:
    -----------
    df : DataFrame
        Input dataframe with SMILES
    smiles_col : str
        Name of SMILES column
    radius : int
        Fingerprint radius
    nBits : int
        Number of bits in fingerprint
    
    Returns:
    --------
    df_with_fingerprints : DataFrame
        DataFrame with added fingerprints
    """
    
    print("\n" + "="*60)
    print("Generating Molecular Fingerprints")
    print("="*60)
    print(f"Parameters: radius={radius}, nBits={nBits}")
    
    fingerprints_list = []
    valid_indices = []
    
    for idx, smiles in enumerate(df[smiles_col]):
        if idx % 100 == 0:
            print(f"  Progress: {idx}/{len(df)}", end='\r')
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
            fingerprints_list.append(list(fp))
            valid_indices.append(idx)
            
        except Exception as e:
            continue
    
    print(f"  Progress: {len(df)}/{len(df)}")
    print(f"\nSuccessfully generated fingerprints for {len(valid_indices)} compounds")
    
    # Create fingerprint dataframe
    fp_columns = [f'bit_{i}' for i in range(nBits)]
    fp_df = pd.DataFrame(fingerprints_list, columns=fp_columns)
    
    # Combine with original data
    result = df.iloc[valid_indices].reset_index(drop=True)
    result = pd.concat([result, fp_df], axis=1)
    
    return result


def prepare_classification_dataset(use_fingerprints=False):
    """
    Complete pipeline to prepare classification dataset
    
    Parameters:
    -----------
    use_fingerprints : bool
        If True, generate fingerprints; if False, generate descriptors
    
    Returns:
    --------
    output_path : str
        Path to saved dataset
    """
    
    print("\n" + "="*80)
    print("CHEMBL CLASSIFICATION DATA PREPARATION PIPELINE")
    print("="*80)
    
    # Create output directory
    output_dir = f"data/processed/{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Load and merge data
    merged_df = load_and_merge_data('known_compounds')
    
    # Save merged data
    merged_path = os.path.join(output_dir, 'chembl_merged_all_data.csv')
    merged_df.to_csv(merged_path, index=False)
    print(f"\n✓ Saved merged data to {merged_path}")
    
    # Step 2: Generate features
    if use_fingerprints:
        print("\nGenerating fingerprints...")
        final_df = generate_fingerprints(merged_df)
        output_filename = 'chembl_classification_fingerprints.csv'
    else:
        print("\nGenerating descriptors...")
        final_df = generate_molecular_descriptors(merged_df)
        output_filename = 'chembl_classification_descriptors.csv'
    
    # Convert canonical_smiles column name if needed
    if 'Smiles' in final_df.columns:
        final_df.rename(columns={'Smiles': 'canonical_smiles'}, inplace=True)
    
    # Save final dataset
    output_path = os.path.join(output_dir, output_filename)
    final_df.to_csv(output_path, index=False)
    
    print("\n" + "="*80)
    print("DATASET PREPARATION COMPLETE")
    print("="*80)
    print(f"\n✓ Final dataset saved to: {output_path}")
    print(f"  Total compounds: {len(final_df)}")
    print(f"  Features: {len(final_df.columns) - len(['canonical_smiles', 'molecule_chembl_id', 'activity_class'])}")
    print(f"\nActivity class distribution:")
    for cls, count in final_df['activity_class'].value_counts().items():
        print(f"  Class {cls}: {count} compounds ({100*count/len(final_df):.1f}%)")
    
    # Show available activity data coverage
    print(f"\nActivity data coverage:")
    activity_cols = [col for col in final_df.columns if 'IC50' in col or 'Kd' in col or 'Ki' in col or 'Inhibition' in col]
    for col in activity_cols:
        n_available = final_df[col].notna().sum()
        print(f"  {col}: {n_available} compounds ({100*n_available/len(final_df):.1f}%)")
    
    return output_path


def train_models_on_prepared_data(data_path, optimize=False):
    """
    Train classification models on the prepared dataset
    
    Parameters:
    -----------
    data_path : str
        Path to prepared dataset CSV
    optimize : bool
        Whether to perform hyperparameter optimization
    """
    
    print("\n" + "="*80)
    print("TRAINING CLASSIFICATION MODELS")
    print("="*80)
    
    # Import the classification model builder
    from classification_model import build_classification_model
    
    output_dir = f"models/{timestamp}/classification"
    os.makedirs(output_dir, exist_ok=True)
    
    models = {}
    
    # Train Random Forest
    print("\n" + "="*60)
    print("Training Random Forest Classifier")
    print("="*60)
    try:
        rf_model, rf_scaler, rf_encoder, rf_metrics = build_classification_model(
            data_path,
            activity_column='activity_class',
            model_type='rf',
            optimize=optimize,
            output_dir=output_dir
        )
        if rf_metrics:
            models['Random Forest'] = rf_metrics
    except Exception as e:
        print(f"Error training Random Forest: {e}")
    
    # Train XGBoost
    print("\n" + "="*60)
    print("Training XGBoost Classifier")
    print("="*60)
    try:
        xgb_model, xgb_scaler, xgb_encoder, xgb_metrics = build_classification_model(
            data_path,
            activity_column='activity_class',
            model_type='xgb',
            optimize=optimize,
            output_dir=output_dir
        )
        if xgb_metrics:
            models['XGBoost'] = xgb_metrics
    except Exception as e:
        print(f"Error training XGBoost: {e}")
    
    # Compare models
    if models:
        print("\n" + "="*80)
        print("MODEL COMPARISON SUMMARY")
        print("="*80)
        comparison_df = pd.DataFrame(models).T
        print(comparison_df)
        comparison_df.to_csv(f"{output_dir}/model_comparison.csv")
        
        print(f"\n✓ All models trained successfully!")
        print(f"✓ Results saved to {output_dir}/")
        
        return output_dir
    else:
        print("\n❌ No models were successfully trained")
        return None


if __name__ == "__main__":
    print("""
    ===================================================================
       ChEMBL Classification Model Training Pipeline
       Processes known compounds with activity classes
    ===================================================================
    """)
    
    # Configuration
    USE_FINGERPRINTS = False  # Set to True for fingerprints, False for descriptors
    OPTIMIZE_HYPERPARAMETERS = False  # Set to True for better performance (slower)
    
    print(f"\nConfiguration:")
    print(f"  Feature type: {'Fingerprints' if USE_FINGERPRINTS else 'Descriptors'}")
    print(f"  Hyperparameter optimization: {'Enabled' if OPTIMIZE_HYPERPARAMETERS else 'Disabled'}")
    
    # Step 1: Prepare dataset
    print("\n" + "="*80)
    print("STEP 1: Data Preparation")
    print("="*80)
    
    data_path = prepare_classification_dataset(use_fingerprints=USE_FINGERPRINTS)
    
    # Step 2: Train models
    print("\n" + "="*80)
    print("STEP 2: Model Training")
    print("="*80)
    
    model_dir = train_models_on_prepared_data(data_path, optimize=OPTIMIZE_HYPERPARAMETERS)
    
    if model_dir:
        print("\n" + "="*80)
        print("✓ PIPELINE COMPLETE!")
        print("="*80)
        print(f"\nOutputs:")
        print(f"  1. Prepared data: {data_path}")
        print(f"  2. Trained models: {model_dir}")
        print(f"\nNext steps:")
        print(f"  - Review model performance in {model_dir}")
        print(f"  - Use classification_prediction_tool.py to screen blind sets")
        print(f"  - Check confusion matrices and ROC curves for model quality")
    else:
        print("\n❌ Pipeline failed during model training")

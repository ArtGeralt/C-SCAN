import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import os
from datetime import datetime

timestamp = ""

def generate_fingerprints(input_csv, output_csv=None, radius=2, nBits=2048):
    """Generate Morgan fingerprints for compounds in a CSV file"""
    print(f"Loading data from {input_csv}")
    df = pd.read_csv(input_csv)
    
    print(f"Generating Morgan fingerprints for {len(df)} compounds...")
    
    # Function to calculate fingerprints for a molecule
    def calc_fingerprint(smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return [0] * nBits
            fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
            return list(fingerprint)
        except Exception as e:
            print(f"Error with SMILES {smiles}: {e}")
            return [0] * nBits
    
    # Calculate fingerprints for each molecule
    fingerprints = []
    for smiles in df['canonical_smiles']:
        fingerprints.append(calc_fingerprint(smiles))
    
    # Convert to DataFrame with bit column names
    fp_columns = [f'bit_{i}' for i in range(nBits)]
    fp_df = pd.DataFrame(fingerprints, columns=fp_columns)
    
    # Combine with original data (only keep molecule ID, SMILES and pIC50)
    result = pd.concat([
        df[['molecule_chembl_id', 'canonical_smiles', 'pIC50']], 
        fp_df
    ], axis=1)
    
    # Remove rows with invalid fingerprints (all zeros)
    orig_len = len(result)
    result = result[(result[fp_columns] != 0).any(axis=1)]
    print(f"Removed {orig_len - len(result)} compounds with invalid structures")
    
    # Save results
    if output_csv is None:
        output_csv = input_csv.replace('.csv', '_fingerprints.csv')
    
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    result.to_csv(output_csv, index=False)
    print(f"Saved {len(result)} compounds with {nBits} fingerprint bits to {output_csv}")
    return result

if __name__ == "__main__":
    # Path to your original KIT data file
    input_file = "data/processed/20251006/kit_pic50_20251006.csv"  
    
    # If the file doesn't exist with this name, look for similar files
    if not os.path.exists(input_file):
        print(f"File not found: {input_file}")
        processed_dir = "data/processed"
        if os.path.exists(processed_dir):
            files = os.listdir(processed_dir)
            csv_files = [f for f in files if f.endswith('.csv') and 'fingerprint' not in f]
            if csv_files:
                print(f"Found alternative files: {csv_files}")
                input_file = os.path.join(processed_dir, csv_files[0])
                print(f"Using: {input_file}")
            else:
                print("No suitable CSV files found in data/processed directory")
        else:
            print(f"Directory not found: {processed_dir}")
            os.makedirs(processed_dir, exist_ok=True)
            print("Created directory. Please place your data file there.")
            exit(1)
    
    # Generate fingerprints
    output_file = f"data/processed/{timestamp}/kit_fingerprints.csv"
    generate_fingerprints(input_file, output_file)
    
    print("\nFingerprint generation complete. You can now run fingerprints_clustering.py")
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import os
from datetime import datetime
from typing import Callable, Optional

timestamp = ""


def _detect_smiles_column(df: pd.DataFrame):
    preferred = ["canonical_smiles", "SMILES", "smiles", "Smiles"]
    for col in preferred:
        if col in df.columns:
            return col
    for col in df.columns:
        if "smiles" in str(col).lower():
            return col
    return None


def _parse_smiles_with_fallback(raw_value):
    if pd.isna(raw_value):
        return None, None

    s = str(raw_value).strip().strip('"').strip("'")
    if not s:
        return None, None

    mol = Chem.MolFromSmiles(s)
    if mol is not None:
        return s, mol

    parts = []
    for sep in ["\t", ",", ";", " "]:
        if sep in s:
            parts.extend([p.strip().strip('"').strip("'") for p in s.split(sep) if p.strip()])
    if parts:
        for token in sorted(set(parts), key=len, reverse=True):
            mol = Chem.MolFromSmiles(token)
            if mol is not None:
                return token, mol

    return None, None

def generate_fingerprints(
    input_csv,
    output_csv=None,
    radius=2,
    nBits=2048,
    progress_callback: Optional[Callable[[dict], None]] = None,
    progress_step: int = 500,
):
    """Generate Morgan fingerprints for compounds in a CSV file"""

    def _emit(stage, current, total, message=None, **extra):
        if progress_callback is None:
            return
        payload = {
            "stage": stage,
            "current": int(current),
            "total": int(total),
            "message": message or "",
        }
        payload.update(extra)
        progress_callback(payload)

    print(f"Loading data from {input_csv}")
    df = pd.read_csv(input_csv)
    _emit("load", 1, 1, f"Loaded input table with {len(df)} rows")
    
    print(f"Generating Morgan fingerprints for {len(df)} compounds...")

    smiles_col = _detect_smiles_column(df)
    if smiles_col is None:
        raise ValueError(
            "No SMILES column found. Expected one of: canonical_smiles, SMILES, smiles, Smiles"
        )
    if smiles_col != 'canonical_smiles':
        print(f"Using SMILES column: {smiles_col}")
    
    # Function to calculate fingerprints for a parsed molecule
    def calc_fingerprint_from_mol(mol):
        try:
            if mol is None:
                return [0] * nBits
            fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
            return list(fingerprint)
        except Exception as e:
            print(f"Error while calculating fingerprint: {e}")
            return [0] * nBits
    
    # Calculate fingerprints for each molecule with robust SMILES parsing
    fingerprints = []
    cleaned_smiles = []
    valid_count = 0
    total_rows = len(df)
    for idx, raw in enumerate(df[smiles_col], start=1):
        parsed_smiles, mol = _parse_smiles_with_fallback(raw)
        cleaned_smiles.append(parsed_smiles if parsed_smiles is not None else np.nan)
        if mol is not None:
            valid_count += 1
        fingerprints.append(calc_fingerprint_from_mol(mol))

        if idx % max(1, progress_step) == 0 or idx == total_rows:
            _emit(
                "fingerprints",
                idx,
                total_rows,
                f"Processed {idx}/{total_rows} compounds",
                valid=valid_count,
                invalid=(idx - valid_count),
            )

    if 'canonical_smiles' not in df.columns:
        df = df.copy()
        df['canonical_smiles'] = cleaned_smiles
    else:
        df = df.copy()
        df['canonical_smiles'] = cleaned_smiles

    if 'molecule_chembl_id' not in df.columns:
        df['molecule_chembl_id'] = [f"MOL_{i+1}" for i in range(len(df))]
    if 'pIC50' not in df.columns:
        df['pIC50'] = np.nan
    
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
    _emit(
        "cleanup",
        len(result),
        orig_len,
        f"Valid structures retained: {len(result)}/{orig_len}",
    )

    if len(result) == 0:
        raise ValueError(
            "No valid structures remained after SMILES parsing for fingerprint generation. "
            "Check that the selected SMILES column contains valid SMILES strings."
        )
    
    # Save results
    if output_csv is None:
        output_csv = input_csv.replace('.csv', '_fingerprints.csv')
    
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    result.to_csv(output_csv, index=False)
    print(f"Saved {len(result)} compounds with {nBits} fingerprint bits to {output_csv}")
    _emit("save", 1, 1, f"Saved fingerprints to {output_csv}")
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
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, MolSurf, GraphDescriptors
from rdkit.Chem import rdPartialCharges, rdMolDescriptors
from datetime import datetime
from typing import Callable, Optional

timestamp = ""


def _detect_smiles_column(df: pd.DataFrame):
    """Return best-effort SMILES column name from common variants."""
    preferred = ["canonical_smiles", "SMILES", "smiles", "Smiles"]
    for col in preferred:
        if col in df.columns:
            return col
    for col in df.columns:
        if "smiles" in str(col).lower():
            return col
    return None


def _parse_smiles_with_fallback(raw_value):
    """Parse SMILES from raw cell; supports mixed tokens like 'MOL_1 CCO'."""
    if pd.isna(raw_value):
        return None, None

    s = str(raw_value).strip().strip('"').strip("'")
    if not s:
        return None, None

    mol = Chem.MolFromSmiles(s)
    if mol is not None:
        return s, mol

    # Try tokenized fallback when ID/extra fields are bundled with SMILES.
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

def compute_max_partial_charge(mol):
    """Compute the maximum partial charge for a molecule"""
    rdPartialCharges.ComputeGasteigerCharges(mol)
    charges = [float(atom.GetProp('_GasteigerCharge')) for atom in mol.GetAtoms()]
    if charges:
        return max(charges)
    return 0.0

def compute_min_partial_charge(mol):
    """Compute the minimum partial charge for a molecule"""
    rdPartialCharges.ComputeGasteigerCharges(mol)
    charges = [float(atom.GetProp('_GasteigerCharge')) for atom in mol.GetAtoms()]
    if charges:
        return min(charges)
    return 0.0

def compute_max_abs_partial_charge(mol):
    """Compute the maximum absolute partial charge for a molecule"""
    rdPartialCharges.ComputeGasteigerCharges(mol)
    charges = [abs(float(atom.GetProp('_GasteigerCharge'))) for atom in mol.GetAtoms()]
    if charges:
        return max(charges)
    return 0.0

def generate_descriptors(
    input_csv,
    output_csv=None,
    selected_only=True,
    descriptor_calculators=None,
    progress_callback: Optional[Callable[[dict], None]] = None,
    progress_step: int = 500,
):
    """Generate molecular descriptors for compounds in a CSV file"""

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
    
    print(f"Generating descriptors for {len(df)} compounds...")

    smiles_col = _detect_smiles_column(df)
    if smiles_col is None:
        raise ValueError(
            "No SMILES column found. Expected one of: canonical_smiles, SMILES, smiles, Smiles"
        )
    if smiles_col != 'canonical_smiles':
        print(f"Using SMILES column: {smiles_col}")
    
    # Define specific descriptors of interest
    if descriptor_calculators is not None:
        # Explicit descriptor set passed by caller (e.g., Streamlit UI selection)
        selected_descriptors = descriptor_calculators
        desc_names = list(selected_descriptors.keys())
    elif selected_only:
        # Common QSAR descriptors - customize this list based on your needs
        selected_descriptors = {
            # Lipinski properties
            'MolWt': Descriptors.MolWt,
            'MolLogP': Descriptors.MolLogP,
            'NumHDonors': Lipinski.NumHDonors,
            'NumHAcceptors': Lipinski.NumHAcceptors,
            
            # Topological properties
            'TPSA': MolSurf.TPSA,
            'NumRotatableBonds': Descriptors.NumRotatableBonds,
            'NumAromaticRings': Lipinski.NumAromaticRings,  # Changed to Lipinski
            'NumAliphaticRings': Lipinski.NumAliphaticRings,  # Changed to Lipinski
            
            # Structural features
            'FractionCSP3': Descriptors.FractionCSP3,
            'NumHeteroatoms': Descriptors.NumHeteroatoms,
            'RingCount': Descriptors.RingCount,
            
            # Electronic/surface properties
            'LabuteASA': Descriptors.LabuteASA,
            'SlogP_VSA1': MolSurf.SlogP_VSA1,
            'SMR_VSA1': MolSurf.SMR_VSA1,
            
            # Size and shape
            'BalabanJ': Descriptors.BalabanJ,
            'BertzCT': Descriptors.BertzCT,
            
            # Extended surface properties
            'PEOE_VSA1': MolSurf.PEOE_VSA1,
            'PEOE_VSA2': MolSurf.PEOE_VSA2,
            'PEOE_VSA3': MolSurf.PEOE_VSA3,
            'SlogP_VSA2': MolSurf.SlogP_VSA2,
            'SlogP_VSA3': MolSurf.SlogP_VSA3,
            'SMR_VSA2': MolSurf.SMR_VSA2,
            'SMR_VSA3': MolSurf.SMR_VSA3,
            
            # Topological and connectivity descriptors
            'Chi0': GraphDescriptors.Chi0,
            'Chi1': GraphDescriptors.Chi1,
            'Chi0v': GraphDescriptors.Chi0v,
            'Chi1v': GraphDescriptors.Chi1v,
            'Kappa1': GraphDescriptors.Kappa1,
            'Kappa2': GraphDescriptors.Kappa2,
            'Kappa3': GraphDescriptors.Kappa3,

            # Electrostatic properties - Using our custom functions
            'MaxPartialCharge': compute_max_partial_charge,
            'MinPartialCharge': compute_min_partial_charge,
            'MaxAbsPartialCharge': compute_max_abs_partial_charge,
            
            # Kinase-relevant features
            'NumHBA_Lipinski': Lipinski.NumHAcceptors,    # Alternative calculation
            'NumHBD_Lipinski': Lipinski.NumHDonors,       # Alternative calculation
            'NumRings': Descriptors.RingCount,            # Overall ring count
            'AromaticRings': rdMolDescriptors.CalcNumAromaticRings,  # Aromatic rings
            'AromaticHetero': rdMolDescriptors.CalcNumAromaticHeterocycles,  # Aromatic heterocycles
            'AromaticCarbocycles': rdMolDescriptors.CalcNumAromaticCarbocycles  # Aromatic carbocycles
        }
        desc_names = list(selected_descriptors.keys())
    else:
        # Use all available descriptors
        desc_names = [x[0] for x in Descriptors._descList]
        selected_descriptors = {name: getattr(Descriptors, name) for name in desc_names}
    
    print(f"Calculating {len(desc_names)} descriptors for each compound")
    _emit("setup", 1, 1, f"Calculating {len(desc_names)} descriptors per compound")
    
    # Function to calculate descriptors for a parsed molecule
    def calc_descriptors_from_mol(mol):
        try:
            if mol is None:
                return [np.nan] * len(desc_names)
            
            # Calculate each descriptor
            values = []
            for name in desc_names:
                calculator = selected_descriptors[name]
                try:
                    val = calculator(mol)
                    # Normalize invalid numeric outputs so they can be imputed later.
                    if val is None:
                        val = np.nan
                    elif isinstance(val, (float, np.floating)) and not np.isfinite(val):
                        val = np.nan
                    values.append(val)
                except Exception:
                    values.append(np.nan)
            return values
        except Exception as e:
            print(f"Error while calculating descriptors: {e}")
            return [np.nan] * len(desc_names)
    
    # Calculate descriptors for each molecule with robust SMILES parsing
    descriptors = []
    cleaned_smiles = []
    invalid_examples = []
    valid_count = 0
    total_rows = len(df)
    parse_valid_mask = []
    for idx, raw in enumerate(df[smiles_col], start=1):
        parsed_smiles, mol = _parse_smiles_with_fallback(raw)
        cleaned_smiles.append(parsed_smiles if parsed_smiles is not None else np.nan)
        if mol is None:
            parse_valid_mask.append(False)
            if len(invalid_examples) < 5:
                invalid_examples.append(str(raw))
            descriptors.append([np.nan] * len(desc_names))
        else:
            parse_valid_mask.append(True)
            valid_count += 1
            descriptors.append(calc_descriptors_from_mol(mol))

        if idx % max(1, progress_step) == 0 or idx == total_rows:
            _emit(
                "descriptors",
                idx,
                total_rows,
                f"Processed {idx}/{total_rows} compounds",
                valid=valid_count,
                invalid=(idx - valid_count),
            )

    # Ensure downstream modules always see canonical_smiles.
    if 'canonical_smiles' not in df.columns:
        df = df.copy()
        df['canonical_smiles'] = cleaned_smiles
    else:
        df = df.copy()
        df['canonical_smiles'] = cleaned_smiles
        
    # Convert to DataFrame
    desc_df = pd.DataFrame(descriptors, columns=desc_names)
    
    # Combine with original data
    result = pd.concat([df, desc_df], axis=1)
    
    # Drop only truly invalid structures (failed SMILES parsing).
    orig_len = len(result)
    valid_struct_mask = pd.Series(parse_valid_mask, index=result.index)
    result = result[valid_struct_mask & result['canonical_smiles'].notna()].copy()

    removed_invalid = orig_len - len(result)
    print(f"Removed {removed_invalid} compounds with invalid structures")

    # Descriptor calculations may still produce sparse NaNs; impute to keep valid molecules.
    nan_cells_before = int(result[desc_names].isna().sum().sum()) if len(result) > 0 else 0
    if len(result) > 0 and nan_cells_before > 0:
        medians = result[desc_names].median(numeric_only=True)
        result.loc[:, desc_names] = result[desc_names].fillna(medians).fillna(0.0)

    nan_cells_after = int(result[desc_names].isna().sum().sum()) if len(result) > 0 else 0
    _emit(
        "cleanup",
        len(result),
        orig_len,
        (
            f"Valid structures retained: {len(result)}/{orig_len}; "
            f"descriptor NaN cells filled: {nan_cells_before - nan_cells_after}"
        ),
        dropped_invalid=removed_invalid,
        nan_filled=(nan_cells_before - nan_cells_after),
    )

    if len(result) == 0:
        msg = (
            "No valid structures remained after SMILES parsing. "
            "This means the SMILES column could not be parsed into valid molecules."
        )
        if invalid_examples:
            msg += " Sample invalid entries: " + " | ".join(invalid_examples)
        raise ValueError(msg)
    
    # Save results
    if output_csv is None:
        output_csv = input_csv.replace('.csv', '_with_descriptors.csv')
    
    result.to_csv(output_csv, index=False)
    print(f"Saved {len(result)} compounds with {len(desc_names)} descriptors to {output_csv}")
    _emit("save", 1, 1, f"Saved descriptors to {output_csv}")
    return result

if __name__ == "__main__":
    input_file = f"data/processed/{timestamp}/kit_pic50_{timestamp}.csv"
    output_file = f"data/processed/{timestamp}/kit_descriptors_selected.csv"
    
    # Generate only selected descriptors
    generate_descriptors(input_file, output_file, selected_only=True)
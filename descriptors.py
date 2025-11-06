import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, MolSurf, GraphDescriptors
from rdkit.Chem import rdPartialCharges, rdMolDescriptors
from datetime import datetime

timestamp = ""

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

def generate_descriptors(input_csv, output_csv=None, selected_only=True):
    """Generate molecular descriptors for compounds in a CSV file"""
    print(f"Loading data from {input_csv}")
    df = pd.read_csv(input_csv)
    
    print(f"Generating descriptors for {len(df)} compounds...")
    
    # Define specific descriptors of interest
    if selected_only:
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
    
    # Function to calculate descriptors for a molecule
    def calc_descriptors(smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return [np.nan] * len(desc_names)
            
            # Calculate each descriptor
            values = []
            for name in desc_names:
                calculator = selected_descriptors[name]
                values.append(calculator(mol))
            return values
        except Exception as e:
            print(f"Error with SMILES {smiles}: {e}")
            return [np.nan] * len(desc_names)
    
    # Calculate descriptors for each molecule
    descriptors = []
    for smiles in df['canonical_smiles']:
        descriptors.append(calc_descriptors(smiles))
        
    # Convert to DataFrame
    desc_df = pd.DataFrame(descriptors, columns=desc_names)
    
    # Combine with original data
    result = pd.concat([df, desc_df], axis=1)
    
    # Remove rows with NaN descriptors
    orig_len = len(result)
    result = result.dropna()
    print(f"Removed {orig_len - len(result)} compounds with invalid structures")
    
    # Save results
    if output_csv is None:
        output_csv = input_csv.replace('.csv', '_with_descriptors.csv')
    
    result.to_csv(output_csv, index=False)
    print(f"Saved {len(result)} compounds with {len(desc_names)} descriptors to {output_csv}")
    return result

if __name__ == "__main__":
    input_file = f"data/processed/{timestamp}/kit_pic50_{timestamp}.csv"
    output_file = f"data/processed/{timestamp}/kit_descriptors_selected.csv"
    
    # Generate only selected descriptors
    generate_descriptors(input_file, output_file, selected_only=True)
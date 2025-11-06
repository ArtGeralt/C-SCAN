from chembl_webresource_client.new_client import new_client
import pandas as pd
import numpy as np
import os
import time
import pickle
from datetime import datetime
from tqdm import tqdm

def fetch_with_progress(target_id, data_type="IC50", units="nM", limit=None, cache_dir="data/cache"):
    """Fetch data from ChEMBL with progress indication and caching"""
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{target_id}_{data_type}_{units}.pkl")
    
    # Check if we have cached results
    if os.path.exists(cache_file):
        cache_age = time.time() - os.path.getmtime(cache_file)
        cache_days = cache_age / (60*60*24)
        print(f"Found cached data from {cache_days:.1f} days ago")
        if cache_days < 30:  # Use cache if less than 30 days old
            print("Using cached data...")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        else:
            print("Cache is old, fetching fresh data...")
    
    print(f"Fetching {data_type} data for target {target_id}...")
    start_time = time.time()
    
    # Build query - note: the limit method isn't available directly on the filter
    # We need to get all the data and then limit it afterwards
    query = new_client.activity.filter(
        target_chembl_id=target_id,
        standard_type=data_type,
        standard_units=units
    ).only(['canonical_smiles', 'standard_value', 'molecule_chembl_id'])
    
    # Show some progress feedback as this can be slow
    print("Downloading data...")
    print("This may take several minutes - ChEMBL API is processing the request")
    
    # Convert to list and then DataFrame
    activities_list = list(query)
    
    # Apply limit after retrieving if specified
    if limit and limit < len(activities_list):
        activities_list = activities_list[:limit]
    
    # Convert to DataFrame
    df = pd.DataFrame(activities_list)
    
    elapsed = time.time() - start_time
    print(f"Download completed in {elapsed:.1f} seconds")
    
    # Cache the results
    with open(cache_file, 'wb') as f:
        pickle.dump(df, f)
    
    return df

def remove_duplicates(df, verbose=True):
    """
    Remove duplicates from ChEMBL data in various ways:
    1. Exact duplicates (same molecule_chembl_id, canonical_smiles, and pIC50)
    2. Same molecule with different pIC50 values (keep the median)
    3. Alternative representations of the same compound (isomeric/tautomeric variants)
    """
    if verbose:
        print(f"Starting with {len(df)} compounds")
    
    # 1. Remove exact duplicates
    initial_count = len(df)
    df = df.drop_duplicates()
    if verbose:
        print(f"Removed {initial_count - len(df)} exact duplicates")
    
    # 2. Handle cases where the same molecule has different pIC50 values
    # Group by molecule_chembl_id and take the median pIC50
    grouped_by_id = df.groupby('molecule_chembl_id').agg({
        'canonical_smiles': 'first',  # Keep first SMILES
        'pIC50': 'median'             # Take median pIC50
    }).reset_index()
    
    if verbose:
        print(f"Found {initial_count - len(grouped_by_id)} compounds with multiple pIC50 values")
    
    # 3. Check for duplicate structures with different ChEMBL IDs
    # This is optional and requires RDKit to standardize molecules
    try:
        from rdkit import Chem
        from rdkit.Chem.MolStandardize import rdMolStandardize
        
        def standardize_smiles(smiles):
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    return smiles
                # Remove stereochemistry
                mol = rdMolStandardize.ChargeParent(mol)
                # Get canonical SMILES
                return Chem.MolToSmiles(mol, isomericSmiles=False)
            except:
                return smiles
        
        # Apply standardization
        grouped_by_id['std_smiles'] = grouped_by_id['canonical_smiles'].apply(standardize_smiles)
        
        # Group by standardized SMILES
        before_std = len(grouped_by_id)
        grouped_by_struct = grouped_by_id.groupby('std_smiles').agg({
            'molecule_chembl_id': 'first',
            'canonical_smiles': 'first',
            'pIC50': 'median'
        }).reset_index()
        
        if verbose:
            print(f"Found {before_std - len(grouped_by_struct)} compounds that are structural duplicates")
        
        # Drop the standardized SMILES column
        final_df = grouped_by_struct.drop(columns=['std_smiles'])
        
    except ImportError:
        if verbose:
            print("RDKit not available for structural duplicate detection")
        final_df = grouped_by_id
    
    if verbose:
        print(f"Final dataset contains {len(final_df)} unique compounds")
    
    return final_df

def main():
    # Create directories if they don't exist
    os.makedirs("data/processed", exist_ok=True)

    # Define target
    target_id = "CHEMBL1936"  # KIT (c-Kit, Mast/stem cell growth factor receptor)
    
    try:
        # Fetch data with progress indication and caching
        df = fetch_with_progress(target_id, limit=2000)  # Limit for testing; remove for production
        
        print("DataFrame columns:", df.columns.tolist())
        print(f"Retrieved {df.shape[0]} records")
        
        if df.empty:
            print("No data was returned from ChEMBL for this target.")
            return
            
        # Clean and filter data
        if 'standard_value' in df.columns and 'canonical_smiles' in df.columns:
            print("Cleaning and processing data...")
            # Filter for valid entries
            df = df[df['standard_value'].notna()]
            df = df[df['canonical_smiles'].notna()]
            
            # Convert to proper data type
            df['standard_value'] = df['standard_value'].astype(float)

            # Remove extreme or invalid IC50 values
            df = df[(df['standard_value'] > 0) & (df['standard_value'] < 1e7)]

            # Calculate pIC50 = -log10(IC50 [M])
            df['pIC50'] = -np.log10(df['standard_value'] * 1e-9)

            # Final dataframe with duplicate removal
            cleaned_df = df[['molecule_chembl_id', 'canonical_smiles', 'pIC50']].dropna()
            final_df = remove_duplicates(cleaned_df)
            
            # Save
            timestamp = "20251006"
            output_path = f"data/processed/{timestamp}/kit_pic50_{timestamp}.csv"
            final_df.to_csv(output_path, index=False)
            print(f"Processing complete. Saved {len(final_df)} entries to {output_path}")
            
            # Print some statistics
            print(f"pIC50 range: {final_df['pIC50'].min():.2f} - {final_df['pIC50'].max():.2f}")
            print(f"pIC50 mean: {final_df['pIC50'].mean():.2f}")
            print(f"pIC50 median: {final_df['pIC50'].median():.2f}")
            
            # Show distribution summary
            bins = [3, 4, 5, 6, 7, 8, 9, 10]
            hist, _ = np.histogram(final_df['pIC50'], bins=bins)
            for i in range(len(bins)-1):
                print(f"pIC50 {bins[i]}-{bins[i+1]}: {hist[i]} compounds")
                
        else:
            missing = []
            if 'standard_value' not in df.columns:
                missing.append('standard_value')
            if 'canonical_smiles' not in df.columns:
                missing.append('canonical_smiles')
            print(f"Required columns missing: {', '.join(missing)}")
            print("Available columns:", df.columns.tolist())

    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()
        print("Check your internet connection and make sure ChEMBL service is available.")

if __name__ == "__main__":
    main()
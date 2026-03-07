"""
Process Docking CSV with Flexible Format
=========================================

Utility script to convert docking results from various formats
to standardized format for integration with ML predictions.

Handles:
- Different column names (compound_ID, CompoundID, MOL_ID, etc.)
- Different score columns (score, binding_affinity, etc.)
- CSV with or without headers
- Tab or comma separated

Author: C-SCAN Pipeline
Date: January 30, 2026
"""

import pandas as pd
import sys
import os


def process_docking_csv(input_file, output_file=None, 
                       id_column=None, score_column=None,
                       separator=None, has_header=True):
    """
    Process docking CSV and standardize to MOL_ID, score format
    
    Parameters:
    -----------
    input_file : str
        Path to input CSV file
    output_file : str, optional
        Path to output CSV (default: adds _processed to input name)
    id_column : str, optional
        Name of compound ID column (auto-detected if None)
    score_column : str, optional
        Name of score column (auto-detected if None)
    separator : str, optional
        CSV separator (auto-detected if None)
    has_header : bool
        Whether CSV has header row (default: True)
    
    Returns:
    --------
    df : DataFrame
        Processed dataframe with standardized columns
    """
    
    print("="*70)
    print("DOCKING CSV PROCESSOR")
    print("="*70)
    print(f"\nInput file: {input_file}")
    
    if not os.path.exists(input_file):
        print(f"ERROR: File not found: {input_file}")
        return None
    
    # Auto-detect separator
    if separator is None:
        with open(input_file, 'r') as f:
            first_line = f.readline()
            if '\t' in first_line:
                separator = '\t'
                print("  Detected separator: TAB")
            else:
                separator = ','
                print("  Detected separator: COMMA")
    
    # Load CSV
    try:
        if has_header:
            df = pd.read_csv(input_file, sep=separator)
        else:
            df = pd.read_csv(input_file, sep=separator, header=None)
            # Auto-assign column names
            if len(df.columns) == 2:
                df.columns = ['compound_ID', 'score']
                print("  No header detected, assigned: compound_ID, score")
            elif len(df.columns) >= 2:
                df.columns = [f'col_{i}' for i in range(len(df.columns))]
                print(f"  No header detected, assigned generic names")
    except Exception as e:
        print(f"ERROR loading CSV: {e}")
        return None
    
    print(f"  Loaded {len(df)} rows")
    print(f"  Columns: {df.columns.tolist()}")
    
    # Auto-detect ID column
    if id_column is None:
        possible_id_cols = ['MOL_ID', 'Mol_ID', 'mol_id', 'compound_ID', 
                           'Compound_ID', 'compound_id', 'CompoundID', 
                           'ID', 'id', 'molecule_id', 'Molecule_ID',
                           'name', 'Name', 'compound_name']
        
        for col in possible_id_cols:
            if col in df.columns:
                id_column = col
                print(f"  Detected ID column: {id_column}")
                break
        
        if id_column is None and len(df.columns) >= 2:
            # Use first column as ID
            id_column = df.columns[0]
            print(f"  Using first column as ID: {id_column}")
    
    # Auto-detect score column
    if score_column is None:
        possible_score_cols = ['score', 'Score', 'SCORE', 'docking_score', 
                              'Docking_Score', 'binding_affinity', 
                              'Binding_Affinity', 'affinity', 'Affinity',
                              'energy', 'Energy']
        
        for col in possible_score_cols:
            if col in df.columns:
                score_column = col
                print(f"  Detected score column: {score_column}")
                break
        
        if score_column is None:
            # Look for numeric columns
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
            if len(numeric_cols) > 0:
                score_column = numeric_cols[0]
                print(f"  Using first numeric column as score: {score_column}")
            elif len(df.columns) >= 2:
                # Use second column as score
                score_column = df.columns[1]
                print(f"  Using second column as score: {score_column}")
    
    if id_column is None or score_column is None:
        print("ERROR: Could not identify ID and score columns")
        print("Please specify manually using --id-column and --score-column")
        return None
    
    # Create standardized dataframe
    processed_df = pd.DataFrame({
        'MOL_ID': df[id_column],
        'score': df[score_column]
    })
    
    # Convert MOL_ID to string if needed
    processed_df['MOL_ID'] = processed_df['MOL_ID'].astype(str)
    
    # Convert score to float
    try:
        processed_df['score'] = processed_df['score'].astype(float)
    except:
        print("Warning: Could not convert scores to float")
    
    # Sort by score (ascending - more negative is better for docking)
    processed_df = processed_df.sort_values('score')
    
    print("\nProcessed data preview:")
    print(processed_df.head(10))
    
    print(f"\nScore statistics:")
    print(f"  Min (best): {processed_df['score'].min():.2f}")
    print(f"  Max (worst): {processed_df['score'].max():.2f}")
    print(f"  Mean: {processed_df['score'].mean():.2f}")
    print(f"  Median: {processed_df['score'].median():.2f}")
    
    # Save to file
    if output_file is None:
        base, ext = os.path.splitext(input_file)
        output_file = f"{base}_processed{ext}"
    
    processed_df.to_csv(output_file, index=False)
    print(f"\n✓ Saved processed file: {output_file}")
    
    return processed_df


def main():
    """Command line interface"""
    
    if len(sys.argv) < 2:
        print("Usage: python process_docking_csv.py <input_file> [output_file]")
        print("\nExample:")
        print("  python process_docking_csv.py docking_results.csv")
        print("  python process_docking_csv.py docking_results.csv processed_results.csv")
        print("\nOptional arguments:")
        print("  --id-column <name>     Specify compound ID column name")
        print("  --score-column <name>  Specify score column name")
        print("  --sep <separator>      Specify separator (tab or comma)")
        print("  --no-header            CSV has no header row")
        return
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith('--') else None
    
    # Parse optional arguments
    id_column = None
    score_column = None
    separator = None
    has_header = True
    
    for i in range(2, len(sys.argv)):
        if sys.argv[i] == '--id-column' and i+1 < len(sys.argv):
            id_column = sys.argv[i+1]
        elif sys.argv[i] == '--score-column' and i+1 < len(sys.argv):
            score_column = sys.argv[i+1]
        elif sys.argv[i] == '--sep' and i+1 < len(sys.argv):
            sep_arg = sys.argv[i+1].lower()
            separator = '\t' if sep_arg in ['tab', '\\t'] else ','
        elif sys.argv[i] == '--no-header':
            has_header = False
    
    # Process file
    df = process_docking_csv(input_file, output_file, id_column, 
                            score_column, separator, has_header)
    
    if df is not None:
        print("\n" + "="*70)
        print("PROCESSING COMPLETE!")
        print("="*70)
        print("\nNext steps:")
        print("  1. Review the processed file")
        print("  2. Use it with integrate_ml_docking.py")
        print("\nExample:")
        print(f"  python integrate_ml_docking.py \\")
        print(f"    --ml predictions/20260130/blind_screening/all_predictions.csv \\")
        print(f"    --docking {output_file if output_file else input_file.replace('.csv', '_processed.csv')}")


if __name__ == "__main__":
    main()

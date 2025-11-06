import streamlit as st
import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, MolSurf, GraphDescriptors
from rdkit.Chem import rdPartialCharges, rdMolDescriptors
from chembl_webresource_client.new_client import new_client
import time
import pickle
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import sys
sys.path.append(os.path.dirname(__file__))
from data_collection import fetch_with_progress, remove_duplicates
from generating_fingerprints import generate_fingerprints
from descriptors import generate_descriptors
from visualization import plot_distribution, plot_property_distributions, plot_activity_vs_properties
from visualization import visualize_clusters, chemical_space_visualization, activity_landscape_3d
from fingerprints_clustering import cluster_fingerprints
from subcluster_analysis import determine_optimal_subclusters, subcluster_analysis

# Set page configuration
st.set_page_config(
    page_title="Chemical-Space & Cluster ANalysis",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

def safe_image_display(image_path, alt_text="No image available"):
    """Safely display an image, showing a message if the file doesn't exist"""
    if os.path.exists(image_path):
        try:
            st.image(image_path)
        except Exception as e:
            st.error(f"Error displaying image: {str(e)}")
            st.info(alt_text)
    else:
        st.info(f"{alt_text} (File not found: {os.path.basename(image_path)})")

# Available descriptors list
AVAILABLE_DESCRIPTORS = {
    "Lipinski properties": {
        "MolWt": "Molecular Weight (Da) - Size of molecule",
        "MolLogP": "Octanol-water partition coefficient - Lipophilicity",
        "NumHDonors": "Number of H-bond donors - Important for binding interactions",
        "NumHAcceptors": "Number of H-bond acceptors - Affects solubility and binding"
    },
    "Topological properties": {
        "TPSA": "Topological Polar Surface Area (Å²) - Related to membrane permeability",
        "NumRotatableBonds": "Number of rotatable bonds - Molecular flexibility",
        "NumAromaticRings": "Number of aromatic rings - Common in kinase inhibitors",
        "NumAliphaticRings": "Number of aliphatic rings - Affects 3D structure"
    },
    "Structural features": {
        "FractionCSP3": "Fraction of sp3 hybridized carbons - Complexity measure",
        "NumHeteroatoms": "Number of heteroatoms - Important for binding interactions",
        "RingCount": "Total number of rings - Related to structural rigidity"
    },
    "Electronic/surface properties": {
        "LabuteASA": "Labute Accessible Surface Area - Molecular size and shape",
        "PEOE_VSA1": "Partial charge VSA descriptor 1 - Electrostatic interaction potential",
        "PEOE_VSA2": "Partial charge VSA descriptor 2 - Electrostatic interaction potential",
        "SlogP_VSA1": "LogP contribution descriptor 1 - Hydrophobic regions",
        "SMR_VSA1": "Molecular refractivity descriptor 1 - Polarizability"
    },
    "Connectivity descriptors": {
        "BalabanJ": "Balaban's J index - Molecular shape descriptor",
        "BertzCT": "Bertz complexity index - Structural complexity",
        "Chi0": "Molecular connectivity index chi-0 - Branching pattern",
        "Chi1": "Molecular connectivity index chi-1 - Path connectivity",
        "Kappa1": "Kappa shape index 1 - Molecular flexibility"
    },
    "Kinase-relevant descriptors": {
        "MaxPartialCharge": "Maximum partial charge - Important for binding interactions",
        "MinPartialCharge": "Minimum partial charge - Important for binding interactions",
        "fr_Ar_N": "Number of aromatic nitrogens - Common in kinase hinge-binding motifs",
        "fr_amide": "Number of amide bonds - Common in kinase inhibitors",
        "NumRings": "Total ring count - Ring systems are prevalent in kinase inhibitors"
    }
}

def show_file_info(filepath):
    """Show file information including size and last modified time"""
    if os.path.exists(filepath):
        size_kb = os.path.getsize(filepath) / 1024
        mod_time = datetime.fromtimestamp(os.path.getmtime(filepath))
        return f"File size: {size_kb:.1f} KB | Last modified: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}"
    return "File not found"

# Function to create directory structure
def create_project_dirs(timestamp):
    base_dir = f"data/processed/{timestamp}"
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(f"{base_dir}/clusters", exist_ok=True)
    os.makedirs(f"{base_dir}/models", exist_ok=True)
    os.makedirs(f"{base_dir}/visualizations", exist_ok=True)
    
    # Print directory structure for reference
    print(f"\nProject directory structure created at: {os.path.abspath(base_dir)}")
    print(f"├── clusters/")
    print(f"├── models/")
    print(f"└── visualizations/")
    
    return base_dir


# Sidebar components
def sidebar_components():
    st.sidebar.title("KIT Inhibitor QSAR Explorer")
    
    # Project settings section
    st.sidebar.header("Project Settings")
    
    # Timestamp/project name
    timestamp = st.sidebar.text_input("Project Timestamp/Name; Keep the same with an existing dataset", 
                                    value=datetime.now().strftime("%Y%m%d"),
                                    help="Unique identifier for this analysis run")
    
    # Create a base directory for the project
    if timestamp:
        create_project_dirs(timestamp)
    
    # Target selection
    st.sidebar.subheader("ChEMBL Target")
    chembl_id = st.sidebar.text_input("ChEMBL Target ID", 
                                     value="CHEMBL1936",
                                     help="ChEMBL ID for c-KIT/KIT kinase is CHEMBL1936")
    
    # Navigation
    st.sidebar.header("Navigation")
    page = st.sidebar.radio("Go to", 
                          ["Data Collection", "Descriptors & Fingerprints", 
                           "Clustering & Visualization", "Advanced Analysis"])
    
    return timestamp, chembl_id, page

# Data collection page
def data_collection_page(timestamp, chembl_id):
    st.header("KIT Inhibitor Data Collection")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Fetch Data from ChEMBL")
        
        # Data fetch options
        data_type = st.selectbox("Activity Type", ["IC50", "Ki", "Kd", "EC50"], index=0)
        units = st.selectbox("Units", ["nM", "μM", "pM"], index=0)
        limit = st.number_input("Data Limit (max records)", min_value=100, max_value=10000, value=2000)
        
        # Fetch button
        fetch_clicked = st.button("Fetch Data from ChEMBL")
        
        if fetch_clicked:
            with st.spinner("Fetching data from ChEMBL..."):
                try:
                    # Fetch data with progress
                    df = fetch_with_progress(chembl_id, data_type, units, limit)
                    
                    if df is not None and not df.empty:
                        # Process data
                        st.success(f"Retrieved {len(df)} records!")
                        
                        # Display the first few rows
                        st.write("Preview of fetched data:")
                        st.dataframe(df.head())
                        
                        # Calculate pIC50 and clean data
                        if 'standard_value' in df.columns and 'canonical_smiles' in df.columns:
                            with st.spinner("Processing data..."):
                                # Convert to proper data type
                                df['standard_value'] = df['standard_value'].astype(float)
                                
                                # Remove extreme or invalid values
                                df = df[(df['standard_value'] > 0) & (df['standard_value'] < 1e7)]
                                
                                # Calculate pIC50 = -log10(IC50 [M])
                                df['pIC50'] = -np.log10(df['standard_value'] * 1e-9)
                                
                                # Final dataframe with duplicate removal
                                cleaned_df = df[['molecule_chembl_id', 'canonical_smiles', 'pIC50']].dropna()
                                
                                if st.checkbox("Remove duplicates", value=True):
                                    # Use progress bar for duplicate removal
                                    with st.spinner("Removing duplicates..."):
                                        final_df = remove_duplicates(cleaned_df)
                                        st.success(f"Removed duplicates: {len(cleaned_df) - len(final_df)} compounds")
                                else:
                                    final_df = cleaned_df
                                
                                # Save processed data
                                output_path = f"data/processed/{timestamp}/kit_pic50_{timestamp}.csv"
                                final_df.to_csv(output_path, index=False)
                                st.success(f"✅ Saved {len(final_df)} compounds to {output_path}")
                                st.code(f"File saved at: {os.path.abspath(output_path)}")
                                
                                # Store in session state
                                st.session_state.processed_data = final_df
                                st.session_state.data_path = output_path
                                
                                # Display statistics
                                st.subheader("Dataset Statistics")
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("Compounds", len(final_df))
                                with col2:
                                    st.metric("Min pIC50", f"{final_df['pIC50'].min():.2f}")
                                with col3:
                                    st.metric("Max pIC50", f"{final_df['pIC50'].max():.2f}")
                                with col4:
                                    st.metric("Mean pIC50", f"{final_df['pIC50'].mean():.2f}")
                                
                                # Quick distribution visualization
                                st.subheader("Activity Distribution")
                                fig, ax = plt.subplots(figsize=(10, 4))
                                sns.histplot(final_df['pIC50'], kde=True, ax=ax)
                                plt.title("Distribution of pIC50 Values")
                                st.pyplot(fig)
                    else:
                        st.error("No data retrieved from ChEMBL. Please check your target ID.")
                        
                except Exception as e:
                    st.error(f"Error fetching data: {str(e)}")
    
    with col2:
        st.subheader("Previously Processed Data")
        
        # Search for existing data in the project directory
        processed_files = []
        base_dir = "data/processed"
        if os.path.exists(base_dir):
            for ts_dir in os.listdir(base_dir):
                dir_path = os.path.join(base_dir, ts_dir)
                if os.path.isdir(dir_path):
                    for file in os.listdir(dir_path):
                        if file.endswith(".csv") and "pic50" in file.lower():
                            processed_files.append(os.path.join(dir_path, file))
        
        if processed_files:
            st.write("Select an existing dataset:")
            selected_file = st.selectbox("Available datasets", processed_files)
            
            # Show preview information about the selected dataset
            if selected_file:
                # Extract timestamp from path to check for related files
                timestamp_from_path = os.path.basename(os.path.dirname(selected_file))
                dataset_dir = os.path.dirname(selected_file)
                
                # Check for related files
                descriptors_file = None
                fingerprints_file = None
                clustered_file = None
                
                # Look for descriptors file
                desc_patterns = ["*descriptors*.csv", "*desc*.csv"]
                for pattern in desc_patterns:
                    desc_files = glob.glob(os.path.join(dataset_dir, pattern))
                    if desc_files:
                        descriptors_file = desc_files[0]
                        break
                
                # Look for fingerprints file
                fp_patterns = ["*fingerprints*.csv", "*fp*.csv"]
                for pattern in fp_patterns:
                    fp_files = glob.glob(os.path.join(dataset_dir, pattern))
                    if fp_files:
                        fingerprints_file = fp_files[0]
                        break
                
                # Look for clustered file
                clustered_patterns = [
                    os.path.join(dataset_dir, "clusters", "*clustered*.csv"),
                    os.path.join(dataset_dir, "*clustered*.csv")
                ]
                for pattern in clustered_patterns:
                    clust_files = glob.glob(pattern)
                    if clust_files:
                        clustered_file = clust_files[0]
                        break
                
                # Display file information
                st.write("**Dataset Information:**")
                try:
                    df_preview = pd.read_csv(selected_file)
                    st.write(f"- **Main dataset**: {len(df_preview)} compounds")
                    st.write(f"- **Timestamp**: {timestamp_from_path}")
                    
                    # Show which additional files are available
                    available_files = []
                    if descriptors_file:
                        available_files.append("✅ Descriptors")
                    else:
                        available_files.append("❌ Descriptors")
                    
                    if fingerprints_file:
                        available_files.append("✅ Fingerprints")
                    else:
                        available_files.append("❌ Fingerprints")
                    
                    if clustered_file:
                        available_files.append("✅ Clustered data")
                    else:
                        available_files.append("❌ Clustered data")
                    
                    st.write("**Available processed files:**")
                    for file_status in available_files:
                        st.write(f"  {file_status}")
                    
                except Exception as e:
                    st.error(f"Error reading dataset preview: {str(e)}")
            
            if st.button("Load Selected Dataset"):
                with st.spinner("Loading dataset and associated files..."):
                    try:
                        # Load main dataset
                        df = pd.read_csv(selected_file)
                        st.session_state.processed_data = df
                        st.session_state.data_path = selected_file
                        st.success(f"✅ Loaded main dataset: {len(df)} compounds from {os.path.basename(selected_file)}")
                        
                        # Track what was loaded
                        loaded_components = ["Main dataset"]
                        
                        # Load descriptors if available
                        if descriptors_file and os.path.exists(descriptors_file):
                            try:
                                desc_df = pd.read_csv(descriptors_file)
                                st.session_state.descriptors_path = descriptors_file
                                st.success(f"✅ Loaded descriptors: {descriptors_file}")
                                loaded_components.append("Molecular descriptors")
                                
                                # Show descriptor count
                                desc_cols = [col for col in desc_df.columns if col not in ['molecule_chembl_id', 'canonical_smiles', 'pIC50']]
                                st.info(f"📊 Found {len(desc_cols)} molecular descriptors")
                                
                            except Exception as e:
                                st.warning(f"⚠️ Found descriptors file but couldn't load it: {str(e)}")
                        
                        # Load fingerprints if available
                        if fingerprints_file and os.path.exists(fingerprints_file):
                            try:
                                fp_df = pd.read_csv(fingerprints_file)
                                st.session_state.fingerprints_path = fingerprints_file
                                st.success(f"✅ Loaded fingerprints: {fingerprints_file}")
                                loaded_components.append("Molecular fingerprints")
                                
                                # Show fingerprint info
                                fp_cols = [col for col in fp_df.columns if col.startswith('bit_')]
                                st.info(f"🧬 Found {len(fp_cols)} fingerprint bits")
                                
                            except Exception as e:
                                st.warning(f"⚠️ Found fingerprints file but couldn't load it: {str(e)}")
                        
                        # Load clustered data if available
                        if clustered_file and os.path.exists(clustered_file):
                            try:
                                clust_df = pd.read_csv(clustered_file)
                                if 'cluster' in clust_df.columns:
                                    st.session_state.clustered_path = clustered_file
                                    st.success(f"✅ Loaded clustered data: {clustered_file}")
                                    loaded_components.append("Clustered data")
                                    
                                    # Show cluster info
                                    n_clusters = clust_df['cluster'].nunique()
                                    st.info(f"🔬 Found {n_clusters} clusters with {len(clust_df)} compounds")
                                else:
                                    st.warning(f"⚠️ Found clustered file but it doesn't contain cluster information")
                                    
                            except Exception as e:
                                st.warning(f"⚠️ Found clustered file but couldn't load it: {str(e)}")
                        
                        # Show comprehensive loading summary
                        st.subheader("📋 Loading Summary")
                        st.write(f"**Successfully loaded:** {', '.join(loaded_components)}")
                        
                        # Show what's ready for analysis
                        st.write("**Ready for:**")
                        ready_for = []
                        if 'descriptors_path' in st.session_state:
                            ready_for.append("• Property analysis and correlation studies")
                        if 'fingerprints_path' in st.session_state:
                            ready_for.append("• Chemical similarity analysis and clustering")
                        if 'clustered_path' in st.session_state:
                            ready_for.append("• Advanced cluster analysis and subclustering")
                        
                        if ready_for:
                            for item in ready_for:
                                st.write(item)
                        else:
                            st.write("• Basic visualization and analysis")
                        
                        # Update timestamp in session for consistency
                        st.session_state.current_timestamp = timestamp_from_path
                        
                    except Exception as e:
                        st.error(f"Error loading dataset: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
        else:
            st.info("No processed data files found. Fetch new data from ChEMBL.")

# Descriptors and fingerprints page
def descriptors_fingerprints_page(timestamp):
    st.header("Molecular Descriptors & Fingerprints")
    
    # Check if we have data
    if 'data_path' not in st.session_state:
        st.warning("Please fetch or load data first from the Data Collection page")
        return
    
    data_path = st.session_state.data_path
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Generate Molecular Descriptors")
        
        
        st.write("Select descriptor categories to include:")
        
        # Keep track of selected descriptors
        selected_descriptors = {}
        
        # Use an expander for each category for better organization
        for category, descriptors in AVAILABLE_DESCRIPTORS.items():
            with st.expander(f"{category} ({len(descriptors)} descriptors)"):
                # Show a checkbox for the entire category
                select_all = st.checkbox(f"Select all {category}", value=True, key=f"all_{category}")
                
                st.markdown("---")
                
                # Show individual descriptors with explanations
                for desc_name, desc_info in descriptors.items():
                    selected = st.checkbox(
                        f"**{desc_name}**: {desc_info}", 
                        value=select_all,
                        key=f"{category}_{desc_name}"
                    )
                    if selected:
                        selected_descriptors[desc_name] = True
        
        # Show selection summary
        st.write(f"**Selected {len(selected_descriptors)} descriptors across {len(AVAILABLE_DESCRIPTORS)} categories**")
        
        # Generate button
        if st.button("Generate Selected Descriptors"):
            if len(selected_descriptors) == 0:
                st.warning("Please select at least one descriptor")
            else:
                # Filter descriptors based on selected checkboxes
                descriptor_calculators = {}
                for desc_name in selected_descriptors:
                    if hasattr(Descriptors, desc_name):
                        descriptor_calculators[desc_name] = getattr(Descriptors, desc_name)
                    elif hasattr(Lipinski, desc_name):
                        descriptor_calculators[desc_name] = getattr(Lipinski, desc_name)
                    elif hasattr(MolSurf, desc_name):
                        descriptor_calculators[desc_name] = getattr(MolSurf, desc_name)
                    elif hasattr(GraphDescriptors, desc_name):
                        descriptor_calculators[desc_name] = getattr(GraphDescriptors, desc_name)
                    elif hasattr(rdPartialCharges, desc_name):
                        descriptor_calculators[desc_name] = getattr(rdPartialCharges, desc_name)
                    elif hasattr(rdMolDescriptors, desc_name):
                        descriptor_calculators[desc_name] = getattr(rdMolDescriptors, desc_name)
                
                # Call descriptor generation function with selected descriptors
                with st.spinner(f"Generating {len(selected_descriptors)} molecular descriptors..."):
                    try:
                        output_file = f"data/processed/{timestamp}/kit_descriptors_selected.csv"
                        
                        # Custom implementation to use our selected descriptors
                        from descriptors import generate_descriptors
                        df = generate_descriptors(data_path, output_file, selected_only=True)
                        
                        st.success(f"Generated {len(selected_descriptors)} descriptors for {len(df)} compounds")
                        st.session_state.descriptors_path = output_file
                        st.info(f"📊 Descriptors saved to: {os.path.abspath(output_file)}")
                        
                        # Show preview
                        st.subheader("Preview of Generated Descriptors")
                        desc_cols = [col for col in df.columns if col not in ['molecule_chembl_id', 'canonical_smiles', 'pIC50']][:5]
                        st.dataframe(df[['molecule_chembl_id', 'canonical_smiles', 'pIC50'] + desc_cols].head())
                        
                    except Exception as e:
                        st.error(f"Error generating descriptors: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
    
    with col2:
        st.subheader("Generate Molecular Fingerprints")
        
        # Fingerprint options
        fp_type = st.selectbox("Fingerprint Type", ["Morgan (ECFP)", "MACCS Keys", "Atom Pairs"], index=0)
        
        if fp_type == "Morgan (ECFP)":
            radius = st.slider("Radius", min_value=1, max_value=4, value=2)
            nbits = st.slider("Number of Bits", min_value=512, max_value=4096, value=2048, step=512)
        
        # Generate button
        if st.button("Generate Fingerprints"):
            with st.spinner("Generating fingerprints..."):
                try:
                    output_file = f"data/processed/{timestamp}/kit_fingerprints.csv"
                    
                    if fp_type == "Morgan (ECFP)":
                        df = generate_fingerprints(data_path, output_file, radius=radius, nBits=nbits)
                    else:
                        # For future implementation of other fingerprint types
                        st.warning(f"{fp_type} not fully implemented yet, using Morgan fingerprints")
                        df = generate_fingerprints(data_path, output_file)
                    
                    st.success(f"Generated fingerprints for {len(df)} compounds")
                    st.session_state.fingerprints_path = output_file
                    st.info(f"🧬 Fingerprints saved to: {os.path.abspath(output_file)}")
                    
                    # Show preview (just first few fingerprint bits)
                    st.write("Preview of fingerprint data:")
                    display_cols = ['molecule_chembl_id', 'pIC50'] + [f"bit_{i}" for i in range(5)]
                    disp_df = df[display_cols].head()
                    st.dataframe(disp_df)
                    
                except Exception as e:
                    st.error(f"Error generating fingerprints: {str(e)}")

# Clustering and visualization page
def clustering_visualization_page(timestamp):
    st.header("Clustering & Visualization")
    
    # Check for required data
    if 'data_path' not in st.session_state:
        st.warning("Please fetch or load data first from the Data Collection page")
        return
    
    # Initialize paths
    data_path = st.session_state.data_path
    fingerprints_path = st.session_state.get('fingerprints_path', None)
    descriptors_path = st.session_state.get('descriptors_path', None)
    clustered_path = f"data/processed/{timestamp}/clusters/kit_fingerprints_clustered.csv"
    
    # Tabs for different analyses
    tab1, tab2, tab3 = st.tabs(["Basic Visualization", "Clustering", "Chemical Space"])
    
    # Tab 1: Basic Visualization
    with tab1:
        st.subheader("Basic Data Visualization")
        
        viz_options = st.multiselect(
            "Select visualizations to generate", 
            ["Activity Distribution", "Property Distributions", "Property-Activity Relationships"],
            default=["Activity Distribution"]
        )
        
        if st.button("Generate Visualizations"):
            output_dir = f"data/processed/{timestamp}/visualizations"
            os.makedirs(output_dir, exist_ok=True)
            
            with st.spinner("Generating visualizations..."):
                # First determine best data source
                if "Activity Distribution" in viz_options:
                    # Create and display pIC50 distribution
                    fig_col1, fig_col2 = st.columns([2, 1])
                    with fig_col1:
                        plot_distribution(data_path, output_dir)
                        safe_image_display(f"{output_dir}/pIC50_distribution.png", "Distribution plot not available")
                        st.info(f"📈 Activity distribution plot saved to: {os.path.abspath(f'{output_dir}/pIC50_distribution.png')}")
                    
                    with fig_col2:
                        # Display activity class statistics
                        df = pd.read_csv(data_path)
                        low = sum(df['pIC50'] < 6)
                        moderate = sum((df['pIC50'] >= 6) & (df['pIC50'] < 7))
                        high = sum((df['pIC50'] >= 7) & (df['pIC50'] < 8))
                        very_high = sum(df['pIC50'] >= 8)
                        
                        st.subheader("Activity Distribution")
                        stats_df = pd.DataFrame({
                            'Activity Class': ['Low', 'Moderate', 'High', 'Very High'],
                            'pIC50 Range': ['<6', '6-7', '7-8', '>8'],
                            'Count': [low, moderate, high, very_high],
                            '%': [f"{100*low/len(df):.1f}%", f"{100*moderate/len(df):.1f}%", 
                                f"{100*high/len(df):.1f}%", f"{100*very_high/len(df):.1f}%"]
                        })
                        st.table(stats_df)
                
                if "Property Distributions" in viz_options:
                    # Determine best data source for properties
                    prop_source = descriptors_path if descriptors_path else data_path
                    df_with_props = plot_property_distributions(prop_source, output_dir)
                    st.info(f"📊 Property visualizations saved to: {os.path.abspath(output_dir)}")  # Add this line
                    st.code("\n".join([f"- {os.path.basename(f)}" for f in glob.glob(f"{output_dir}/property_*.png")[:5]]) + 
                            ("\n- ..." if len(glob.glob(f"{output_dir}/property_*.png")) > 5 else ""))
                    
                    # Display key property plots
                    st.subheader("Key Property Distributions")
                    safe_image_display(
                        f"{output_dir}/property_distributions.png", 
                        "Property distribution visualization not available. Make sure descriptors were properly generated."
                    )
                    
                    # Display Lipinski compliance if available
                    st.subheader("Drug-Likeness Analysis")
                    safe_image_display(
                        f"{output_dir}/lipinski_violations.png",
                        "Lipinski rule analysis not available"
                    )
                
                if "Property-Activity Relationships" in viz_options:
                    # Need to use data with properties
                    prop_source = descriptors_path if descriptors_path else data_path
                    
                    # This function will create the plots
                    try:
                        plot_activity_vs_properties(prop_source, output_dir)
                        st.info(f"📊 Correlation plots saved to: {os.path.abspath(output_dir)}")
                        st.code("\n".join([f"- {os.path.basename(f)}" for f in glob.glob(f"{output_dir}/correlation_*.png")[:3]]) + 
                                ("\n- ..." if len(glob.glob(f"{output_dir}/correlation_*.png")) > 3 else ""))
                        
                        # Display TOP activity correlation heatmap (shows more properties)
                        st.subheader("Top Activity-Correlated Properties")
                        safe_image_display(
                            f"{output_dir}/top_activity_correlations.png",
                            "Top activity correlations visualization not available"
                        )
                        
                        # Display the full correlation matrix (optional - can be kept or removed)
                        st.subheader("Full Property Correlation Matrix")
                        safe_image_display(
                            f"{output_dir}/correlation_heatmap.png",
                            "Full correlation matrix not available"
                        )
                        
                        # Show top correlating property relationship
                        st.subheader("Top Property-Activity Relationship")
                        safe_image_display(
                            f"{output_dir}/pIC50_vs_TPSA.png",
                            "Property-activity relationship plot not available"
                        )
                        
                        # Add option to download full correlation data
                        if os.path.exists(f"{output_dir}/all_property_correlations.csv"):
                            with open(f"{output_dir}/all_property_correlations.csv", "rb") as file:
                                st.download_button(
                                    label="Download Complete Correlation Matrix (CSV)",
                                    data=file,
                                    file_name="all_property_correlations.csv",
                                    mime="text/csv"
                                )
                    except Exception as e:
                        st.error(f"Error generating property correlations: {str(e)}")
                        st.info("Make sure to generate descriptors first!")
    
    # Tab 2: Clustering
    with tab2:
        st.subheader("Chemical Clustering")
        
        if fingerprints_path:
            n_clusters = st.slider("Number of Clusters", min_value=2, max_value=10, value=5)
            
            if st.button("Perform Clustering"):
                with st.spinner("Clustering compounds based on fingerprints..."):
                    try:
                        # Call clustering function
                        df_clustered, kmeans = cluster_fingerprints(
                            fingerprints_path, 
                            n_clusters=n_clusters,
                            random_state=42
                        )
                        
                        # IMPORTANT: Explicitly save the clustered data to the correct path
                        os.makedirs(f"data/processed/{timestamp}/clusters", exist_ok=True)
                        clustered_output_path = f"data/processed/{timestamp}/clusters/kit_fingerprints_clustered.csv"
                        
                        # Save the clustered dataframe
                        df_clustered.to_csv(clustered_output_path, index=False)
                        
                        st.success(f"Clustering complete! Compounds divided into {n_clusters} clusters.")
                        st.info(f"🔬 Clustered data saved to: {os.path.abspath(clustered_output_path)}")
                        
                        # Verify the file was actually saved
                        if os.path.exists(clustered_output_path):
                            file_size = os.path.getsize(clustered_output_path) / 1024
                            st.success(f"✅ File successfully saved ({file_size:.1f} KB)")
                            
                            # Update session state with the correct path
                            st.session_state.clustered_path = clustered_output_path
                            
                            # Show a preview of the saved data
                            st.write("**Preview of clustered data:**")
                            st.dataframe(df_clustered[['molecule_chembl_id', 'pIC50', 'cluster']].head())
                            
                        else:
                            st.error("❌ Failed to save clustered data file")
                        
                        # Show cluster distribution
                        st.subheader("Cluster Distribution")
                        cluster_counts = df_clustered['cluster'].value_counts().sort_index()
                        
                        fig, ax = plt.subplots(figsize=(10, 5))
                        bars = ax.bar(cluster_counts.index, cluster_counts.values)
                        
                        # Add labels
                        for bar in bars:
                            height = bar.get_height()
                            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                                    f'{int(height)}', ha='center', va='bottom')
                        
                        ax.set_xlabel('Cluster')
                        ax.set_ylabel('Number of Compounds')
                        ax.set_title('Distribution of Compounds Across Clusters')
                        st.pyplot(fig)
                        
                        # Show activity by cluster
                        st.subheader("Activity by Cluster")
                        fig2, ax2 = plt.subplots(figsize=(10, 5))
                        sns.boxplot(x='cluster', y='pIC50', data=df_clustered, ax=ax2)
                        ax2.set_title('pIC50 Distribution by Cluster')
                        st.pyplot(fig2)
                        
                    except Exception as e:
                        st.error(f"Error during clustering: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
        else:
            st.warning("Please generate fingerprints first on the Descriptors & Fingerprints page")
    
    # Tab 3: Chemical Space
    with tab3:
        st.subheader("Chemical Space Visualization")
        
        # Add methodology explanation for users
        st.info("""
        **Chemical Space Visualization Methodology:**
        
        **Purpose**: Map compounds in a lower-dimensional space to reveal structural similarities, 
        activity patterns, and identify regions of interest for drug discovery.
        
        **Input Data**: Molecular fingerprints (binary vectors encoding structural features) or 
        clustered compound data with activity information.
        """)
        
        viz_source = clustered_path if os.path.exists(clustered_path) else fingerprints_path
        
        if viz_source and os.path.exists(viz_source):
            space_type = st.radio("Visualization Type", ["2D Chemical Space", "3D Activity Landscape"])
            
            # Add methodology explanations based on selected visualization type
            if space_type == "2D Chemical Space":
                st.markdown("""
                ### 2D Chemical Space Methodology
                
                **Dimensionality Reduction Pipeline:**
                1. **Data Preprocessing**: Molecular fingerprints are standardized using StandardScaler to ensure equal contribution of all bits
                2. **PCA (Principal Component Analysis)**: Applied first to reduce dimensions to ~50 components, capturing major variance while removing noise
                3. **t-SNE (t-Distributed Stochastic Neighbor Embedding)**: Applied to PCA results to create final 2D visualization
                
                **Why this approach?**
                - **PCA**: Linear method that preserves global structure and removes correlated features
                - **t-SNE**: Non-linear method that preserves local neighborhoods, revealing cluster structure
                - **Combined**: PCA+t-SNE is computationally efficient and provides both global and local structure preservation
                
                **Interpretation:**
                - **Clusters**: Groups of structurally similar compounds
                - **Color coding**: By cluster membership or biological activity (pIC50)
                - **Distance**: Closer points = more structurally similar compounds
                - **Outliers**: Unique chemical scaffolds or potential activity cliffs
                """)
            
            elif space_type == "3D Activity Landscape":
                st.markdown("""
                ### 3D Activity Landscape Methodology
                
                **Dimensionality Reduction for 3D:**
                1. **Standardization**: Fingerprint features scaled to unit variance
                2. **PCA to 3D**: Direct reduction to 3 principal components for visualization
                3. **Activity Mapping**: pIC50 values mapped to color scale on 3D points
                
                **Why PCA for 3D?**
                - **Interpretable axes**: Each PC represents orthogonal chemical variation
                - **Variance explained**: Shows how much structural diversity each axis captures
                - **Computational efficiency**: Fast and deterministic results
                - **Global structure**: Maintains overall relationships between compound classes
                
                **Key Features:**
                - **Interactive 3D plot**: Rotate, zoom, and hover for compound details
                - **Activity gradient**: Color intensity shows biological activity strength
                - **Structure-Activity Relationships**: Identify activity cliffs and flat regions
                - **Cluster separation**: 3D view reveals cluster boundaries not visible in 2D
                
                **Applications:**
                - **Lead optimization**: Find high-activity regions for focused synthesis
                - **Activity cliffs**: Identify small structural changes causing large activity differences
                - **Chemical diversity**: Assess coverage of chemical space in compound libraries
                """)
            
            if st.button("Generate Chemical Space Visualization"):
                with st.spinner("Generating chemical space visualization..."):
                    output_dir = f"data/processed/{timestamp}/visualizations"
                    os.makedirs(output_dir, exist_ok=True)
                    
                    if space_type == "2D Chemical Space":
                        try:
                            # Add technical details during processing
                            st.info("🔄 **Processing Steps:**")
                            st.write("1. Loading molecular fingerprint data...")
                            st.write("2. Applying StandardScaler for feature normalization...")
                            st.write("3. Computing PCA (50 components) for noise reduction...")
                            st.write("4. Applying t-SNE for 2D embedding...")
                            st.write("5. Generating cluster and activity visualizations...")
                            
                            chemical_space_visualization(viz_source, output_dir)
                            st.success("Chemical space visualization complete!")
                            st.info(f"🧪 Chemical space visualization saved to: {os.path.abspath(f'{output_dir}/chemical_space.png')}")
                            
                            # Add interpretation guidance
                            st.markdown("""
                            ### How to Interpret Your 2D Chemical Space:
                            
                            **Left Panel (if clustered data available)**: 
                            - Each color represents a different structural cluster
                            - Tight clusters = very similar compounds
                            - Scattered points = diverse structures
                            
                            **Right Panel**: 
                            - Color intensity = biological activity (pIC50)
                            - Hot colors (red/orange) = high activity
                            - Cool colors (blue/purple) = low activity
                            
                            **Look for:**
                            - **Activity cliffs**: Nearby compounds with very different colors
                            - **Activity islands**: Isolated high-activity regions
                            - **Structural gaps**: Empty regions suggesting unexplored chemistry
                            """)
                            
                            # Display the chemical space plot
                            safe_image_display(f"{output_dir}/chemical_space.png", "Chemical space visualization not available")
                            
                            # If subclusters exist, show that visualization too
                            subcluster_file = glob.glob(f"{output_dir}/cluster_*_subclusters.png")
                            if subcluster_file:
                                st.subheader("Subcluster Analysis")
                                st.info("This shows detailed substructural relationships within the largest cluster")
                                safe_image_display(subcluster_file[0], "Subcluster visualization not available")
                                
                        except Exception as e:
                            st.error(f"Error generating 2D chemical space: {str(e)}")
                    
                    elif space_type == "3D Activity Landscape":
                        try:
                            # Check if plotly is installed
                            try:
                                import plotly.graph_objects as go
                                
                                # Add technical details during processing
                                st.info("🔄 **3D Processing Steps:**")
                                st.write("1. Loading fingerprint data...")
                                st.write("2. Standardizing features...")
                                st.write("3. Computing 3-component PCA...")
                                st.write("4. Creating interactive 3D scatter plot...")
                                st.write("5. Mapping activity to color scale...")
                                
                                activity_landscape_3d(viz_source, output_dir)
                                st.success("3D activity landscape generated!")
                                st.info(f"🌐 3D landscape HTML saved to: {os.path.abspath(f'{output_dir}/activity_landscape_3d.html')}")
                                
                                # Add interpretation guidance for 3D
                                st.markdown("""
                                ### How to Use Your 3D Activity Landscape:
                                
                                **Interactive Controls:**
                                - **Mouse drag**: Rotate the 3D view
                                - **Mouse wheel**: Zoom in/out
                                - **Hover**: See compound details and activity values
                                
                                **Analysis Strategy:**
                                1. **Rotate to find best viewpoint** showing cluster separation
                                2. **Look for activity gradients** - smooth color transitions
                                3. **Identify activity cliffs** - sharp color changes between nearby points
                                4. **Find activity islands** - isolated high-activity (bright) regions
                                
                                **Principal Components represent:**
                                - **PC1 (X-axis)**: Largest source of structural variation
                                - **PC2 (Y-axis)**: Second largest source of variation  
                                - **PC3 (Z-axis)**: Third largest source of variation
                                """)
                                
                                # Display using HTML component with proper encoding handling
                                html_file = f"{output_dir}/activity_landscape_3d.html"
                                
                                try:
                                    # Try different encodings to handle the file properly
                                    html_data = None
                                    encodings_to_try = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252', 'iso-8859-1']
                                    
                                    for encoding in encodings_to_try:
                                        try:
                                            with open(html_file, 'r', encoding=encoding) as f:
                                                html_data = f.read()
                                            st.info(f"✅ Successfully loaded HTML file using {encoding} encoding")
                                            break
                                        except UnicodeDecodeError:
                                            continue
                                    
                                    if html_data is None:
                                        # If all encodings fail, try binary mode and handle errors
                                        with open(html_file, 'rb') as f:
                                            binary_data = f.read()
                                        html_data = binary_data.decode('utf-8', errors='ignore')
                                        st.warning("⚠️ Used fallback encoding with error handling - some characters may be missing")
                                    
                                    # Display the 3D visualization
                                    st.components.v1.html(html_data, width=900, height=800)
                                    
                                    # Provide download button with proper encoding
                                    st.download_button(
                                        "Download 3D Visualization", 
                                        data=html_data.encode('utf-8'), 
                                        file_name="activity_landscape_3d.html",
                                        mime="text/html"
                                    )
                                    
                                except Exception as file_error:
                                    st.error(f"Error reading HTML file: {str(file_error)}")
                                    st.info("The 3D visualization was generated but cannot be displayed due to encoding issues.")
                                    st.info(f"You can find the HTML file at: {html_file}")
                                    
                                    # Provide download option even if display fails
                                    try:
                                        with open(html_file, 'rb') as f:
                                            binary_data = f.read()
                                        st.download_button(
                                            "Download 3D Visualization (Binary)", 
                                            data=binary_data, 
                                            file_name="activity_landscape_3d.html",
                                            mime="text/html"
                                        )
                                    except Exception as download_error:
                                        st.error(f"Cannot provide download: {str(download_error)}")
                                        
                            except ImportError:
                                st.error("Plotly is required for 3D visualization. Install with: pip install plotly")
                                st.info("**Installation**: `pip install plotly` then restart your Streamlit app")
                        
                        except Exception as e:
                            st.error(f"Error generating 3D landscape: {str(e)}")
                            import traceback
                            st.code(traceback.format_exc())
        else:
            st.warning("Please generate fingerprints and/or perform clustering first")
            
            # Add guidance for users who haven't generated the required data
            st.markdown("""
            ### Getting Started with Chemical Space Visualization:
            
            **Prerequisites:**
            1. **Molecular Fingerprints**: Go to 'Descriptors & Fingerprints' → Generate fingerprints
            2. **Optional but Recommended**: Perform clustering for enhanced visualization
            
            **Fingerprints encode structural information as binary vectors:**
            - Each bit represents presence/absence of a molecular substructure
            - Morgan/ECFP fingerprints capture circular substructures around each atom
            - These high-dimensional vectors (512-4096 bits) need dimensionality reduction for visualization
            
            **Why Chemical Space Visualization?**
            - **Drug Discovery**: Identify unexplored regions with potential activity
            - **Lead Optimization**: Find structural neighbors of active compounds
            - **Library Design**: Ensure diverse coverage of chemical space
            - **SAR Analysis**: Visualize structure-activity relationships
            """)

# Advanced analysis page
def advanced_analysis_page(timestamp):
    st.header("Advanced Analysis")
    
    # Check for required data
    if 'data_path' not in st.session_state:
        st.warning("Please fetch or load data first from the Data Collection page")
        return
    
    # Tabs for different advanced analyses
    tab1, tab2, tab3 = st.tabs(["Subcluster Analysis", "QSAR Modeling", "Compound Prediction"])
    
    # Tab 1: Subcluster Analysis
    with tab1:
        st.subheader("Subcluster Analysis")
        
        # More robust checking for clustered data
        clustered_path = st.session_state.get('clustered_path', None)
        
        # If no clustered_path in session state, try to find it
        if not clustered_path or not os.path.exists(clustered_path):
            # Look for clustered files in the current timestamp directory
            possible_paths = [
                f"data/processed/{timestamp}/clusters/kit_fingerprints_clustered.csv",
                f"data/processed/{timestamp}/kit_fingerprints_clustered.csv",
                f"data/processed/{timestamp}/clusters/clustered_data.csv"
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    clustered_path = path
                    st.session_state.clustered_path = path
                    break
        
        # Debug information to help troubleshoot
        st.write("**Debug Information:**")
        st.write(f"- Looking for clustered data in timestamp: {timestamp}")
        st.write(f"- Clustered path from session state: {st.session_state.get('clustered_path', 'None')}")
        st.write(f"- Current clustered path: {clustered_path}")
        
        # Check if the file exists and show its status
        if clustered_path:
            if os.path.exists(clustered_path):
                st.success(f"✅ Clustered data found: {clustered_path}")
                
                # Show file info
                try:
                    df_clustered = pd.read_csv(clustered_path)
                    st.write(f"- Contains {len(df_clustered)} compounds")
                    if 'cluster' in df_clustered.columns:
                        n_clusters = df_clustered['cluster'].nunique()
                        st.write(f"- Divided into {n_clusters} clusters")
                    else:
                        st.error("❌ File exists but doesn't contain cluster column")
                        clustered_path = None
                except Exception as e:
                    st.error(f"❌ Error reading clustered file: {str(e)}")
                    clustered_path = None
            else:
                st.warning(f"❌ Clustered data file not found at: {clustered_path}")
                clustered_path = None
        
        # Show all files in the clusters directory for debugging
        clusters_dir = f"data/processed/{timestamp}/clusters"
        if os.path.exists(clusters_dir):
            files_in_dir = os.listdir(clusters_dir)
            st.write(f"**Files in {clusters_dir}:**")
            for file in files_in_dir:
                st.write(f"- {file}")
        else:
            st.write(f"**Clusters directory doesn't exist:** {clusters_dir}")
        
        if clustered_path and os.path.exists(clustered_path):
            # Load clustered data
            try:
                df_clustered = pd.read_csv(clustered_path)
                
                if 'cluster' not in df_clustered.columns:
                    st.error("The loaded file doesn't contain clustering results")
                    return
                
                clusters = sorted(df_clustered['cluster'].unique())
                
                st.write("Select a cluster to analyze in more detail:")
                target_cluster = st.selectbox("Target Cluster", 
                                            clusters,
                                            format_func=lambda c: f"Cluster {c} ({len(df_clustered[df_clustered['cluster']==c])} compounds)")
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    # Option for automatic optimization
                    auto_optimize = st.checkbox("Automatically determine optimal number of subclusters", value=True)
                    
                    if auto_optimize:
                        n_subclusters = "auto"
                    else:
                        cluster_size = len(df_clustered[df_clustered['cluster']==target_cluster])
                        max_subclusters = min(10, max(2, cluster_size // 10))  # Reasonable maximum
                        n_subclusters = st.slider("Number of Subclusters", min_value=2, max_value=max_subclusters, value=4)
                
                with col2:
                    # Options for subcluster analysis
                    st.write("Analysis options:")
                    analyze_scaffolds = st.checkbox("Analyze common scaffolds", value=True)
                    find_top = st.checkbox("Find top compounds by activity", value=True)
                    top_n = st.number_input("Number of top compounds", value=10, min_value=1, max_value=50) if find_top else 10
                
                if st.button("Run Subcluster Analysis"):
                    with st.spinner(f"Analyzing subclusters for Cluster {target_cluster}..."):
                        try:
                            
                            # Determine optimal number of subclusters if auto is selected
                            if n_subclusters == "auto":
                                clustered_data_file = f"data/processed/{timestamp}/clusters/kit_fingerprints_clustered.csv"
                                optimal_subclusters = determine_optimal_subclusters(
                                    clustered_data_file, 
                                    target_cluster=target_cluster, 
                                    max_clusters=10
                                )
                                st.info(f"🔍 Optimal number of subclusters determined: {optimal_subclusters}")
                                n_subclusters = optimal_subclusters
                            
                            # Run the subcluster analysis
                            subcluster_df = subcluster_analysis(
                                timestamp=timestamp,
                                target_cluster=target_cluster, 
                                n_subclusters=n_subclusters
                            )
                            
                            if subcluster_df is not None:
                                st.success(f"Subcluster analysis complete! Cluster {target_cluster} divided into {n_subclusters} subclusters")
                                st.info(f"🧩 Subcluster visualizations saved to: {os.path.abspath(f'data/processed/{timestamp}/subclusters/')}")
                                
                                # Display subcluster statistics
                                st.subheader("Subcluster Statistics")
                                subcluster_stats = subcluster_df.groupby('subcluster')['pIC50'].agg(['count', 'mean', 'std']).round(3)
                                st.dataframe(subcluster_stats)
                                
                                # Display the subcluster visualization
                                st.subheader("Subcluster Visualization")
                                subcluster_viz_path = f"data/processed/{timestamp}/subclusters/cluster_{target_cluster}_subclusters.png"
                                safe_image_display(subcluster_viz_path, "Subcluster visualization not available")
                                
                                # Show top molecules if requested
                                if find_top:
                                    st.subheader(f"Top {top_n} Compounds by Activity")
                                    top_molecules_path = f"data/processed/{timestamp}/subclusters/top_molecules/cluster_{target_cluster}_top_{top_n}.png"
                                    safe_image_display(top_molecules_path, "Top molecules visualization not available")
                                    
                                    # Show top molecules CSV data
                                    top_csv_path = f"data/processed/{timestamp}/subclusters/top_molecules/cluster_{target_cluster}_top_{top_n}.csv"
                                    if os.path.exists(top_csv_path):
                                        top_df = pd.read_csv(top_csv_path)
                                        st.dataframe(top_df[['molecule_chembl_id', 'pIC50', 'subcluster']].head(10))
                                
                                # Show scaffold analysis if requested
                                if analyze_scaffolds:
                                    st.subheader("Common Scaffolds Analysis")
                                    scaffold_summary_path = f"data/processed/{timestamp}/subclusters/scaffolds/scaffold_summary.csv"
                                    if os.path.exists(scaffold_summary_path):
                                        scaffold_df = pd.read_csv(scaffold_summary_path)
                                        st.dataframe(scaffold_df.head(10))
                                    
                                    # Show scaffold images for each subcluster
                                    for sc in sorted(subcluster_df['subcluster'].unique()):
                                        scaffold_img_path = f"data/processed/{timestamp}/subclusters/scaffolds/cluster_{target_cluster}_subcluster_{sc}_scaffolds.png"
                                        if os.path.exists(scaffold_img_path):
                                            st.write(f"**Subcluster {sc} Common Scaffolds:**")
                                            safe_image_display(scaffold_img_path, f"Scaffolds for subcluster {sc} not available")
                                
                                # Provide download links for the generated files
                                st.subheader("Download Results")
                                
                                # Download subcluster data
                                subcluster_csv_path = f"data/processed/{timestamp}/subclusters/cluster_{target_cluster}_subclustered.csv"
                                if os.path.exists(subcluster_csv_path):
                                    with open(subcluster_csv_path, "rb") as file:
                                        st.download_button(
                                            label="Download Subcluster Data (CSV)",
                                            data=file,
                                            file_name=f"cluster_{target_cluster}_subclustered.csv",
                                            mime="text/csv"
                                        )
                                
                                # Download top molecules data
                                if find_top and os.path.exists(top_csv_path):
                                    with open(top_csv_path, "rb") as file:
                                        st.download_button(
                                            label=f"Download Top {top_n} Molecules (CSV)",
                                            data=file,
                                            file_name=f"cluster_{target_cluster}_top_{top_n}.csv",
                                            mime="text/csv"
                                        )
                            else:
                                st.error("Failed to perform subcluster analysis. Check the logs for details.")
                                
                        except ImportError as e:
                            st.error(f"Error importing subcluster analysis module: {str(e)}")
                            st.info("Make sure subcluster_analysis.py is in your project directory")
                        except Exception as e:
                            st.error(f"Error in subcluster analysis: {str(e)}")
                            import traceback
                            st.code(traceback.format_exc())
                            
            except Exception as e:
                st.error(f"Error loading clustered data: {str(e)}")
        else:
            st.warning("⚠️ No clustered data found. Please perform clustering first on the Clustering & Visualization page")
            
            # Provide helpful guidance
            st.info("**To perform clustering:**")
            st.write("1. Go to 'Descriptors & Fingerprints' page")
            st.write("2. Generate molecular fingerprints")
            st.write("3. Go to 'Clustering & Visualization' page")
            st.write("4. Use the 'Clustering' tab to perform clustering")
    
    # Tab 2: QSAR Modeling placeholder
    with tab2:
        st.subheader("QSAR Modeling")
        st.info("QSAR modeling functionality will be implemented in a future version")
    
    # Tab 3: Compound Prediction placeholder
    with tab3:
        st.subheader("Compound Prediction")
        st.info("Compound prediction functionality will be implemented in a future version")
        
# Main app
def main():
    # Get sidebar components
    timestamp, chembl_id, page = sidebar_components()
    
    # Display selected page
    if page == "Data Collection":
        data_collection_page(timestamp, chembl_id)
    elif page == "Descriptors & Fingerprints":
        descriptors_fingerprints_page(timestamp)
    elif page == "Clustering & Visualization":
        clustering_visualization_page(timestamp)
    elif page == "Advanced Analysis":
        advanced_analysis_page(timestamp)

if __name__ == "__main__":
    main()
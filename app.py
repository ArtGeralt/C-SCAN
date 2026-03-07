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
            with open(image_path, "rb") as _f:
                st.image(_f.read())
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


# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------

def _init_session() -> None:
    """Initialise session-state keys on first run so they always exist."""
    defaults: dict = {
        "active_timestamp": datetime.now().strftime("%Y%m%d"),
        "_load_banner": None,
        "data_path": None,
        "descriptors_path": None,
        "fingerprints_path": None,
        "clustered_path": None,
        "classification_model_dir": None,
        "screening_results_path": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _session_status_bar() -> None:
    """Compact 4-column status strip shown at the top of every page."""
    data = st.session_state.get("data_path")
    desc = st.session_state.get("descriptors_path")
    fp   = st.session_state.get("fingerprints_path")
    clus = st.session_state.get("clustered_path")

    def _badge(label: str, path, hint: str) -> None:
        if path and os.path.exists(str(path)):
            st.success(f"✅ **{label}**\n`{os.path.basename(str(path))}`")
        else:
            st.info(f"○ **{label}**\n{hint}")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        _badge("Data", data, "→ Data Collection")
    with c2:
        _badge("Descriptors", desc, "→ Descriptors & Fingerprints")
    with c3:
        _badge("Fingerprints", fp, "→ Descriptors & Fingerprints")
    with c4:
        _badge("Clustered", clus, "→ Clustering & Visualization")
    st.divider()


# Sidebar components
def sidebar_components():
    _init_session()
    st.sidebar.title("KIT Inhibitor QSAR Explorer")

    # ── Project settings ──────────────────────────────────────────────────
    st.sidebar.header("Project Settings")

    # Timestamp: read from active_timestamp (free key, any page can write to it).
    # Do NOT use key= here — that would bind the widget and block external writes.
    timestamp = st.sidebar.text_input(
        "Project Timestamp / Name",
        value=st.session_state.get("active_timestamp", datetime.now().strftime("%Y%m%d")),
        help=(
            "Unique identifier for this analysis run. "
            "Automatically updated when you load a previous dataset."
        ),
    )
    # Mirror the widget value back so other pages can read the current project name.
    st.session_state.active_timestamp = timestamp

    # Create a base directory for the project
    if timestamp:
        create_project_dirs(timestamp)

    # ── Target selection ──────────────────────────────────────────────────
    st.sidebar.subheader("ChEMBL Target")
    chembl_id = st.sidebar.text_input(
        "ChEMBL Target ID",
        value="CHEMBL1936",
        help="ChEMBL ID for c-KIT/KIT kinase is CHEMBL1936",
    )

    # ── Navigation ────────────────────────────────────────────────────────
    st.sidebar.header("Navigation")
    page = st.sidebar.radio(
        "Go to",
        [
            "Data Collection",
            "Descriptors & Fingerprints",
            "Clustering & Visualization",
            "Subcluster Analysis",
            "QSAR Modeling",
            "Classification Pipeline",
            "Predictive Extensions",
        ],
    )

    # ── Active session badge ───────────────────────────────────────────────
    st.sidebar.divider()
    st.sidebar.header("📂 Active Session")

    def _sb(label: str, key: str) -> None:
        path = st.session_state.get(key)
        if path and os.path.exists(str(path)):
            st.sidebar.success(f"✅ {label}")
            st.sidebar.caption(os.path.basename(str(path)))
        else:
            st.sidebar.info(f"○ {label} — not loaded")

    _sb("Data",         "data_path")
    _sb("Descriptors",  "descriptors_path")
    _sb("Fingerprints", "fingerprints_path")
    _sb("Clustered",    "clustered_path")

    return timestamp, chembl_id, page

# Data collection page
def data_collection_page(timestamp, chembl_id):
    st.header("KIT Inhibitor Data Collection")
    _session_status_bar()

    # Show banner from a previous load and immediately clear it
    if st.session_state.get("_load_banner"):
        st.success(st.session_state["_load_banner"])
        st.session_state["_load_banner"] = None

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
                                st.session_state.active_timestamp = timestamp  # keep sidebar in sync
                                
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
                        
                        # Update timestamp in session for consistency
                        st.session_state.current_timestamp = timestamp_from_path
                        st.session_state.active_timestamp = timestamp_from_path  # update sidebar

                        # Build banner message, then rerun so sidebar + status bar update immediately
                        ready_items = []
                        if st.session_state.get("descriptors_path"):
                            ready_items.append("descriptors")
                        if st.session_state.get("fingerprints_path"):
                            ready_items.append("fingerprints")
                        if st.session_state.get("clustered_path"):
                            ready_items.append("clustered data")
                        banner = (
                            f"Dataset ({timestamp_from_path}) loaded: "
                            f"{len(loaded_components)} components — "
                            + ", ".join(loaded_components)
                            + (f" | Also ready: {', '.join(ready_items)}" if ready_items else "")
                        )
                        st.session_state["_load_banner"] = banner
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error loading dataset: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
        else:
            st.info("No processed data files found. Fetch new data from ChEMBL.")

# Descriptors and fingerprints page
def descriptors_fingerprints_page(timestamp):
    st.header("Molecular Descriptors & Fingerprints")
    _session_status_bar()
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
    _session_status_bar()
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

# Subcluster analysis page
def advanced_analysis_page(timestamp):
    st.header("Subcluster Analysis")
    _session_status_bar()
    # Check for required data
    if 'data_path' not in st.session_state:
        st.warning("Please fetch or load data first from the Data Collection page")
        return
    
    # Tabs for different advanced analyses
    tab1, = st.tabs(["Subcluster Analysis"])

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
    

# ---------------------------------------------------------------------------
# QSAR Modeling page
# ---------------------------------------------------------------------------
def small_data_qsar_page(timestamp: str):
    st.header("📊 QSAR Modeling")
    _session_status_bar()
    st.markdown(
        "Train and rigorously validate QSAR models on small datasets ("
        "< 200 compounds) using PLS, SVM, Random Forest, and XGBoost with "
        "LOOCV, Repeated K-Fold, and Y-Randomization testing."
    )

    # ── Input dataset ──────────────────────────────────────────────────────
    st.subheader("1. Input Dataset")
    col1, col2 = st.columns(2)
    with col1:
        # ── Resolve CSV default: session state first, then filesystem scan ──
        _session_desc = st.session_state.get("descriptors_path") or ""
        _candidates = [
            str(_session_desc),  # what's already loaded in this session
            f"data/processed/{timestamp}/kit_descriptors_selected.csv",
            f"data/processed/{timestamp}/kit_descriptors.csv",
            f"data/processed/{timestamp}/chembl_classification_descriptors.csv",
        ]
        default_csv = next(
            (p for p in _candidates if p and os.path.exists(p)),
            f"data/processed/{timestamp}/kit_descriptors.csv",
        )
        csv_path = st.text_input(
            "Path to CSV with descriptors + target column",
            value=default_csv,
            key="qsar_csv_path",
            help=(
                "Auto-filled from your session. "
                "For regression use kit_descriptors.csv (target: pIC50); "
                "for classification use chembl_classification_descriptors.csv "
                "(target: activity_class). Run Descriptors & Fingerprints first."
            ),
        )
        # Friendly hint about loaded / available files
        if _session_desc and os.path.exists(str(_session_desc)):
            st.caption(f"📌 Loaded from session: `{os.path.basename(str(_session_desc))}`")
        else:
            _found = [p for p in _candidates[1:] if os.path.exists(p)]
            if _found:
                st.caption("Found on disk: " + "  |  ".join(
                    os.path.basename(p) for p in _found
                ))
            else:
                st.warning(
                    f"No descriptor CSV found for project **{timestamp}**. "
                    "Run the **Descriptors & Fingerprints** page first."
                )

        # ── Auto-detect task and target from the CSV columns ──────────────
        _detected_task = "regression"
        _detected_target = "pIC50"
        _csv_cols: list[str] = []
        if os.path.exists(csv_path):
            try:
                _csv_cols = pd.read_csv(csv_path, nrows=0).columns.tolist()
                if "activity_class" in _csv_cols:
                    _detected_task = "classification"
                    _detected_target = "activity_class"
                elif "pIC50" in _csv_cols:
                    _detected_task = "regression"
                    _detected_target = "pIC50"
                st.caption(
                    f"🔍 Auto-detected: **{_detected_task}** · target `{_detected_target}`  "
                    f"({len(_csv_cols)} columns total)"
                )
            except Exception:
                pass

        task = st.selectbox(
            "Task",
            ["regression", "classification"],
            index=0 if _detected_task == "regression" else 1,
            # no key= here: Streamlit widget-state cache would override index= above
        )
        # If the user manually changed task, allow a manual target override;
        # otherwise keep the auto-detected target.
        _default_target = _detected_target if task == _detected_task else (
            "pIC50" if task == "regression" else "activity_class"
        )
        # Validate that the default target actually exists in the file
        if _csv_cols and _default_target not in _csv_cols:
            _fallback_targets = [c for c in _csv_cols
                                  if c not in ["molecule_chembl_id", "canonical_smiles",
                                               "Smiles", "SMILES"]]
            _default_target = _fallback_targets[0] if _fallback_targets else _default_target
            st.warning(
                f"⚠️ Default target `{_detected_target}` not found in this file. "
                f"Suggested: `{_default_target}`. Adjust if needed."
            )
        target_col = st.text_input("Target column name", value=_default_target)
        # no key= here for same reason
    with col2:
        n_features = st.number_input("Features to select (RFE, 0 = auto)",
                                      min_value=0, max_value=500, value=0)
        optimize_svm = st.checkbox("Grid-search SVM hyperparameters", value=True)
        y_rand_trials = st.slider("Y-Randomization trials", 20, 200, 100)
        test_size = st.slider("Test set fraction", 0.1, 0.4, 0.2, 0.05)

    output_dir = f"models/{timestamp}/small_data_{task}"

    # ── Run ────────────────────────────────────────────────────────────────
    if st.button("▶ Run QSAR Pipeline", type="primary"):
        if not os.path.exists(csv_path):
            st.error(f"File not found: {csv_path}")
            return
        try:
            from small_data_qsar import SmallDataQSAR
            with st.spinner("Running feature selection, model training, and validation…"):
                qsar = SmallDataQSAR(
                    task=task,
                    n_features_to_select=int(n_features) if n_features > 0 else None,
                    optimize_svm=optimize_svm,
                    n_y_rand_trials=int(y_rand_trials),
                )
                results = qsar.fit_evaluate(
                    csv_path, target_col=target_col,
                    test_size=test_size, output_dir=output_dir
                )
            st.success(f"Pipeline complete – results saved to `{output_dir}`")

            # ── Comparison table ──────────────────────────────────────────
            st.subheader("Model Comparison")
            cmp_path = os.path.join(output_dir, "model_comparison.csv")
            if os.path.exists(cmp_path):
                st.dataframe(pd.read_csv(cmp_path, index_col=0).round(3))

            # ── Y-Randomization verdict ───────────────────────────────────
            st.subheader("Y-Randomization Results")
            for model_name, m in results.items():
                yr = m.get("y_randomization", {})
                passed = yr.get("passed", None)
                icon = "✅" if passed else "⚠️"
                st.write(f"{icon} **{model_name}**: {yr.get('verdict', 'N/A')}  "
                         f"(true score: `{yr.get('true_score', float('nan')):.3f}`, "
                         f"rand mean: `{yr.get('rand_mean', float('nan')):.3f}±"
                         f"{yr.get('rand_std', float('nan')):.3f}`)")

            # ── Plots ─────────────────────────────────────────────────────
            st.subheader("Plots")
            plot_files = sorted(glob.glob(os.path.join(output_dir, "*.png")))
            if plot_files:
                cols = st.columns(min(3, len(plot_files)))
                for i, p in enumerate(plot_files):
                    with cols[i % len(cols)]:
                        with open(p, "rb") as _f:
                            st.image(_f.read(), caption=os.path.basename(p), use_container_width=True)
            else:
                st.info("No plots generated yet.")

        except Exception as exc:
            st.error(f"Error: {exc}")

    # ── Show existing results ──────────────────────────────────────────────
    elif os.path.isdir(output_dir):
        st.info(f"Showing results from previous run in `{output_dir}`")
        cmp_path = os.path.join(output_dir, "model_comparison.csv")
        if os.path.exists(cmp_path):
            st.subheader("Model Comparison")
            st.dataframe(pd.read_csv(cmp_path, index_col=0).round(3))
        plots = sorted(glob.glob(os.path.join(output_dir, "*.png")))
        if plots:
            st.subheader("Plots")
            cols = st.columns(min(3, len(plots)))
            for i, p in enumerate(plots):
                with cols[i % len(cols)]:
                    with open(p, "rb") as _f:
                        st.image(_f.read(), caption=os.path.basename(p), use_container_width=True)


# ---------------------------------------------------------------------------
# Predictive Extensions page
# ---------------------------------------------------------------------------
def predictive_extensions_page(timestamp: str):
    st.header("🔬 Predictive Extensions")
    _session_status_bar()

    ext_tab1, ext_tab2, ext_tab3, ext_tab4 = st.tabs([
        "Applicability Domain",
        "Conformal Prediction",
        "ADMET Scoring",
        "Multi-Task QSAR",
    ])

    # ── Tab 1 – Applicability Domain ──────────────────────────────────────
    with ext_tab1:
        st.subheader("Applicability Domain (AD) Filtering")
        st.markdown(
            "Filter virtual-screening hits to compounds that lie within the "
            "chemical space covered by your training set, using Tanimoto "
            "similarity (fingerprints) or hat-matrix leverage (descriptors)."
        )

        # ── helper: sniff sep, detect header, keep only numeric feature cols ──
        _AD_META = ["molecule_chembl_id", "canonical_smiles", "Smiles", "SMILES",
                    "pIC50", "activity_class", "Activity_Level",
                    "IC50_pActivity", "Kd_pActivity", "Ki_pActivity", "Inhibition_percent"]

        def _sniff_csv(path: str) -> pd.DataFrame:
            """Read CSV/TSV robustly regardless of separator or missing header."""
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                first = fh.readline()
            sep = "\t" if first.count("\t") > first.count(",") else ","
            # Detect whether first row is a header (non-numeric dominant row)
            parts = first.strip().split(sep)
            has_header = not all(_p.replace(".", "").replace("-", "").replace("e", "").isdigit()
                                 for _p in parts if _p)
            return pd.read_csv(path, sep=sep, header=0 if has_header else None)

        def _preview_csv(path: str, label: str) -> tuple[pd.DataFrame | None, list[str]]:
            """Show column info and return (df, feature_cols) or (None, []) on error."""
            if not os.path.exists(path):
                st.warning(f"⚠️ {label}: file not found — `{path}`")
                return None, []
            try:
                df = _sniff_csv(path)
                num_cols = [c for c in df.columns
                            if c not in _AD_META and pd.api.types.is_numeric_dtype(df[c])]
                str_cols = [c for c in df.columns if c not in num_cols]
                with st.expander(f"📋 {label} — {len(df)} rows, "
                                 f"{len(num_cols)} numeric feature cols", expanded=False):
                    c_a, c_b = st.columns(2)
                    with c_a:
                        st.markdown("**Numeric feature columns (used):**")
                        st.code("\n".join(str(c) for c in num_cols[:20])
                                + ("\n…" if len(num_cols) > 20 else ""))
                    with c_b:
                        st.markdown("**Non-numeric / meta columns (excluded):**")
                        st.code("\n".join(str(c) for c in str_cols[:20])
                                + ("\n…" if len(str_cols) > 20 else ""))
                    st.dataframe(df.head(3))
                if len(num_cols) == 0:
                    st.error(
                        f"❌ **{label}**: no numeric feature columns found.  \n"
                        "This looks like a raw SMILES file. "
                        "The AD training set must be a **descriptor CSV** "
                        "(e.g. `chembl_classification_descriptors.csv` or `kit_descriptors_selected.csv`). "
                        "The query set must also have the same feature columns."
                    )
                else:
                    st.success(f"✅ **{label}**: {len(num_cols)} numeric features ready.")
                return df, num_cols
            except Exception as exc:
                st.error(f"❌ {label}: could not read file — {exc}")
                return None, []

        def _load_X_from_df(df: pd.DataFrame, feat_cols: list[str]) -> np.ndarray:
            arr = df[feat_cols].fillna(0).values.astype(float)
            return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

        ad_method = st.selectbox("AD method", ["tanimoto", "leverage"])
        ad_threshold = st.slider("Tanimoto threshold (tanimoto only)", 0.1, 0.9, 0.4, 0.05)
        ad_k = st.number_input("k-NN (tanimoto only)", 1, 20, 5)

        # ── Auto-fill training CSV from session state ──────────────────────
        _ad_train_cands = [
            st.session_state.get("descriptors_path") or "",
            f"data/processed/{timestamp}/chembl_classification_descriptors.csv",
            f"data/processed/{timestamp}/chembl_classification_fingerprints.csv",
            f"data/processed/{timestamp}/kit_descriptors_selected.csv",
            f"data/processed/{timestamp}/kit_descriptors.csv",
        ]
        _ad_train_default = next(
            (p for p in _ad_train_cands if p and os.path.exists(str(p))),
            f"data/processed/{timestamp}/chembl_classification_descriptors.csv",
        )

        col1, col2 = st.columns(2)
        with col1:
            train_csv = st.text_input(
                "Training set CSV (descriptors / fingerprints)",
                value=_ad_train_default,
                key="ad_train",
            )
        with col2:
            query_csv = st.text_input(
                "Query / screening CSV (same feature columns)",
                value=(
                    st.session_state.get("screening_results_path")
                    or f"predictions/{timestamp}/blind_screening/all_predictions.csv"
                ),
                key="ad_query",
            )

        # ── Pre-flight previews ────────────────────────────────────────────
        tr_df, tr_feat = _preview_csv(train_csv,  "Training set")
        q_df,  q_feat  = _preview_csv(query_csv,  "Query set")

        # Warn about column mismatch
        if tr_feat and q_feat:
            common = [c for c in tr_feat if c in q_feat]
            only_tr = [c for c in tr_feat if c not in q_feat]
            only_q  = [c for c in q_feat  if c not in tr_feat]
            if only_tr or only_q:
                st.warning(
                    f"⚠️ Column mismatch: **{len(common)}** shared, "
                    f"{len(only_tr)} only in training, {len(only_q)} only in query.  \n"
                    "AD will use only the shared columns."
                )

        _ad_ready = bool(tr_feat and q_feat and
                         [c for c in tr_feat if c in q_feat])

        if st.button("Run AD Analysis", key="run_ad", disabled=not _ad_ready):
            try:
                from predictive_extensions import ApplicabilityDomain

                common_feat = [c for c in tr_feat if c in q_feat]
                with st.spinner("Computing AD…"):
                    X_tr = _load_X_from_df(tr_df, common_feat)   # type: ignore[arg-type]
                    X_q  = _load_X_from_df(q_df,  common_feat)   # type: ignore[arg-type]
                    ad = ApplicabilityDomain(method=ad_method,
                                            threshold=float(ad_threshold), k=int(ad_k))
                    ad.fit(X_tr)
                    mask = ad.predict(X_q)

                df_q = q_df.copy()                               # type: ignore[union-attr]
                df_q["inside_AD"] = mask
                inside_n  = int(mask.sum())
                outside_n = int((~mask).sum())
                st.success(f"AD computed – **{inside_n} inside** / {outside_n} outside  "
                           f"({100*inside_n/len(mask):.1f}% coverage)")

                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Inside AD", inside_n)
                with col_b:
                    st.metric("Outside AD", outside_n)

                st.subheader("AD-filtered compounds (inside AD)")
                st.dataframe(df_q[df_q["inside_AD"]].head(50))

                out_csv = f"predictions/{timestamp}/ad_filtered.csv"
                os.makedirs(os.path.dirname(out_csv), exist_ok=True)
                df_q.to_csv(out_csv, index=False)
                st.info(f"Full results saved to `{out_csv}`")
            except Exception as exc:
                st.error(f"Error: {exc}")
                import traceback; st.code(traceback.format_exc())

    # ── Tab 2 – Conformal Prediction ──────────────────────────────────────
    with ext_tab2:
        st.subheader("Conformal Prediction Intervals")
        st.markdown(
            "Wrap any trained model with **Inductive Conformal Prediction** "
            "to obtain statistically guaranteed prediction intervals (regression) "
            "or prediction sets (classification)."
        )
        cp_model_dir = st.text_input(
            "Trained model directory",
            value=f"models/{timestamp}/small_data_regression",
            key="cp_model_dir"
        )
        cp_model_type = st.selectbox("Model file prefix",
                                     ["random_forest", "xgboost", "svm", "pls"],
                                     key="cp_model_type")
        cp_task = st.selectbox("Task", ["regression", "classification"], key="cp_task")
        cp_alpha = st.slider("Significance level α (1-α = coverage)", 0.05, 0.3, 0.10, 0.01,
                             key="cp_alpha")
        cp_cal_csv = st.text_input(
            "Calibration set CSV",
            value=f"data/processed/{timestamp}/chembl_classification_descriptors.csv",
            key="cp_cal"
        )
        cp_target = st.text_input("Target column",
                                   value="pIC50" if cp_task == "regression" else "activity_class",
                                   key="cp_target")
        cp_query_csv = st.text_input("Query CSV", value="blind_set/blind_set.csv",
                                     key="cp_query")

        if st.button("Run Conformal Predictor", key="run_cp"):
            model_path = os.path.join(cp_model_dir, f"{cp_model_type}_model.pkl")
            if not os.path.exists(model_path):
                st.error(f"Model not found: {model_path}"); return  # type: ignore[return-value]
            for p in [cp_cal_csv, cp_query_csv]:
                if not os.path.exists(p):
                    st.error(f"File not found: {p}"); return  # type: ignore[return-value]
            try:
                from predictive_extensions import ConformalPredictor
                _excl = ["molecule_chembl_id", "canonical_smiles", "Smiles",
                         "pIC50", "activity_class", "Activity_Level"]

                def load_Xy(path: str, tgt: str):
                    df = pd.read_csv(path)
                    feat_cols = [c for c in df.columns if c not in _excl and c != tgt]
                    X = df[feat_cols].fillna(0).values.astype(float)
                    y = df[tgt].values if tgt in df.columns else np.zeros(len(df))
                    return X, y

                with open(model_path, "rb") as fh:
                    base_model = pickle.load(fh)

                scaler_path = os.path.join(cp_model_dir, "scaler.pkl")
                with open(scaler_path, "rb") as fh:
                    scaler = pickle.load(fh)

                with st.spinner("Calibrating conformal predictor…"):
                    X_cal, y_cal = load_Xy(cp_cal_csv, cp_target)
                    X_q,  _     = load_Xy(cp_query_csv, cp_target)

                    if cp_task == "classification":
                        from sklearn.preprocessing import LabelEncoder
                        le = LabelEncoder()
                        y_cal_enc = le.fit_transform(y_cal)
                    else:
                        y_cal_enc = y_cal.astype(float)  # type: ignore[assignment]

                    X_cal_sc = scaler.transform(X_cal)
                    X_q_sc   = scaler.transform(X_q)

                    cp_obj = ConformalPredictor(base_model, task=cp_task, alpha=cp_alpha)
                    cp_obj.calibrate(X_cal_sc, y_cal_enc)
                    preds = cp_obj.predict(X_q_sc)

                st.success("Conformal prediction complete!")
                st.dataframe(preds.head(50))

                out_csv = f"predictions/{timestamp}/conformal_predictions.csv"
                os.makedirs(os.path.dirname(out_csv), exist_ok=True)
                preds.to_csv(out_csv, index=False)
                st.info(f"Saved to `{out_csv}`")

                if cp_task == "regression":
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.hist(preds["interval_width"], bins=30, edgecolor="k", alpha=0.7)
                    ax.set_xlabel("Prediction interval width")
                    ax.set_ylabel("Count")
                    ax.set_title(f"Conformal interval widths (α={cp_alpha})")
                    st.pyplot(fig)
                    plt.close(fig)
            except Exception as exc:
                st.error(f"Error: {exc}")

    # ── Tab 3 – ADMET Scoring ─────────────────────────────────────────────
    with ext_tab3:
        st.subheader("ADMET Scoring & Multi-Objective Ranking")
        st.markdown(
            "Query **SwissADME** or **pkCSM** for ADMET properties, then combine "
            "them with a QSAR activity score into a single composite ranking."
        )

        admet_source = st.radio("API source", ["SwissADME", "pkCSM"], horizontal=True)
        admet_results_csv = st.text_input(
            "CSV with SMILES + QSAR score column",
            value=(
                st.session_state.get("screening_results_path")
                or f"predictions/{timestamp}/blind_screening/all_predictions.csv"
            ),
            key="admet_csv"
        )
        smiles_col_admet = st.text_input("SMILES column name", value="SMILES", key="admet_smiles_col")
        qsar_col_admet   = st.text_input("QSAR score column", value="Prob_1", key="admet_qsar_col")
        delay = st.slider("Delay between API calls (s)", 0.5, 5.0, 1.0, 0.5)

        if st.button("Fetch ADMET Properties", key="run_admet"):
            if not os.path.exists(admet_results_csv):
                st.error(f"File not found: {admet_results_csv}"); return  # type: ignore[return-value]
            try:
                from predictive_extensions import AdmetFilter
                df_in = pd.read_csv(admet_results_csv)
                if smiles_col_admet not in df_in.columns:
                    st.error(f"Column '{smiles_col_admet}' not found."); return  # type: ignore[return-value]
                smiles_list = df_in[smiles_col_admet].dropna().tolist()
                limit = st.session_state.get("admet_limit", 20)
                smiles_list = smiles_list[:limit]

                admet = AdmetFilter(request_delay=delay)
                with st.spinner(f"Querying {admet_source} for {len(smiles_list)} compounds…"):
                    if admet_source == "SwissADME":
                        df_admet = admet.query_swissadme(smiles_list)
                    else:
                        df_admet = admet.query_pkcsm(smiles_list)

                st.dataframe(df_admet.head(20))

                # Merge with input and rank
                df_merged = df_in.merge(df_admet, on=smiles_col_admet, how="left") if smiles_col_admet in df_admet.columns else df_admet

                if qsar_col_admet in df_merged.columns:
                    st.subheader("Multi-Objective Ranking")
                    admet_extra = [c for c in df_admet.columns
                                   if c != smiles_col_admet and c in df_merged.columns]
                    ranked = admet.rank_compounds(
                        df_merged, qsar_col=qsar_col_admet,
                        admet_cols=admet_extra[:5] if admet_extra else None
                    )
                    st.dataframe(ranked[[smiles_col_admet, qsar_col_admet, "composite_score"]].head(20))
                    out_csv = f"predictions/{timestamp}/admet_ranked.csv"
                    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
                    ranked.to_csv(out_csv, index=False)
                    st.info(f"Saved to `{out_csv}`")
            except Exception as exc:
                st.error(f"Error: {exc}")

        st.caption(
            "⚠️ ADMET queries use public APIs (SwissADME, pkCSM). "
            "Limit batch size to avoid overwhelming servers. Default: first 20 compounds."
        )
        st.number_input("Max compounds to query", 1, 200, 20, key="admet_limit")

    # ── Tab 4 – Multi-Task QSAR ───────────────────────────────────────────
    with ext_tab4:
        st.subheader("Multi-Task QSAR")
        st.markdown(
            "Train a single model simultaneously on **multiple activity targets**, "
            "sharing structural representations to compensate for small per-target datasets."
        )

        mt_csv = st.text_input(
            "CSV with descriptors + multiple target columns",
            value=f"data/processed/{timestamp}/chembl_merged_all_data.csv",
            key="mt_csv"
        )
        mt_task = st.selectbox("Task", ["regression", "classification"], key="mt_task")
        mt_base = st.selectbox("Base model", ["rf", "xgb", "svm"], key="mt_base")
        mt_targets_raw = st.text_input(
            "Target columns (comma-separated)",
            value="IC50_pActivity,Ki_pActivity,Kd_pActivity",
            key="mt_targets"
        )
        mt_n_splits = st.slider("CV folds", 2, 10, 5, key="mt_cv")
        mt_n_repeats = st.slider("CV repeats", 1, 5, 3, key="mt_rep")
        mt_output = f"models/{timestamp}/multitask"

        if st.button("Train Multi-Task Model", key="run_mt"):
            if not os.path.exists(mt_csv):
                st.error(f"File not found: {mt_csv}"); return  # type: ignore[return-value]
            try:
                from predictive_extensions import MultiTaskQSAR
                target_cols = [t.strip() for t in mt_targets_raw.split(",") if t.strip()]
                df_mt = pd.read_csv(mt_csv)
                missing = [c for c in target_cols if c not in df_mt.columns]
                if missing:
                    st.error(f"Target columns not found: {missing}"); return  # type: ignore[return-value]

                _excl = ["molecule_chembl_id", "canonical_smiles", "Smiles",
                         "activity_class", "Activity_Level"] + target_cols
                feat_cols = [c for c in df_mt.columns if c not in _excl]
                X = df_mt[feat_cols].fillna(0).values.astype(float)
                Y = df_mt[target_cols].fillna(0)

                with st.spinner("Training multi-task model…"):
                    mt_model = MultiTaskQSAR(task=mt_task, base_model=mt_base)
                    mt_model.fit(X, Y, target_names=target_cols)

                st.success(f"Multi-task model trained on {X.shape[0]} compounds, "
                           f"{len(target_cols)} targets.")

                with st.spinner("Cross-validating…"):
                    cv_results = mt_model.cross_validate(
                        X, Y, n_splits=mt_n_splits, n_repeats=mt_n_repeats
                    )
                st.subheader("Per-Target CV Results")
                st.dataframe(cv_results.round(3))

                mt_model.save(mt_output)
                st.info(f"Model saved to `{mt_output}/multitask_model.pkl`")

                # Bar chart of per-target performance
                metric = "mean_r2" if mt_task == "regression" else "mean_f1_weighted"
                if metric in cv_results.columns:
                    fig, ax = plt.subplots(figsize=(max(6, len(target_cols)*1.5), 4))
                    ax.bar(cv_results.index, cv_results[metric], alpha=0.8)
                    ax.set_ylabel(metric)
                    ax.set_title("Per-Target CV Performance")
                    plt.xticks(rotation=20, ha="right")
                    fig.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
            except Exception as exc:
                st.error(f"Error: {exc}")
        
# ---------------------------------------------------------------------------
# Classification Pipeline page
# ---------------------------------------------------------------------------
def classification_pipeline_page(timestamp: str) -> None:
    st.header("🧬 Classification Pipeline")
    _session_status_bar()
    st.markdown(
        "Prepare ChEMBL classification datasets, train binary activity classifiers "
        "(RF / XGBoost), and screen a blind set. "
        "Outputs feed directly into **Predictive Extensions** (AD, ADMET, Conformal)."
    )

    tab1, tab2, tab3 = st.tabs([
        "1 · Prepare Data",
        "2 · Train Classifiers",
        "3 · Screen Compounds",
    ])

    # ── Tab 1 – Prepare classification dataset ────────────────────────────────
    with tab1:
        st.subheader("Prepare Classification Dataset")
        st.markdown(
            "Loads `known_compounds/*.csv`, merges IC50 / Ki / Kd / Inhibition "
            "activity data, assigns binary activity classes, then generates "
            "molecular descriptors or Morgan fingerprints."
        )

        kc_dir = "known_compounds"
        required_files = {
            "chembl_all_class.csv":  os.path.join(kc_dir, "chembl_all_class.csv"),
            "chembl_IC50.csv":       os.path.join(kc_dir, "chembl_IC50.csv"),
            "chembl_Ki.csv":         os.path.join(kc_dir, "chembl_Ki.csv"),
            "chembl_Kd.csv":         os.path.join(kc_dir, "chembl_Kd.csv"),
            "chembl_Inhibition.csv": os.path.join(kc_dir, "chembl_Inhibition.csv"),
        }

        col_status, col_cfg = st.columns(2)
        with col_status:
            st.markdown("**Required input files:**")
            all_present = True
            for fname, fpath in required_files.items():
                if os.path.exists(fpath):
                    st.success(f"✅ {fname}")
                else:
                    st.error(f"❌ {fname} — missing!")
                    all_present = False
        with col_cfg:
            feature_type = st.radio(
                "Feature type",
                ["Descriptors (RDKit)", "Fingerprints (Morgan ECFP4)"],
                key="cls_feature_type",
            )
            use_fp = feature_type.startswith("Fingerprints")

        if not all_present:
            st.warning(
                "Some source files are missing from `known_compounds/`. "
                "Ensure all ChEMBL CSV files are present before running."
            )

        # Show what already exists
        out_merged = f"data/processed/{timestamp}/chembl_merged_all_data.csv"
        out_desc   = f"data/processed/{timestamp}/chembl_classification_descriptors.csv"
        out_fp_csv = f"data/processed/{timestamp}/chembl_classification_fingerprints.csv"
        existing_out = []
        for p in [out_merged, out_desc, out_fp_csv]:
            if os.path.exists(p):
                existing_out.append(f"✅ `{os.path.basename(p)}` ({os.path.getsize(p)//1024} KB)")
        if existing_out:
            st.info("**Already generated for** `" + timestamp + "`:  " + "  |  ".join(existing_out))

        if st.button("▶ Prepare Classification Data", type="primary", key="run_cls_prep",
                     disabled=not all_present):
            try:
                import prepare_and_train_classification as pat
                pat.timestamp = timestamp  # point module at current project

                with st.spinner("Loading and merging ChEMBL data…"):
                    merged_df = pat.load_and_merge_data(kc_dir)
                    _out_dir = f"data/processed/{timestamp}"
                    os.makedirs(_out_dir, exist_ok=True)
                    merged_df.to_csv(out_merged, index=False)

                st.success(f"✅ Merged **{len(merged_df)}** compounds → `{out_merged}`")
                cls_dist = merged_df["activity_class"].value_counts().to_dict()
                st.write("**Activity class distribution:**", cls_dist)

                with st.spinner("Generating features (this may take a few minutes)…"):
                    if use_fp:
                        final_df = pat.generate_fingerprints(merged_df)
                        out_path = out_fp_csv
                    else:
                        final_df = pat.generate_molecular_descriptors(merged_df)
                        out_path = out_desc

                    if "Smiles" in final_df.columns:
                        final_df.rename(columns={"Smiles": "canonical_smiles"}, inplace=True)
                    final_df.to_csv(out_path, index=False)

                feat_cols = [
                    c for c in final_df.columns
                    if c not in ["canonical_smiles", "molecule_chembl_id", "activity_class"]
                    and not ("IC50" in c or "Kd" in c or "Ki" in c or "Inhibition" in c)
                ]
                st.success(
                    f"✅ Saved **{len(final_df)}** compounds × **{len(feat_cols)}** features → `{out_path}`"
                )

                st.session_state.descriptors_path = out_path
                st.session_state["_load_banner"] = (
                    f"Classification data prepared ({timestamp}): "
                    f"{len(final_df)} compounds, {len(feat_cols)} features"
                )
                st.rerun()

            except Exception as exc:
                st.error(f"Error: {exc}")
                import traceback; st.code(traceback.format_exc())

    # ── Tab 2 – Train classifiers ─────────────────────────────────────────────
    with tab2:
        st.subheader("Train Classification Models")

        # Candidate list: classification-specific filenames first (they always have
        # activity_class), then fall back to session descriptors_path only if it
        # actually contains the activity_class column.
        _sess_desc = st.session_state.get("descriptors_path") or ""
        _cls_specific = [
            f"data/processed/{timestamp}/chembl_classification_descriptors.csv",
            f"data/processed/{timestamp}/chembl_classification_fingerprints.csv",
        ]
        # Accept session path only when it contains activity_class
        def _has_activity_class(p: str) -> bool:
            try:
                return "activity_class" in pd.read_csv(p, nrows=0).columns.tolist()
            except Exception:
                return False

        _cls_cands = [p for p in _cls_specific if os.path.exists(p)]
        if _sess_desc and os.path.exists(str(_sess_desc)) and _has_activity_class(str(_sess_desc)):
            _cls_cands.insert(0, str(_sess_desc))

        _cls_default = (
            _cls_cands[0]
            if _cls_cands
            else f"data/processed/{timestamp}/chembl_classification_descriptors.csv"
        )
        cls_data_path = st.text_input(
            "Prepared dataset (CSV with descriptors or fingerprints)",
            value=_cls_default,
            # no key= — widget-state cache would lock in the old regression path
        )
        if _cls_cands:
            st.caption(f"📌 Auto-selected: `{os.path.basename(_cls_default)}`")

        # Pre-flight: verify the file actually contains activity_class
        if os.path.exists(cls_data_path):
            try:
                _cls_cols = pd.read_csv(cls_data_path, nrows=0).columns.tolist()
                if "activity_class" in _cls_cols:
                    st.success("✅ `activity_class` column found — file is ready for classification training.")
                elif "pIC50" in _cls_cols and "activity_class" not in _cls_cols:
                    st.error(
                        "❌ This file contains `pIC50` (regression) but no `activity_class` column.  \n"
                        "Run **Tab 1 · Prepare Data** first to generate a classification dataset "
                        "(`chembl_classification_descriptors.csv`)."
                    )
            except Exception:
                pass

        col1, col2 = st.columns(2)
        with col1:
            train_rf  = st.checkbox("Random Forest", value=True,  key="cls_train_rf")
            train_xgb = st.checkbox("XGBoost",       value=True,  key="cls_train_xgb")
        with col2:
            cls_optimize = st.checkbox(
                "Hyperparameter optimisation (slower)", value=False, key="cls_optimize"
            )

        model_out_dir = f"models/{timestamp}/classification"
        existing_models = sorted(glob.glob(f"{model_out_dir}/*.pkl"))
        if existing_models:
            st.info(
                f"**Existing models in** `{model_out_dir}`:  "
                + "  |  ".join(os.path.basename(f) for f in existing_models)
            )

        if st.button("▶ Train Classifiers", type="primary", key="run_cls_train"):
            if not os.path.exists(cls_data_path):
                st.error(f"Data file not found: {cls_data_path}")
            elif not (train_rf or train_xgb):
                st.warning("Select at least one model type.")
            else:
                try:
                    from classification_model import build_classification_model
                    os.makedirs(model_out_dir, exist_ok=True)
                    trained: dict = {}

                    for model_key, should_train, label in [
                        ("rf",  train_rf,  "Random Forest"),
                        ("xgb", train_xgb, "XGBoost"),
                    ]:
                        if not should_train:
                            continue
                        with st.spinner(f"Training {label}…"):
                            try:
                                _m, _s, _e, metrics = build_classification_model(
                                    cls_data_path,
                                    activity_column="activity_class",
                                    model_type=model_key,
                                    optimize=cls_optimize,
                                    output_dir=model_out_dir,
                                )
                                if metrics:
                                    trained[label] = metrics
                                    st.success(f"✅ {label} trained")
                            except Exception as exc:
                                st.error(f"Error training {label}: {exc}")

                    if trained:
                        st.subheader("Model Comparison")
                        cmp_df = pd.DataFrame(trained).T.round(3)
                        st.dataframe(cmp_df)
                        cmp_df.to_csv(f"{model_out_dir}/model_comparison.csv")

                        plots = sorted(glob.glob(f"{model_out_dir}/*.png"))
                        if plots:
                            st.subheader("Training Plots")
                            _pc = st.columns(min(3, len(plots)))
                            for i, p in enumerate(plots):
                                with _pc[i % len(_pc)]:
                                    with open(p, "rb") as _f:
                                        st.image(_f.read(), caption=os.path.basename(p),
                                                 use_container_width=True)

                        st.session_state["classification_model_dir"] = model_out_dir
                        # Don't st.rerun() here — images would expire before the
                        # new render fetches them. Session state is already updated;
                        # sidebar badge refreshes on next user interaction.
                        st.info(f"Models saved to `{model_out_dir}`. "
                                "Navigate to **Screen Compounds** tab to screen a blind set.")

                except Exception as exc:
                    st.error(f"Error: {exc}")
                    import traceback; st.code(traceback.format_exc())

    # ── Tab 3 – Screen blind set ───────────────────────────────────────────────
    with tab3:
        st.subheader("Screen Blind Set")
        st.markdown(
            "Load a trained classifier and screen a compound library. "
            "The output `all_predictions.csv` feeds directly into "
            "**Predictive Extensions → ADMET Scoring** and **Applicability Domain**."
        )

        _default_model_dir = (
            st.session_state.get("classification_model_dir")
            or f"models/{timestamp}/classification"
        )

        col1, col2 = st.columns(2)
        with col1:
            screen_model_dir = st.text_input(
                "Model directory",
                value=_default_model_dir,
                key="screen_model_dir",
            )
            _pkl_files = sorted(glob.glob(f"{screen_model_dir}/*.pkl")) \
                if os.path.isdir(screen_model_dir) else []
            _model_opts = [
                os.path.basename(f).replace("_classifier.pkl", "").replace("_model.pkl", "")
                for f in _pkl_files
            ] or ["xgb", "rf"]
            screen_model_type = st.selectbox(
                "Model type", _model_opts, key="screen_model_type"
            )

        with col2:
            screen_blind_file = st.text_input(
                "Blind set file (CSV with SMILES column)",
                value=st.session_state.get("data_path") or "blind_set/blind_set.csv",
                key="screen_blind_file",
            )
            screen_target_class = st.text_input(
                "Target class label (active compounds)",
                value="1",
                key="screen_target_cls",
                help="Use '1' for active, '0' for inactive. "
                     "Match the class labels in your training data.",
            )
            screen_confidence = st.slider(
                "Confidence threshold",
                0.5, 0.95, 0.7, 0.05,
                key="screen_conf",
            )

        screen_out_dir = f"predictions/{timestamp}/blind_screening"
        st.caption(f"Results saved to: `{screen_out_dir}/all_predictions.csv`")

        # Show existing results if any
        _existing_res = f"{screen_out_dir}/all_predictions.csv"
        if os.path.exists(_existing_res):
            try:
                _prev = pd.read_csv(_existing_res)
                st.info(
                    f"ℹ️ Previous run: **{len(_prev)}** compounds in "
                    f"`{_existing_res}` — run again to overwrite."
                )
            except Exception:
                pass

        if st.button("▶ Screen Compounds", type="primary", key="run_screen"):
            if not os.path.isdir(screen_model_dir):
                st.error(f"Model directory not found: {screen_model_dir}")
            elif not os.path.exists(screen_blind_file):
                st.error(f"Blind set file not found: {screen_blind_file}")
            else:
                try:
                    from classification_prediction_tool import load_classifier

                    with st.spinner(f"Loading `{screen_model_type}` classifier…"):
                        classifier = load_classifier(
                            screen_model_dir, model_type=screen_model_type
                        )
                    st.success("✅ Classifier loaded")

                    os.makedirs(screen_out_dir, exist_ok=True)

                    with st.spinner(
                        f"Screening `{os.path.basename(screen_blind_file)}`…"
                    ):
                        results = classifier.screen_blind_set(
                            screen_blind_file,
                            output_dir=screen_out_dir,
                            target_class=str(screen_target_class),
                            confidence_threshold=screen_confidence,
                        )

                    if results is not None:
                        prob_col = f"Prob_{screen_target_class}"
                        high_conf_mask = results.get(
                            "High_Confidence",
                            pd.Series([False] * len(results))
                        )
                        n_hits = int(high_conf_mask.sum())

                        col_a, col_b, col_c = st.columns(3)
                        with col_a: st.metric("Total screened", len(results))
                        with col_b: st.metric("High-confidence hits", n_hits)
                        with col_c:
                            if prob_col in results.columns:
                                st.metric(
                                    "Mean probability",
                                    f"{results[prob_col].mean():.3f}"
                                )

                        st.subheader("Top 20 Hits")
                        _disp = [c for c in
                                 ["SMILES", "compound_id", "Predicted_Class",
                                  prob_col, "High_Confidence"]
                                 if c in results.columns]
                        st.dataframe(results[_disp].head(20))

                        with open(_existing_res, "rb") as fh:
                            st.download_button(
                                "⬇ Download all_predictions.csv",
                                fh, "all_predictions.csv", "text/csv",
                            )

                        st.session_state["screening_results_path"] = _existing_res
                        st.session_state["_load_banner"] = (
                            f"Screening complete: {len(results)} compounds, "
                            f"{n_hits} hits → `{_existing_res}`"
                        )
                        st.rerun()
                    else:
                        st.error("Screening returned no results.")

                except Exception as exc:
                    st.error(f"Error: {exc}")
                    import traceback; st.code(traceback.format_exc())


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
    elif page == "Subcluster Analysis":
        advanced_analysis_page(timestamp)
    elif page == "QSAR Modeling":
        small_data_qsar_page(timestamp)
    elif page == "Classification Pipeline":
        classification_pipeline_page(timestamp)
    elif page == "Predictive Extensions":
        predictive_extensions_page(timestamp)

if __name__ == "__main__":
    main()
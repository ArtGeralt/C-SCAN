# Chemical Space and Cluster-Specific Analysis of Small ChEMBL Datasets

> **Hello! This is my first GitHub project!** As a student diving into cheminformatics and computational chemistry, I've put together this tool to apply what I'm learning about molecular descriptors, clustering algorithms, and chemical space analysis.

A comprehensive tool for analyzing chemical space and building cluster-specific models from small bioactivity datasets retrieved from ChEMBL. This project represents my exploration into the fascinating world of computational drug discovery, combining concepts from medicinal chemistry, machine learning, and data visualization.

## Overview & Learning Journey

This tool addresses the challenge of working with small bioactivity datasets by leveraging chemical space analysis and cluster-specific modeling approaches. Throughout this project, I've applied course knowledge in:

- **Molecular Informatics**: Understanding how to represent chemical structures computationally
- **Statistical Analysis**: Implementing PCA, t-SNE, and correlation analysis for data exploration
- **Machine Learning**: Applying clustering algorithms and exploring their applications in chemistry
- **Data Visualization**: Creating meaningful plots to communicate chemical insights
- **Software Development**: Building a complete pipeline with proper documentation and user interface

This provides a complete pipeline from data collection to advanced analysis, all accessible through an intuitive web interface.

## Key Features

### Data Collection & Processing
- **Automated ChEMBL retrieval** with duplicate removal and data validation
- **Flexible target selection** supporting various protein targets
- **Comprehensive data preprocessing** with activity standardization (pIC50)

### Molecular Characterization
- **30+ molecular descriptors** across multiple categories (physicochemical, topological, electronic)
- **Morgan fingerprints** for structural similarity analysis
- **Kinase-specific features** for targeted analysis

### Chemical Space Analysis
- **Advanced clustering** using K-means with standardized features
- **Dimensionality reduction** with PCA and t-SNE
- **Interactive visualizations** including 2D/3D chemical space maps
- **Activity landscape analysis** with structure-activity relationships

### Advanced Analytics
- **Subcluster analysis** for detailed chemical space decomposition
- **Scaffold analysis** with Murcko scaffolds
- **Property-activity correlations** with statistical validation
- **Cluster-specific modeling** for improved predictions on small datasets

### Interactive Web Interface
- **Streamlit-based GUI** with organized workflow tabs
- **Real-time progress tracking** for all analysis steps
- **Export capabilities** for results and visualizations
- **Methodology explanations** integrated throughout the interface

## Quick Start

### Installation

```bash
git clone https://github.com/your-username/qsar-chemical-space-analysis.git
cd qsar-chemical-space-analysis
pip install -r requirements.txt
```

### Launch the Application

```bash
streamlit run app.py
```

Navigate to `http://localhost:8501` in your web browser.

## Usage Guide

### 1. Data Collection
- Enter a ChEMBL target ID (e.g., CHEMBL1936 for c-Kit)
- Set minimum activity threshold
- Click "Fetch Data" to retrieve and process bioactivity data

### 2. Molecular Analysis
- Generate molecular descriptors and fingerprints
- Review descriptor distributions and correlations
- Export processed data for further analysis

### 3. Chemical Space Clustering
- Configure clustering parameters (number of clusters, random state)
- Visualize clusters in 2D/3D chemical space
- Analyze cluster representatives and diversity

### 4. Advanced Analysis
- Perform subcluster analysis for detailed decomposition
- Generate scaffold analysis and activity landscapes
- Explore property-activity correlations

## Technical Architecture

### Core Components

- **`app.py`**: Main Streamlit application with multi-tab interface
- **`data_collection.py`**: ChEMBL data retrieval and preprocessing
- **`descriptors.py`**: Molecular descriptor calculation engine
- **`fingerprints_clustering.py`**: Fingerprint generation and clustering
- **`visualization.py`**: Comprehensive plotting and visualization suite
- **`subcluster_analysis.py`**: Advanced cluster decomposition tools

### Key Dependencies

- **RDKit**: Molecular informatics and descriptor calculation
- **Streamlit**: Interactive web application framework
- **ChEMBL WebResource Client**: Bioactivity data access
- **Scikit-learn**: Machine learning and clustering algorithms
- **Plotly/Matplotlib/Seaborn**: Visualization libraries

## Example Analysis Workflow

1. **Data Retrieval**: Fetch c-Kit inhibitor data from ChEMBL
2. **Descriptor Generation**: Calculate physicochemical and structural descriptors
3. **Clustering Analysis**: Identify 5 distinct chemical clusters
4. **Subcluster Decomposition**: Further analyze cluster 3 into subclusters
5. **Activity Landscape**: Generate 3D visualization of structure-activity relationships
6. **Correlation Analysis**: Identify key molecular properties driving activity

## Specialized for Small Datasets

This tool is specifically designed for scenarios where:
- Limited bioactivity data is available (< 3000 compounds)
- Maximum insight extraction is crucial
- Cluster-specific patterns may provide better predictions than global models
- Chemical space understanding is essential for lead optimization

## Output and Results

### Generated Files
- **Processed datasets** with descriptors and fingerprints
- **Cluster assignments** and representative structures
- **Visualization exports** (HTML, PNG formats)
- **Analysis reports** with statistical summaries

### Visualizations
- Chemical space maps (2D PCA, t-SNE projections)
- Activity landscapes (3D structure-activity relationships)
- Correlation heatmaps and property distributions
- Cluster analysis plots and dendrograms

## Future Development Plans

As I continue my studies in computational chemistry and machine learning, I plan to expand this project with:

### Machine Learning Enhancements
- **QSAR Model Development**: Implementing regression and classification models for activity prediction
- **Advanced Algorithms**: Exploring random forests, gradient boosting, and neural networks
- **Model Validation**: Cross-validation strategies and performance metrics for chemical datasets
- **Feature Selection**: Automated descriptor selection and importance ranking

### Enhanced Analytics
- **Predictive Modeling**: Compound activity prediction with confidence intervals
- **Virtual Screening**: Tools for filtering large compound libraries
- **ADMET Predictions**: Absorption, distribution, metabolism, excretion, and toxicity modeling
- **Multi-target Analysis**: Simultaneous analysis of multiple biological targets

### Advanced Cheminformatics
- **3D Molecular Descriptors**: Incorporating conformational and pharmacophore features
- **Fragment-based Analysis**: Molecular fragment contributions to activity
- **Similarity Searching**: Advanced similarity metrics and chemical space navigation

This project serves as my foundation for understanding how computational methods can accelerate drug discovery processes!

## Contributing

As a learning project, I welcome feedback, suggestions, and contributions! Whether you're a fellow student or an experienced researcher, feel free to submit issues, feature requests, or pull requests. I'm always eager to learn from the community!

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments & Learning Resources

- **ChEMBL Database** for providing open bioactivity data that made this project possible
- **ChEMBL WebResource Client** developers for creating such an elegant Python interface to access ChEMBL data programmatically
- **RDKit** community for incredible molecular informatics tools and documentation
- **Streamlit** team for making web app development accessible to scientists
- **My coursework instructors** for introducing me to the fascinating world of computational chemistry
- **Open-source community** for countless tutorials, examples, and inspiration

Special thanks to the educational resources that helped me understand:
- Molecular descriptor theory and applications
- Chemical space concepts and visualization techniques  
- Machine learning applications in drug discovery
- Best practices in scientific programming

## Learning References

Key concepts and methodologies I've explored in this project:
- **Lipinski's Rule of Five** and drug-likeness criteria
- **Morgan (Circular) Fingerprints** for molecular similarity
- **Principal Component Analysis (PCA)** for dimensionality reduction
- **K-means Clustering** for chemical space partitioning
- **Structure-Activity Relationships (SAR)** analysis

If you use this tool in your research or educational work, please cite:

```
Chemical Space and Cluster-Specific Analysis of Small ChEMBL Datasets
A Student Project in Computational Chemistry and Cheminformatics
[GitHub Repository URL]
```

---

**Note**: This is an educational project developed as part of my learning journey in computational chemistry. While the methods are based on established scientific principles, always validate results with appropriate experimental methods and consult relevant literature for research applications. I'm continuously learning and improving this tool - feedback is always welcome! 
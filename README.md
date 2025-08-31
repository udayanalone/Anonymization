# Data Anonymization System using Conditional GAN (CGAN)

A comprehensive data anonymization system that protects patient privacy while maintaining data utility using advanced machine learning techniques.

## 🚀 Features

- **Smart Field Identification**: Automatically categorizes fields as identifiers, quasi-identifiers, or critical features
- **Advanced Pseudonymization**: Replaces direct identifiers with secure pseudonyms while maintaining referential integrity
- **Conditional GAN (CGAN)**: Generates synthetic quasi-identifiers conditioned on preserved critical features
- **Comprehensive Validation**: Privacy metrics (k-anonymity, l-diversity, t-closeness) and utility metrics (ML performance, statistical similarity)
- **Web Interface**: User-friendly Flask web application for easy data upload and result visualization
- **Modular Architecture**: Well-structured, extensible codebase with clear separation of concerns

## 🏗️ Architecture

The system implements a multi-stage anonymization pipeline:

1. **Data Loading & Analysis**: Loads patient data and analyzes structure
2. **Field Classification**: Identifies different types of fields for anonymization strategy
3. **Data Preprocessing**: Converts birthdates to age features, normalizes numerical values, encodes categorical variables
4. **Pseudonymization**: Replaces direct identifiers with secure pseudonyms
5. **CGAN Training**: Trains conditional GAN to generate synthetic quasi-identifiers
6. **Synthetic Generation**: Creates synthetic dataset using trained CGAN
7. **Validation**: Comprehensive privacy and utility assessment
8. **Web Interface**: Interactive visualization and download of results

## 📁 Project Structure

```
Anonymization/
├── data/                          # Data directories
│   ├── raw/                       # Original patient data
│   ├── processed/                 # Preprocessed data
│   └── synthetic/                 # Generated synthetic data
├── src/                           # Core implementation
│   ├── preprocessing.py           # Data preprocessing and field identification
│   ├── pseudonymization.py       # Identifier pseudonymization
│   ├── gan_model.py              # Conditional GAN implementation
│   ├── synthetic_generator.py    # Synthetic data generation
│   └── validation.py             # Privacy and utility validation
├── web/                          # Web interface
│   ├── app.py                    # Flask application
│   └── templates/                # HTML templates
├── models/                        # Trained CGAN models
├── results/                       # Validation results and reports
├── run_project.py                 # Main pipeline runner
├── test_system.py                 # System testing script
└── requirements.txt               # Python dependencies
```

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- PyTorch 1.12+
- CUDA-compatible GPU (optional, for faster training)

### Setup

1. **Clone the repository**:
```bash
git clone <repository-url>
cd Anonymization
   ```

2. **Install dependencies**:
   ```bash
pip install -r requirements.txt
```

3. **Verify installation**:
```bash
   python test_system.py
   ```

## 🚀 Usage

### Command Line Interface

#### Complete Pipeline
```bash
# Run the full anonymization pipeline
python run_project.py --input data/raw/patient.csv --n-records 2000 --epochs 100

# Skip CGAN training (use existing models)
python run_project.py --input data/raw/patient.csv --skip-training

# Validate existing results only
python run_project.py --validate-only
```

#### Individual Components
```python
from src.preprocessing import load_patient_data, identify_field_types
from src.pseudonymization import Pseudonymizer
from src.gan_model import ConditionalGAN
from src.validation import validate_anonymization

# Load and analyze data
df = load_patient_data('data/raw/patient.csv', n_records=2000)
field_types = identify_field_types(df)

# Pseudonymize identifiers
pseudonymizer = Pseudonymizer()
df_pseudo = pseudonymizer.pseudonymize_dataset(df)

# Train CGAN
cgan = ConditionalGAN(noise_dim=100)
cgan.build_models(condition_dim=5, target_dim=8)
cgan.train(condition_features, target_features, epochs=100)

# Validate results
results = validate_anonymization('data/raw/patient.csv', 'data/processed/patient_pseudonymized.csv')
```

### Web Interface

1. **Start the web server**:
   ```bash
   cd web
   python app.py
   ```

2. **Open browser** and navigate to `http://localhost:5000`

3. **Upload patient data** and run the anonymization pipeline

4. **View results** with interactive visualizations and download anonymized datasets

## 📊 Privacy Metrics

### K-Anonymity
- Ensures each record is indistinguishable from at least k-1 other records
- Tested for k = 2, 5, 10, 20
- Achieved when minimum group size ≥ k

### L-Diversity
- Ensures sensitive attributes have at least l diverse values in each equivalence class
- Tested for l = 2, 3, 5
- Achieved when 80%+ groups are l-diverse

### T-Closeness
- Ensures distribution of sensitive attributes in each group is close to overall distribution
- Tested for t = 0.1, 0.2, 0.3
- Achieved when 80%+ groups are t-close

### Re-identification Risk
- Based on uniqueness of quasi-identifier combinations
- Risk levels: LOW (< 50%), MEDIUM (50-80%), HIGH (> 80%)

## 📈 Utility Metrics

### Machine Learning Performance
- Trains ML models on both original and synthetic data
- Compares accuracy and AUC scores
- Utility preserved if accuracy difference < 10%

### Statistical Similarity
- Compares mean and standard deviation of numerical features
- Overall similarity score based on relative differences
- Higher scores indicate better utility preservation

## 🔧 Configuration

### CGAN Parameters
- **Noise dimension**: 100 (default)
- **Hidden layers**: [128, 256, 128] (default)
- **Training epochs**: 100 (default)
- **Batch size**: 32 (default)

### Field Classification Rules
- **Identifiers**: ID, SSN, DRIVERS, PASSPORT, FIRST, LAST, ADDRESS
- **Quasi-identifiers**: BIRTHDATE, ZIP, RACE, ETHNICITY, CITY, STATE, COUNTY
- **Critical features**: GENDER, HEALTHCARE_EXPENSES, HEALTHCARE_COVERAGE

## 📝 Output Files

### Processed Data
- `patient_processed.csv`: Preprocessed dataset with age features
- `condition_features.npy`: Critical features for CGAN conditioning
- `target_features.npy`: Quasi-identifiers for synthetic generation

### Pseudonymized Data
- `patient_pseudonymized.csv`: Dataset with pseudonymized identifiers
- `pseudonymization_mapping.json`: Mapping of original to pseudonymized values

### Validation Results
- `validation_results.json`: Complete validation metrics
- `validation_summary.txt`: Human-readable summary report

### Models
- `generator.pth`: Trained CGAN generator
- `discriminator.pth`: Trained CGAN discriminator
- `training_history.json`: Training loss history

## 🧪 Testing

Run the comprehensive test suite:
```bash
python test_system.py
```

Tests cover:
- Module imports
- Data loading
- Field identification
- Pseudonymization
- CGAN model creation
- Validation functionality

## 🌐 Web Interface Features

- **Drag & Drop Upload**: Easy CSV file upload
- **Real-time Progress**: Live pipeline status and logs
- **Interactive Visualizations**: Plotly charts for data comparison
- **Comprehensive Results**: Privacy and utility metrics display
- **Download Options**: Easy access to processed datasets

## 🔒 Security Features

- **Secure Pseudonymization**: UUID-based patient ID replacement
- **Hashed SSNs**: SHA-256 hashing for SSN pseudonymization
- **Address Masking**: Preserves city/state while masking street details
- **Mapping Storage**: Separate storage of pseudonymization mappings

## 📚 API Reference

### Preprocessing Module
- `load_patient_data(file_path, n_records)`: Load and limit patient data
- `identify_field_types(df)`: Categorize fields by type
- `prepare_data_for_cgan(df, field_types, output_dir)`: Complete preprocessing pipeline

### Pseudonymization Module
- `Pseudonymizer`: Main pseudonymization class
- `pseudonymize_dataset(df)`: Apply pseudonymization to entire dataset
- `pseudonymize_patient_data(input_file, output_file)`: Convenience function

### CGAN Module
- `ConditionalGAN`: Main CGAN implementation
- `build_models(condition_dim, target_dim)`: Create generator and discriminator
- `train(condition_features, target_features, epochs)`: Train the CGAN
- `generate_synthetic_data(condition_features, num_samples)`: Generate synthetic data

### Validation Module
- `PrivacyValidator`: Privacy metrics calculation
- `UtilityValidator`: Utility metrics calculation
- `validate_anonymization(original_file, anonymized_file)`: Complete validation pipeline

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Synthea for providing synthetic patient data
- PyTorch team for the deep learning framework
- Flask team for the web framework
- Plotly for interactive visualizations

## 📞 Support

For questions and support:
- Create an issue on GitHub
- Check the documentation
- Review the test examples

## 🔮 Future Enhancements

- **Differential Privacy**: Add mathematical privacy guarantees
- **Advanced CGANs**: Implement WGAN-GP, StyleGAN variants
- **Real-time Processing**: Stream processing for large datasets
- **Cloud Deployment**: AWS/Azure deployment options
- **API Endpoints**: RESTful API for integration
- **Multi-modal Data**: Support for images, text, and structured data
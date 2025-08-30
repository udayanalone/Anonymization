# Enhanced Data Anonymization Process

[GitHub Repository](https://github.com/<your-username>/<your-repo-name>) - Source code and documentation

This project implements a comprehensive data anonymization process that focuses on protecting personally identifiable information (PII) while maintaining data utility for analysis and machine learning tasks.

## Anonymization Process

The anonymization process follows these steps:

1. **Data Collection and Loading**: Import data from various sources and formats.

2. **Preprocessing and Cleaning**: Handle missing values, outliers, and normalize data.

3. **Attribute Identification**: Identify different types of attributes in the dataset.

4. **Sensitive Attribute Determination**: Automatically detect sensitive attributes such as:
   - Names
   - Social Security Numbers (SSNs)
   - Email addresses
   - Phone numbers
   - Physical addresses
   - Birthdates
   - Credit card numbers

5. **Synthetic Data Generation**: Replace sensitive attributes with realistic but fake data using specialized techniques for each attribute type.

6. **Final Dataset Generation**: Create the anonymized dataset with no PII while retaining data quality and utility.

## Anonymization Techniques

The project implements multiple anonymization techniques:

- **Enhanced Anonymization**: Replaces sensitive attributes with synthetic data while preserving statistical properties.
- **K-Anonymity**: Ensures each record is indistinguishable from at least k-1 other records.
- **L-Diversity**: Ensures sensitive attributes have at least l diverse values in each equivalence class.
- **T-Closeness**: Ensures the distribution of sensitive attributes in each equivalence class is close to their distribution in the overall dataset.
- **Differential Privacy**: Adds calibrated noise to numerical attributes to provide mathematical privacy guarantees.

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd Anonymization

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Running the Complete Anonymization Process

```bash
python -m src.anonymization_process --input_path "path/to/input.csv" --output_path "path/to/output.csv" --method "enhanced"
```

Available methods:
- `enhanced`: Uses the enhanced anonymization for sensitive attributes
- `k_anonymity`: Applies k-anonymity
- `l_diversity`: Applies l-diversity
- `t_closeness`: Applies t-closeness
- `differential_privacy`: Applies differential privacy

### Using Enhanced Anonymization Only

```bash
python -m src.enhanced_anonymization
```

## Project Structure

```
Anonymization/
├── data/
│   ├── processed/         # Processed and anonymized data
│   └── raw/               # Raw input data
├── src/
│   ├── anonymization.py           # Traditional anonymization techniques
│   ├── anonymization_process.py   # Complete anonymization pipeline
│   ├── cgan_model.py              # CGAN for synthetic data generation
│   ├── data_generation.py         # Basic synthetic data generation
│   ├── data_processing.py         # Data preprocessing utilities
│   ├── enhanced_anonymization.py  # Enhanced anonymization for PII
│   ├── evaluate_synthetic_data.py # Evaluation metrics
│   ├── train_cgan.py              # CGAN training script
│   └── validation.py              # Validation utilities
├── tests/                 # Unit tests
├── requirements.txt       # Project dependencies
└── README.md             # Project documentation
```

## Evaluation Metrics

The anonymized data is evaluated based on:

1. **Privacy Protection**: Measures how well PII is protected
2. **Data Utility**: Measures how well the anonymized data preserves statistical properties
3. **Synthetic Data Quality**: Measures how realistic the synthetic data is

## License

This project is licensed under the MIT License - see the LICENSE file for details.
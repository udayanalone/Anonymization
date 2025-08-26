# Data Anonymization and Privacy Project

This project demonstrates various data anonymization techniques for protecting privacy in healthcare datasets while maintaining data utility for analysis and machine learning tasks.

## Project Overview

The project implements and evaluates several privacy-preserving techniques:

- **k-anonymity**: Ensures that each record is indistinguishable from at least k-1 other records with respect to certain attributes
- **l-diversity**: Extends k-anonymity by ensuring diversity in sensitive attributes
- **t-closeness**: Further enhances privacy by controlling the distribution of sensitive values
- **Differential Privacy**: Adds calibrated noise to data to provide mathematical privacy guarantees

## Directory Structure

```
Data-Anonymization-and-Privacy-Synthetic/
├── data/
│   ├── raw/
│   │   └── synthetic_data.csv        # Raw synthetic data generated using NumPy
│   ├── processed/
│   │   └── anonymized_data.csv       # Anonymized dataset after applying techniques
│   └── README.md                     # Data description and preprocessing steps
├── src/
│   ├── __init__.py
│   ├── data_generation.py            # Script to generate synthetic data using NumPy
│   ├── data_processing.py            # Data cleaning and preprocessing functions
│   ├── anonymization.py              # Functions implementing anonymization techniques
│   └── validation.py                 # Functions to validate anonymization effectiveness
├── tests/
│   ├── __init__.py
│   ├── test_data_generation.py       # Unit tests for data generation
│   ├── test_data_processing.py       # Unit tests for data processing
│   ├── test_anonymization.py         # Unit tests for anonymization methods
│   └── test_validation.py            # Unit tests for validation functions
├── requirements.txt                  # List of required Python packages
├── README.md                         # Project overview and instructions
└── .gitignore                        # Git ignore file to exclude unnecessary files
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Data-Anonymization-and-Privacy-Synthetic.git
   cd Data-Anonymization-and-Privacy-Synthetic
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Generating Synthetic Data

```bash
python -m src.data_generation
```

This will create a synthetic dataset in the `data/raw/` directory.

### Processing Data

```bash
python -m src.data_processing
```

This will clean and preprocess the raw data, saving the result in the `data/processed/` directory.

### Applying Anonymization

```bash
python -m src.anonymization
```

This will apply k-anonymity to the processed data by default. You can modify the script to use other anonymization techniques.

### Validating Anonymization

```bash
python -m src.validation
```

This will evaluate the effectiveness of the anonymization techniques and report metrics on privacy protection and utility preservation.

## Running Tests

To run all tests:

```bash
python -m pytest tests/
```

To run specific test files:

```bash
python -m pytest tests/test_anonymization.py
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
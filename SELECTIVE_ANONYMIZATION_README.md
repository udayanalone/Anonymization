# Selective Anonymization

## Overview

The Selective Anonymization module provides a targeted approach to data anonymization that preserves the utility of research-relevant data while anonymizing only the identifying attributes. This approach ensures that:

1. **Research data integrity is maintained** - Fields that contribute to research value are left untouched
2. **Privacy is protected** - Identifying attributes are replaced with synthetic data
3. **Data utility is maximized** - The balance between privacy and utility is optimized

## Key Features

- **Intelligent identifier detection** - Automatically identifies columns that could be used to identify individuals
- **Research data preservation** - Keeps research-relevant columns intact
- **Format-preserving anonymization** - Maintains the format and distribution of identifier columns
- **CGAN integration** - Uses Conditional Generative Adversarial Networks when available
- **Fallback mechanisms** - Gracefully falls back to traditional methods when needed

## Usage

### Basic Usage

```python
from selective_anonymization import selectively_anonymize_dataset
import pandas as pd

# Load your dataset
df = pd.read_csv('your_data.csv')

# Specify columns that are important for research
research_columns = ['Age', 'Gender', 'Glucose', 'BloodPressure', 'BMI', 'Outcome']

# Anonymize only identifier columns
anonymized_df, anonymized_columns = selectively_anonymize_dataset(df, research_columns)

# Save the anonymized dataset
anonymized_df.to_csv('selectively_anonymized_data.csv', index=False)
```

### Command Line Usage

```bash
python src/selective_anonymization.py --input data/raw/your_data.csv --output data/processed/selectively_anonymized_data.csv --research-columns Age Gender Glucose BloodPressure BMI Outcome
```

Options:
- `--input`: Path to the input CSV file
- `--output`: Path to save the output CSV file
- `--research-columns`: List of columns that are important for research and should be preserved

## How It Works

### 1. Identifier Column Detection

The module identifies columns that could be used as identifiers through multiple methods:

- **Pattern matching** - Detects common identifier column names (e.g., PatientID, Name, Email)
- **Sensitivity detection** - Uses the enhanced_anonymization module to identify sensitive attributes
- **Cardinality analysis** - Identifies columns with high uniqueness that aren't research-relevant
- **Exclusion list** - Excludes columns explicitly marked as research-relevant

### 2. Synthetic Data Generation

For each identified identifier column, synthetic data is generated using:

- **CGAN models** - When available, for high-quality synthetic data that preserves distributions
- **Faker library** - For traditional synthetic data generation as a fallback
- **Format preservation** - Ensures ID formats (numeric ranges, prefixes) are maintained

### 3. Selective Replacement

Only the identified identifier columns are replaced with synthetic data, while research-relevant columns remain untouched.

## Benefits Over Traditional Anonymization

1. **Preserves data utility** - Research-relevant data remains unchanged, maintaining statistical validity
2. **Reduces information loss** - Only modifies data that could compromise privacy
3. **Maintains relationships** - Preserves relationships between non-identifier columns
4. **Optimizes privacy-utility tradeoff** - Provides maximum privacy protection with minimal impact on utility

## Limitations

1. **Requires domain knowledge** - Optimal results require specifying which columns are research-relevant
2. **Potential for re-identification** - If too many quasi-identifiers are preserved, re-identification risk may increase
3. **Computational requirements** - CGAN-based anonymization requires more computational resources

## Future Improvements

1. **Automated research relevance detection** - Machine learning to automatically identify research-relevant columns
2. **Privacy risk quantification** - Tools to measure re-identification risk in the anonymized dataset
3. **Advanced synthetic data validation** - Methods to ensure synthetic identifiers maintain realistic properties
4. **Differential privacy integration** - Add differential privacy guarantees to the anonymization process
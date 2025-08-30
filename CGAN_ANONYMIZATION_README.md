# CGAN-Based Data Anonymization with Dynamic Conditions

This document provides implementation guidance for using Conditional Generative Adversarial Networks (CGANs) for data anonymization with dynamic condition selection based on sensitive attributes.

## Overview

The implementation replaces the traditional Faker library with a CGAN-based approach that can generate more realistic synthetic data while preserving statistical properties and relationships between attributes. The key components of this approach are:

1. **CGAN Model Implementation** - A neural network architecture that generates synthetic data conditioned on specific attributes
2. **Dynamic Condition Selection** - Automatic identification of relevant condition columns for each sensitive attribute
3. **Enhanced Anonymization Integration** - Modifications to the existing anonymization process to use CGAN models

## Implementation Components

### 1. CGAN Model (`cgan_model.py`)

The CGAN model implementation provides:

- A generator network that creates synthetic data
- A discriminator network that distinguishes between real and synthetic data
- Specialized preprocessing for different attribute types (names, emails, SSNs, etc.)
- Training functionality with condition embedding
- Post-processing to ensure valid format for each attribute type

### 2. Dynamic Condition Selection (`dynamic_anonymization.py`)

The dynamic condition selection process:

- Identifies potential condition columns based on cardinality (number of unique values)
- Analyzes correlations between sensitive attributes and potential conditions
- Selects the most relevant conditions for each sensitive attribute
- Creates combined conditions when multiple relevant columns are found

### 3. Enhanced Anonymization Integration

The enhanced anonymization process has been modified to:

- Use CGAN models when available, with fallback to Faker
- Load or train CGAN models for each sensitive attribute
- Dynamically determine suitable condition columns
- Generate synthetic data using the trained models and appropriate conditions

## Usage Guide

### Basic Usage

```python
# Import the necessary modules
from enhanced_anonymization import anonymize_dataset, identify_sensitive_attributes
from dynamic_anonymization import train_dynamic_cgan_models, generate_dynamic_synthetic_data

# Load your dataset
df = pd.read_csv('your_data.csv')

# Identify sensitive attributes
sensitive_attributes = identify_sensitive_attributes(df)

# Train CGAN models with dynamic conditions
models, used_conditions = train_dynamic_cgan_models(df, sensitive_attributes)

# Generate synthetic data
synthetic_df = generate_dynamic_synthetic_data(df, models, used_conditions)

# Save the anonymized data
synthetic_df.to_csv('anonymized_data.csv', index=False)
```

### Command Line Usage

You can also use the provided script directly from the command line:

```bash
python src/dynamic_anonymization.py --input data/raw/your_data.csv --output data/processed/anonymized_data.csv --visualize
```

Options:
- `--input`: Path to the input CSV file
- `--output`: Path to save the output CSV file
- `--visualize`: Flag to generate visualizations comparing original and synthetic data distributions

## Implementation Details

### CGAN Architecture

The CGAN model consists of:

1. **Generator Network**:
   - Input: Random noise vector + condition embedding
   - Hidden layers: Dense layers with LeakyReLU activation
   - Output: Synthetic data with appropriate dimensions for the attribute type

2. **Discriminator Network**:
   - Input: Data sample + condition embedding
   - Hidden layers: Dense layers with LeakyReLU activation and dropout
   - Output: Probability that the input is real (vs. synthetic)

3. **Training Process**:
   - Alternating training of discriminator and generator
   - Wasserstein loss with gradient penalty for improved stability
   - Condition embedding to guide the generation process

### Dynamic Condition Selection Process

1. **Identify Potential Conditions**:
   - Look for categorical columns with few unique values (≤ 10)
   - Exclude other sensitive attributes from consideration

2. **Analyze Correlations**:
   - For categorical sensitive attributes: Use contingency table analysis
   - For numerical sensitive attributes: Compare mean differences between categories

3. **Select Best Conditions**:
   - Rank conditions by correlation strength
   - Select top N conditions (default: 3)

4. **Create Combined Conditions**:
   - Concatenate selected condition values with underscores
   - Use as input to the CGAN model

### Data Preprocessing

Different preprocessing techniques are applied based on attribute type:

- **Names**: Tokenization at character level
- **Emails**: Split into username and domain parts
- **SSNs/Phone Numbers**: Digit extraction and normalization
- **Addresses**: Tokenization with special handling for numbers and separators
- **Dates**: Conversion to numerical representation (year, month, day)
- **Credit Card Numbers**: Digit extraction and normalization

## Advantages Over Traditional Methods

1. **Preserves Statistical Relationships**: The CGAN approach maintains relationships between sensitive attributes and other variables
2. **Realistic Synthetic Data**: Generated data follows the distribution of the original data while being completely synthetic
3. **Dynamic Conditioning**: Automatically identifies and uses relevant conditions for each sensitive attribute
4. **Customizable Privacy-Utility Tradeoff**: Can be tuned to balance privacy protection and data utility

## Limitations and Considerations

1. **Computational Requirements**: Training CGAN models requires more computational resources than using Faker
2. **Training Data Size**: Effective training requires sufficient examples for each condition combination
3. **Rare Categories**: May struggle with very rare condition combinations
4. **Model Persistence**: Models need to be saved and loaded for reuse

## Future Improvements

1. **Differential Privacy Integration**: Add differential privacy guarantees to the training process
2. **Advanced Architectures**: Experiment with transformer-based architectures for sequence data
3. **Federated Learning**: Enable training across multiple data sources without sharing raw data
4. **Evaluation Metrics**: Implement comprehensive privacy and utility metrics

## References

1. Goodfellow, I., et al. (2014). Generative Adversarial Nets
2. Mirza, M., & Osindero, S. (2014). Conditional Generative Adversarial Nets
3. Arjovsky, M., et al. (2017). Wasserstein GAN
4. Xu, L., et al. (2019). Modeling Tabular Data using Conditional GAN
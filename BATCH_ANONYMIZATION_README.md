# Batch Anonymization Script

This script processes all CSV files in a specified input directory and converts them into anonymized data. The processed files are saved in an output directory with the prefix "anonymized_" added to each filename.

## Features

- Processes all CSV files in the input directory
- Identifies sensitive attributes in each file using pattern matching
- Anonymizes identified sensitive attributes using synthetic data generation
- Preserves data structure and non-sensitive attributes
- Saves anonymized files with a configurable prefix

## Requirements

The script requires the following Python packages:

- pandas
- numpy
- scikit-learn
- faker

You can install these packages using pip:

```bash
pip install pandas numpy scikit-learn faker
```

## Usage

### Default Usage

Simply run the script with Python:

```bash
python batch_anonymize.py
```

By default, the script will:
- Process all CSV files in `D:/Projects/Final_Year/Anonymization/data/raw`
- Save anonymized files in `D:/Projects/Final_Year/Anonymization/data/processed` with the prefix "anonymized_"

### Customizing Input/Output Directories

To use different directories, modify the following lines in the `main()` function:

```python
# Define input and output directories
input_dir = "D:/Projects/Final_Year/Anonymization/data/raw"
output_dir = "D:/Projects/Final_Year/Anonymization/data/processed"
```

### Customizing the Prefix

To change the prefix added to anonymized files, modify the `batch_anonymize()` function call in `main()`:

```python
# Run batch anonymization with custom prefix
batch_anonymize(input_dir, output_dir, prefix="custom_prefix_")
```

## How It Works

1. The script scans the input directory for CSV files
2. For each file:
   - Loads the data into a pandas DataFrame
   - Identifies sensitive attributes using pattern matching
   - Anonymizes identified sensitive attributes using synthetic data generation
   - Saves the anonymized data to the output directory with the specified prefix
3. Reports the number of successfully processed and failed files

## Sensitive Attribute Detection

The script can automatically detect the following types of sensitive attributes:

- Names (e.g., "John Smith")
- Social Security Numbers (e.g., "123-45-6789")
- Email addresses (e.g., "example@domain.com")
- Phone numbers (e.g., "(123) 456-7890" or "123-456-7890")
- Addresses (containing keywords like "street", "avenue", etc.)
- Birth dates (in various formats)
- Credit card numbers (e.g., "1234-5678-9012-3456")

## Limitations

- The script only processes CSV files
- Sensitive attribute detection relies on pattern matching and may not identify all sensitive data
- The anonymization process replaces sensitive data with synthetic data, which may not preserve all statistical properties of the original data
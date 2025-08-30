import os
import pandas as pd
import numpy as np
import argparse
from pathlib import Path

# Import the enhanced anonymization module
from enhanced_anonymization import identify_sensitive_attributes

# Import CGAN model functions if available
try:
    from cgan_model import CGAN, train_cgan_model, generate_synthetic_data_with_cgan
    has_cgan = True
except ImportError:
    has_cgan = False
    from faker import Faker
    fake = Faker()


def identify_identifier_columns(df, research_columns=None):
    """
    Identify columns that could be used as identifiers but don't contribute to research.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame to analyze
    research_columns : list, optional
        List of columns that are important for research and should be preserved
        
    Returns:
    --------
    dict
        Dictionary mapping column names to attribute types
    """
    # If research_columns is not provided, use all columns except obvious identifiers
    if research_columns is None:
        # Default columns that are typically not needed for research but can identify individuals
        potential_identifiers = [
            'PatientID', 'ID', 'MemberID', 'CustomerID', 'EmployeeID',
            'FullName', 'Name', 'FirstName', 'LastName',
            'Email', 'EmailAddress',
            'SSN', 'SocialSecurityNumber',
            'Phone', 'PhoneNumber', 'ContactNumber',
            'Address', 'StreetAddress', 'City', 'State', 'ZipCode', 'PostalCode',
            'BirthDate', 'DateOfBirth', 'DOB',
            'CreditCard', 'CreditCardNumber', 'CCNumber'
        ]
        research_columns = [col for col in df.columns if col not in potential_identifiers]
    
    # Identify sensitive attributes using the enhanced_anonymization module
    all_sensitive_attributes = identify_sensitive_attributes(df)
    
    # Add columns that match common identifier patterns
    identifier_columns = {}
    
    for column in df.columns:
        # Skip columns that are important for research
        if column in research_columns:
            continue
        
        # Check if it's already identified as sensitive
        if column in all_sensitive_attributes:
            identifier_columns[column] = all_sensitive_attributes[column]
            continue
        
        # Check for ID-like columns (unique values that are not research-relevant)
        if 'id' in column.lower() or column.lower().endswith('_id') or column.lower().startswith('id_'):
            if df[column].nunique() > len(df) * 0.5:  # High cardinality
                identifier_columns[column] = 'id'
                continue
        
        # Check for columns with unique values (potential identifiers)
        if df[column].nunique() == len(df) and len(df) > 10:
            identifier_columns[column] = 'unique_identifier'
            continue
    
    return identifier_columns


def generate_synthetic_data_for_column(column_name, column_data, attribute_type):
    """
    Generate synthetic data for a column based on its attribute type.
    
    Parameters:
    -----------
    column_name : str
        Name of the column
    column_data : pandas.Series
        Original column data
    attribute_type : str
        Type of the attribute (e.g., 'name', 'email', 'id')
        
    Returns:
    --------
    pandas.Series
        Synthetic data for the column
    """
    n_samples = len(column_data)
    
    # If CGAN is available, try to use it
    if has_cgan:
        try:
            # Create a simple condition (we're not using dynamic conditions here)
            conditions = ['default'] * n_samples
            
            # Train a simple CGAN model
            model = train_cgan_model(column_data, conditions, attribute_type, epochs=100)
            
            # Generate synthetic data
            synthetic_data = generate_synthetic_data_with_cgan(model, conditions)
            return pd.Series(synthetic_data)
        except Exception as e:
            print(f"Error using CGAN for {column_name}: {e}")
            print("Falling back to traditional methods...")
    
    # Fall back to Faker or simple methods
    if attribute_type == 'name':
        return pd.Series([fake.name() for _ in range(n_samples)])
    elif attribute_type == 'email':
        return pd.Series([fake.email() for _ in range(n_samples)])
    elif attribute_type == 'ssn':
        return pd.Series([fake.ssn() for _ in range(n_samples)])
    elif attribute_type == 'phone':
        return pd.Series([fake.phone_number() for _ in range(n_samples)])
    elif attribute_type == 'address':
        return pd.Series([fake.address().replace('\n', ', ') for _ in range(n_samples)])
    elif attribute_type == 'birthdate':
        return pd.Series([fake.date_of_birth(minimum_age=18, maximum_age=90).strftime('%Y-%m-%d') for _ in range(n_samples)])
    elif attribute_type == 'creditcard':
        return pd.Series([fake.credit_card_number() for _ in range(n_samples)])
    elif attribute_type == 'id' or attribute_type == 'unique_identifier':
        # For IDs, generate new unique identifiers that preserve the format
        if column_data.dtype == 'int64' or column_data.dtype == 'int32':
            # For numeric IDs, generate random numbers in a similar range
            min_val = column_data.min()
            max_val = column_data.max()
            return pd.Series(np.random.randint(min_val, max_val + 1000000, size=n_samples))
        else:
            # For string IDs, try to preserve the format (length and character types)
            sample = str(column_data.iloc[0])
            prefix = ''.join([c for c in sample if not c.isdigit()])
            if prefix:
                # If there's a prefix, preserve it and randomize the rest
                digit_count = sum(1 for c in sample if c.isdigit())
                return pd.Series([prefix + ''.join([str(np.random.randint(0, 10)) for _ in range(digit_count)]) for _ in range(n_samples)])
            else:
                # Otherwise, just generate random strings of the same length
                return pd.Series([fake.bothify('?' * len(sample)) for _ in range(n_samples)])
    else:
        # For unknown types, generate placeholder values
        return pd.Series([f"ANONYMIZED_{column_name}_{i}" for i in range(n_samples)])


def selectively_anonymize_dataset(df, research_columns=None):
    """
    Anonymize only identifier columns while preserving research-relevant data.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame to anonymize
    research_columns : list, optional
        List of columns that are important for research and should be preserved
        
    Returns:
    --------
    pandas.DataFrame
        Anonymized DataFrame
    dict
        Dictionary of anonymized columns and their types
    """
    # Identify columns that could be used as identifiers
    identifier_columns = identify_identifier_columns(df, research_columns)
    
    if not identifier_columns:
        print("No identifier columns found that need anonymization.")
        return df.copy(), {}
    
    print(f"Found {len(identifier_columns)} columns to anonymize: {list(identifier_columns.keys())}")
    
    # Create a copy of the DataFrame to avoid modifying the original
    anonymized_df = df.copy()
    
    # Anonymize each identifier column
    for column, attr_type in identifier_columns.items():
        print(f"Anonymizing {column} (type: {attr_type})...")
        anonymized_df[column] = generate_synthetic_data_for_column(column, df[column], attr_type)
    
    return anonymized_df, identifier_columns


def main():
    """
    Main function to demonstrate selective anonymization.
    """
    parser = argparse.ArgumentParser(description='Selective anonymization of identifier columns')
    parser.add_argument('--input', type=str, default=os.path.join('..', 'data', 'processed', 'demo_with_pii.csv'),
                        help='Path to input CSV file')
    parser.add_argument('--output', type=str, default=os.path.join('..', 'data', 'processed', 'selectively_anonymized_data.csv'),
                        help='Path to output CSV file')
    parser.add_argument('--research-columns', type=str, nargs='+',
                        help='List of columns that are important for research and should be preserved')
    args = parser.parse_args()
    
    # Check if CGAN is available
    if not has_cgan:
        print("Warning: CGAN module not available. Will use traditional anonymization methods.")
    
    # Load data
    print(f"Loading data from {args.input}...")
    try:
        df = pd.read_csv(args.input)
        print(f"Loaded {len(df)} rows with {len(df.columns)} columns")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Anonymize the dataset
    anonymized_df, anonymized_columns = selectively_anonymize_dataset(df, args.research_columns)
    
    # Save the anonymized dataset
    output_path = args.output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    anonymized_df.to_csv(output_path, index=False)
    print(f"Saved anonymized data to {output_path}")
    
    # Print summary
    print("\nAnonymization Summary:")
    print(f"Total columns: {len(df.columns)}")
    print(f"Anonymized columns: {len(anonymized_columns)}")
    print(f"Preserved columns: {len(df.columns) - len(anonymized_columns)}")
    
    if anonymized_columns:
        print("\nAnonymized columns and their types:")
        for column, attr_type in anonymized_columns.items():
            print(f"  - {column}: {attr_type}")


if __name__ == "__main__":
    main()
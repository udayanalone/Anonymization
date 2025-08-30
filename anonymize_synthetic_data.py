#!/usr/bin/env python3
"""
Script to anonymize synthetic_data.csv using the enhanced anonymization module.
This script adds synthetic PII attributes to the medical data and then anonymizes them.
"""

import pandas as pd
import numpy as np
import os
import sys
from faker import Faker
import random

# Add the src directory to the path so we can import the modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from enhanced_anonymization import identify_sensitive_attributes, anonymize_dataset
    print("Successfully imported enhanced anonymization module")
except ImportError as e:
    print(f"Error importing enhanced anonymization module: {e}")
    print("Please ensure the src directory contains the required modules")
    sys.exit(1)

def add_synthetic_pii(df):
    """
    Add synthetic PII attributes to the medical dataset for demonstration purposes.
    In a real scenario, these would be actual sensitive data that needs anonymization.
    """
    print("Adding synthetic PII attributes for demonstration...")
    
    fake = Faker()
    
    # Add synthetic sensitive attributes
    df['PatientName'] = [fake.name() for _ in range(len(df))]
    df['Email'] = [fake.email() for _ in range(len(df))]
    df['SSN'] = [f"{random.randint(100, 999)}-{random.randint(10, 99)}-{random.randint(1000, 9999)}" for _ in range(len(df))]
    df['Phone'] = [fake.phone_number() for _ in range(len(df))]
    df['Address'] = [fake.address().replace('\n', ', ') for _ in range(len(df))]
    df['BirthDate'] = [fake.date_of_birth(minimum_age=18, maximum_age=90).strftime('%Y-%m-%d') for _ in range(len(df))]
    df['CreditCard'] = [fake.credit_card_number(card_type=None) for _ in range(len(df))]
    
    print("Added synthetic PII attributes:")
    print("- PatientName: Full names")
    print("- Email: Email addresses")
    print("- SSN: Social Security Numbers")
    print("- Phone: Phone numbers")
    print("- Address: Physical addresses")
    print("- BirthDate: Birth dates")
    print("- CreditCard: Credit card numbers")
    
    return df

def main():
    """Main function to anonymize the synthetic data."""
    
    # File paths
    input_path = os.path.join('data', 'raw', 'synthetic_data.csv')
    output_path = os.path.join('data', 'processed', 'anonymized_data.csv')
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print("Starting anonymization process...")
    print(f"Input file: {input_path}")
    print(f"Output file: {output_path}")
    
    # Step 1: Load the synthetic data
    print("\nStep 1: Loading synthetic data...")
    try:
        df = pd.read_csv(input_path)
        print(f"Successfully loaded data from {input_path}")
        print(f"Dataset shape: {df.shape}")
        print(f"Original columns: {', '.join(df.columns)}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Step 2: Add synthetic PII attributes
    print("\nStep 2: Adding synthetic PII attributes...")
    df_with_pii = add_synthetic_pii(df)
    print(f"Dataset shape after adding PII: {df_with_pii.shape}")
    print(f"Columns with PII: {', '.join(df_with_pii.columns)}")
    
    # Step 3: Identify sensitive attributes
    print("\nStep 3: Identifying sensitive attributes...")
    try:
        sensitive_attrs = identify_sensitive_attributes(df_with_pii)
        print(f"Identified sensitive attributes: {sensitive_attrs}")
        
        if not sensitive_attrs:
            print("No sensitive attributes were automatically identified.")
            print("This might happen if the data format doesn't match expected patterns.")
            return
    except Exception as e:
        print(f"Error during sensitive attribute identification: {e}")
        return
    
    # Step 4: Apply anonymization
    print("\nStep 4: Applying enhanced anonymization...")
    try:
        df_anonymized = anonymize_dataset(df_with_pii, sensitive_attrs)
        print("Enhanced anonymization applied successfully")
        
        # Show sample of anonymized data
        print("\nSample of anonymized data:")
        print(df_anonymized[list(sensitive_attrs.keys())].head())
        
    except Exception as e:
        print(f"Error during anonymization: {e}")
        return
    
    # Step 5: Save anonymized data
    print("\nStep 5: Saving anonymized data...")
    try:
        df_anonymized.to_csv(output_path, index=False)
        print(f"Anonymized data saved successfully to {output_path}")
        
        # Show summary statistics
        print(f"\nFinal dataset shape: {df_anonymized.shape}")
        print(f"Total columns: {len(df_anonymized.columns)}")
        print(f"Anonymized columns: {len(sensitive_attrs)}")
        
        # Verify that sensitive data has been changed
        print("\nVerification - Sample of anonymized sensitive data:")
        for col in list(sensitive_attrs.keys())[:3]:  # Show first 3 sensitive columns
            if col in df_anonymized.columns:
                print(f"{col}: {df_anonymized[col].iloc[0]} (anonymized)")
        
    except Exception as e:
        print(f"Error saving anonymized data: {e}")
        return
    
    print("\nAnonymization process completed successfully!")
    print(f"Original data: {input_path}")
    print(f"Anonymized data: {output_path}")

if __name__ == "__main__":
    main()

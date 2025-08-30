import os
import pandas as pd
import numpy as np
from faker import Faker
import random
from src.anonymization_process import run_anonymization_process

# Initialize Faker for generating synthetic data
fake = Faker()


def create_sample_dataset(n_records=100, output_path=None):
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate sample data
    data = {
        'PatientID': range(1, n_records+1),
        'Age': np.random.randint(18, 90, size=n_records),
        'Gender': np.random.choice(['Male', 'Female'], size=n_records),
        'FullName': [fake.name() for _ in range(n_records)],
        'Email': [fake.email() for _ in range(n_records)],
        'SSN': [f"{random.randint(100, 999)}-{random.randint(10, 99)}-{random.randint(1000, 9999)}" for _ in range(n_records)],
        'Phone': [fake.phone_number() for _ in range(n_records)],
        'Address': [fake.address().replace('\n', ', ') for _ in range(n_records)],
        'BirthDate': [fake.date_of_birth(minimum_age=18, maximum_age=90).strftime('%Y-%m-%d') for _ in range(n_records)],
        'CreditCard': [fake.credit_card_number(card_type=None) for _ in range(n_records)],
        'BloodPressure': np.round(np.random.uniform(90, 180, size=n_records), 1),
        'Cholesterol': np.round(np.random.uniform(120, 300, size=n_records), 1),
        'Glucose': np.round(np.random.uniform(70, 200, size=n_records), 1),
        'Outcome': np.random.choice([0, 1], size=n_records)
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV if output_path is provided
    if output_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Sample dataset created successfully and saved to {output_path}!")
    
    return df


def main():
    """
    Main function to demonstrate the anonymization process.
    """
    # Paths for sample data and anonymized output
    sample_data_path = os.path.join('data', 'raw', 'sample_data.csv')
    anonymized_data_path = os.path.join('data', 'processed', 'demo_anonymized_data.csv')
    
    # Create sample dataset
    print("Creating sample dataset with sensitive attributes...")
    create_sample_dataset(n_records=100, output_path=sample_data_path)
    
    # Run the anonymization process
    print("\nRunning the anonymization process...")
    run_anonymization_process(
        input_path=sample_data_path,
        output_path=anonymized_data_path,
        anonymization_method='enhanced'
    )
    
    # Compare original and anonymized data
    print("\nComparing original and anonymized data:")
    original_df = pd.read_csv(sample_data_path)
    anonymized_df = pd.read_csv(anonymized_data_path)
    
    # Display sample rows from both datasets
    print("\nOriginal data sample:")
    print(original_df.head(3).to_string())
    
    print("\nAnonymized data sample:")
    print(anonymized_df.head(3).to_string())
    
    # Check if sensitive attributes have been anonymized
    sensitive_attrs = ['FullName', 'Email', 'SSN', 'Phone', 'Address', 'BirthDate', 'CreditCard']
    print("\nVerifying anonymization of sensitive attributes:")
    
    for attr in sensitive_attrs:
        if attr in original_df.columns and attr in anonymized_df.columns:
            # Check if values are different
            common_values = set(original_df[attr]).intersection(set(anonymized_df[attr]))
            anonymization_rate = 1 - (len(common_values) / len(original_df))
            print(f"{attr}: {anonymization_rate*100:.2f}% anonymized")


if __name__ == "__main__":
    main()
import pandas as pd
import numpy as np
import os
import sys
import argparse

# Add the src directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_processing import load_data, clean_data, preprocess_data, save_processed_data
from src.enhanced_anonymization import identify_sensitive_attributes, anonymize_dataset
from src.anonymization import anonymize_data


def run_anonymization_process(input_path, output_path, anonymization_method='enhanced'):
    print("Starting anonymization process...")
    
    # Step 1: Data collection and loading
    print("\nStep 1: Data collection and loading")
    try:
        df = load_data(input_path)
        print(f"Successfully loaded data from {input_path}")
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {', '.join(df.columns)}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Step 2: Preprocessing and cleaning
    print("\nStep 2: Preprocessing and cleaning")
    try:
        df_clean = clean_data(df)
        df_processed = preprocess_data(df_clean)
        print("Data preprocessing and cleaning completed successfully")
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        return
    
    # Step 3: Attribute identification
    print("\nStep 3: Attribute identification")
    try:
        # Identify column types
        numerical_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df_processed.select_dtypes(include=['object']).columns.tolist()
        print(f"Numerical columns: {', '.join(numerical_cols)}")
        print(f"Categorical columns: {', '.join(categorical_cols) if categorical_cols else 'None'}")
    except Exception as e:
        print(f"Error during attribute identification: {e}")
        return
    
    # Step 4: Sensitive attribute determination
    print("\nStep 4: Sensitive attribute determination")
    try:
        sensitive_attrs = identify_sensitive_attributes(df_processed)
        print(f"Identified sensitive attributes: {sensitive_attrs}")
        
        # If no sensitive attributes were automatically identified, we can manually specify some
        if not sensitive_attrs and anonymization_method == 'enhanced':
            print("No sensitive attributes automatically identified. Adding synthetic sensitive data for demonstration.")
            # This is just for demonstration - in a real scenario, you'd work with actual sensitive data
            from faker import Faker
            import random
            
            fake = Faker()
            df_processed['FullName'] = [fake.name() for _ in range(len(df_processed))]
            df_processed['Email'] = [fake.email() for _ in range(len(df_processed))]
            df_processed['SSN'] = [f"{random.randint(100, 999)}-{random.randint(10, 99)}-{random.randint(1000, 9999)}" for _ in range(len(df_processed))]
            df_processed['Phone'] = [fake.phone_number() for _ in range(len(df_processed))]
            df_processed['Address'] = [fake.address().replace('\n', ', ') for _ in range(len(df_processed))]
            df_processed['BirthDate'] = [fake.date_of_birth(minimum_age=18, maximum_age=90).strftime('%Y-%m-%d') for _ in range(len(df_processed))]
            df_processed['CreditCard'] = [fake.credit_card_number(card_type=None) for _ in range(len(df_processed))]
            
            # Re-identify sensitive attributes
            sensitive_attrs = identify_sensitive_attributes(df_processed)
            print(f"Added synthetic sensitive attributes: {sensitive_attrs}")
    except Exception as e:
        print(f"Error during sensitive attribute determination: {e}")
        return
    
    # Step 5 & 6: Apply anonymization and generate final dataset
    print("\nStep 5 & 6: Applying anonymization and generating final dataset")
    try:
        if anonymization_method == 'enhanced':
            # Use enhanced anonymization for sensitive attributes
            df_anonymized = anonymize_dataset(df_processed, sensitive_attrs)
            print("Enhanced anonymization applied to sensitive attributes")
        else:
            # Use traditional anonymization methods
            if anonymization_method == 'k_anonymity':
                # Define quasi-identifiers (columns that could potentially identify individuals)
                quasi_identifiers = [col for col in numerical_cols if col not in ['PatientID', 'Outcome']]
                df_anonymized = anonymize_data(df_processed, method='k_anonymity', 
                                             sensitive_columns=quasi_identifiers, k=5)
                print("K-anonymity applied to quasi-identifiers")
            elif anonymization_method == 'l_diversity':
                quasi_identifiers = [col for col in numerical_cols if col not in ['PatientID', 'Outcome']]
                df_anonymized = anonymize_data(df_processed, method='l_diversity', 
                                             sensitive_columns=quasi_identifiers, 
                                             sensitive_attribute='Outcome', l=3)
                print("L-diversity applied to quasi-identifiers")
            elif anonymization_method == 't_closeness':
                quasi_identifiers = [col for col in numerical_cols if col not in ['PatientID', 'Outcome']]
                df_anonymized = anonymize_data(df_processed, method='t_closeness', 
                                             sensitive_columns=quasi_identifiers, 
                                             sensitive_attribute='Outcome', t=0.2)
                print("T-closeness applied to quasi-identifiers")
            elif anonymization_method == 'differential_privacy':
                df_anonymized = anonymize_data(df_processed, method='differential_privacy', epsilon=1.0)
                print("Differential privacy applied to numerical attributes")
            else:
                print(f"Unknown anonymization method: {anonymization_method}")
                return
        
        # Save the anonymized dataset
        save_processed_data(df_anonymized, output_path)
        print(f"\nAnonymization process completed successfully!")
        print(f"Anonymized data saved to {output_path}")
        
        # Print some statistics to verify data quality and utility
        print("\nData quality and utility verification:")
        print(f"Original shape: {df.shape}, Anonymized shape: {df_anonymized.shape}")
        
        # Check if numerical columns exist in both dataframes
        common_num_cols = [col for col in numerical_cols if col in df.columns and col in df_anonymized.columns]
        if common_num_cols:
            # Compare basic statistics for numerical columns
            print("\nStatistical comparison (original vs. anonymized):")
            for col in common_num_cols[:3]:  # Limit to first 3 columns to avoid too much output
                if col in df.columns and col in df_anonymized.columns:
                    try:
                        orig_mean = df[col].mean()
                        anon_mean = df_anonymized[col].mean()
                        orig_std = df[col].std()
                        anon_std = df_anonymized[col].std()
                        print(f"{col} - Original: mean={orig_mean:.2f}, std={orig_std:.2f} | "
                              f"Anonymized: mean={anon_mean:.2f}, std={anon_std:.2f}")
                    except:
                        pass
        
    except Exception as e:
        print(f"Error during anonymization: {e}")
        return


def main():
    """
    Main function to run the anonymization process when script is run directly.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run the complete anonymization process')
    parser.add_argument('--input_path', type=str, default=os.path.join('..', 'data', 'raw', 'synthetic_data.csv'),
                        help='Path to the input data file')
    parser.add_argument('--output_path', type=str, default=os.path.join('..', 'data', 'processed', 'complete_anonymized_data.csv'),
                        help='Path to save the anonymized data')
    parser.add_argument('--method', type=str, default='enhanced',
                        choices=['enhanced', 'k_anonymity', 'l_diversity', 't_closeness', 'differential_privacy'],
                        help='Anonymization method to use')
    
    args = parser.parse_args()
    
    # Run the anonymization process
    run_anonymization_process(
        input_path=args.input_path,
        output_path=args.output_path,
        anonymization_method=args.method
    )


if __name__ == "__main__":
    main()
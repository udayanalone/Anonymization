#!/usr/bin/env python3
"""
Script to run preprocessing on patient data for the first 2000 records.
"""

import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from preprocessing import preprocess_patient_data

def main():
    """Main function to run preprocessing."""
    
    # File paths
    input_file = "data/raw/patient.csv"
    output_file = "data/processed/patient_processed.csv"
    
    print("Starting patient data preprocessing...")
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    
    # Run preprocessing on first 2000 records
    df_processed = preprocess_patient_data(
        file_path=input_file,
        n_records=2000,
        output_path=output_file
    )
    
    if df_processed is not None:
        print(f"\n✅ Preprocessing completed successfully!")
        print(f"Final dataset shape: {df_processed.shape}")
        print(f"Processed data saved to: {output_file}")
    else:
        print("\n❌ Preprocessing failed!")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

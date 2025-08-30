import os
import pandas as pd
import sys
import glob
from pathlib import Path

# Add the src directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.enhanced_anonymization import identify_sensitive_attributes, anonymize_dataset


def load_data(file_path):
    """
    Load data from a CSV file.
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file
        
    Returns:
    --------
    pandas.DataFrame
        Loaded DataFrame
    """
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return None


def save_processed_data(df, file_path):
    """
    Save processed data to a CSV file.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame to save
    file_path : str
        Path to save the CSV file
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        print(f"Data successfully saved to {file_path}")
    except Exception as e:
        print(f"Error saving data to {file_path}: {e}")


def process_file(input_file, output_file):
    """
    Process a single CSV file: load, anonymize, and save.
    
    Parameters:
    -----------
    input_file : str
        Path to the input CSV file
    output_file : str
        Path to save the anonymized CSV file
        
    Returns:
    --------
    bool
        True if processing was successful, False otherwise
    """
    print(f"\nProcessing file: {input_file}")
    
    # Load data
    df = load_data(input_file)
    if df is None:
        return False
    
    print(f"Loaded data with shape: {df.shape}")
    
    # Identify sensitive attributes
    sensitive_attrs = identify_sensitive_attributes(df)
    print(f"Identified sensitive attributes: {sensitive_attrs}")
    
    # Anonymize data
    df_anonymized = anonymize_dataset(df, sensitive_attrs)
    
    # Save anonymized data
    save_processed_data(df_anonymized, output_file)
    
    return True


def batch_anonymize(input_dir, output_dir, prefix="anonymized_"):
    """
    Process all CSV files in the input directory and save anonymized versions
    to the output directory with the specified prefix.
    
    Parameters:
    -----------
    input_dir : str
        Directory containing CSV files to process
    output_dir : str
        Directory to save anonymized CSV files
    prefix : str, default="anonymized_"
        Prefix to add to output filenames
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all CSV files in the input directory
    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        return
    
    print(f"Found {len(csv_files)} CSV files to process")
    
    # Process each file
    successful = 0
    failed = 0
    
    for input_file in csv_files:
        # Get the filename without path
        filename = os.path.basename(input_file)
        
        # Create output filename with prefix
        output_file = os.path.join(output_dir, f"{prefix}{filename}")
        
        # Process the file
        if process_file(input_file, output_file):
            successful += 1
        else:
            failed += 1
    
    print(f"\nBatch processing complete!")
    print(f"Successfully processed: {successful} files")
    print(f"Failed to process: {failed} files")


def main():
    """
    Main function to run the batch anonymization process.
    """
    # Define input and output directories
    input_dir = "D:/Projects/Final_Year/Anonymization/data/raw"
    output_dir = "D:/Projects/Final_Year/Anonymization/data/processed"
    
    print("Starting batch anonymization process...")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    
    # Run batch anonymization
    batch_anonymize(input_dir, output_dir)


if __name__ == "__main__":
    main()
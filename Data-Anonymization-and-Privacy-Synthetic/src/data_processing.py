import pandas as pd
import numpy as np
import os

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
        DataFrame containing the loaded data
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    return pd.read_csv(file_path)


def clean_data(df):
    """
    Clean the data by handling missing values and outliers.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame to clean
        
    Returns:
    --------
    pandas.DataFrame
        Cleaned DataFrame
    """
    # Create a copy to avoid modifying the original DataFrame
    df_clean = df.copy()
    
    # Handle missing values
    for col in df_clean.columns:
        if df_clean[col].dtype in [np.float64, np.int64]:
            # Replace missing numerical values with median
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        else:
            # Replace missing categorical values with mode
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
    
    # Handle outliers using capping
    for col in df_clean.select_dtypes(include=[np.number]).columns:
        if col != 'PatientID' and col != 'Outcome':  # Skip ID and target columns
            # Calculate Q1, Q3 and IQR
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            
            # Define bounds for outliers
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Cap outliers
            df_clean[col] = df_clean[col].clip(lower_bound, upper_bound)
    
    return df_clean


def preprocess_data(df):
    """
    Preprocess the data for analysis and modeling.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame to preprocess
        
    Returns:
    --------
    pandas.DataFrame
        Preprocessed DataFrame
    """
    # Create a copy to avoid modifying the original DataFrame
    df_processed = df.copy()
    
    # Convert categorical variables to one-hot encoding
    if 'Gender' in df_processed.columns:
        df_processed = pd.get_dummies(df_processed, columns=['Gender'], drop_first=True)
    
    # Normalize numerical features
    numerical_cols = df_processed.select_dtypes(include=[np.number]).columns
    numerical_cols = [col for col in numerical_cols if col not in ['PatientID', 'Outcome']]
    
    for col in numerical_cols:
        df_processed[col] = (df_processed[col] - df_processed[col].mean()) / df_processed[col].std()
    
    return df_processed


def save_processed_data(df, output_path):
    """
    Save processed data to a CSV file.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame to save
    output_path : str
        Path to save the processed data
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Processed data saved successfully to {output_path}!")


def main():
    """
    Main function to process data when script is run directly.
    """
    # Default paths
    input_path = os.path.join('..', 'data', 'raw', 'synthetic_data.csv')
    output_path = os.path.join('..', 'data', 'processed', 'processed_data.csv')
    
    # Load, clean, preprocess, and save data
    df = load_data(input_path)
    df_clean = clean_data(df)
    df_processed = preprocess_data(df_clean)
    save_processed_data(df_processed, output_path)


if __name__ == "__main__":
    main()
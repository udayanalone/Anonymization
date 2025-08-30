import os
import pandas as pd
import numpy as np
import argparse
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# Import the CGAN model
from cgan_model import CGAN, train_cgan_model, generate_synthetic_data_with_cgan

# Import sensitive attribute identification functions from enhanced_anonymization
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.enhanced_anonymization import (
    identify_sensitive_attributes,
    is_name, is_ssn, is_email, is_phone_number,
    is_address, is_birthdate, is_credit_card_number
)


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
        Loaded data
    """
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return None


def identify_attribute_types(df):
    """
    Identify the types of attributes in the DataFrame.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame to analyze
        
    Returns:
    --------
    dict
        Dictionary mapping column names to attribute types
    """
    attribute_types = {}
    
    for column in df.columns:
        # Sample some values for faster processing
        sample_size = min(100, len(df))
        sample_values = df[column].dropna().astype(str).sample(sample_size, replace=True).tolist()
        
        # Check for different attribute types
        if any(is_name(value) for value in sample_values):
            attribute_types[column] = 'name'
        elif any(is_ssn(value) for value in sample_values):
            attribute_types[column] = 'ssn'
        elif any(is_email(value) for value in sample_values):
            attribute_types[column] = 'email'
        elif any(is_phone_number(value) for value in sample_values):
            attribute_types[column] = 'phone'
        elif any(is_address(value) for value in sample_values):
            attribute_types[column] = 'address'
        elif any(is_birthdate(value) for value in sample_values):
            attribute_types[column] = 'birthdate'
        elif any(is_credit_card_number(value) for value in sample_values):
            attribute_types[column] = 'credit_card'
    
    return attribute_types


def identify_condition_columns(df, sensitive_columns):
    """
    Identify columns that can be used as conditions for generating synthetic data.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame to analyze
    sensitive_columns : list
        List of sensitive column names
        
    Returns:
    --------
    dict
        Dictionary mapping sensitive columns to their condition columns
    """
    condition_columns = {}
    
    # Categorical columns with few unique values make good condition columns
    categorical_columns = []
    for column in df.columns:
        if column not in sensitive_columns:
            unique_values = df[column].nunique()
            if unique_values > 1 and unique_values <= 10:
                categorical_columns.append(column)
    
    # For each sensitive column, find the most correlated categorical column
    for sensitive_column in sensitive_columns:
        if not categorical_columns:
            # If no good categorical columns, use None (unconditional generation)
            condition_columns[sensitive_column] = None
            continue
        
        # Try to find correlations
        best_column = None
        best_correlation = 0
        
        for cat_column in categorical_columns:
            # For categorical sensitive columns, use chi-squared test
            if df[sensitive_column].dtype == 'object':
                # Create a contingency table
                contingency = pd.crosstab(df[sensitive_column], df[cat_column])
                # Use the shape of the table as a simple measure of association
                correlation = contingency.size
            else:
                # For numerical sensitive columns, use mean differences between categories
                means = df.groupby(cat_column)[sensitive_column].mean()
                correlation = means.max() - means.min() if len(means) > 1 else 0
            
            if correlation > best_correlation:
                best_correlation = correlation
                best_column = cat_column
        
        condition_columns[sensitive_column] = best_column
    
    return condition_columns


def train_models_for_sensitive_attributes(df, attribute_types, condition_columns, epochs=500):
    """
    Train CGAN models for each sensitive attribute.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the data
    attribute_types : dict
        Dictionary mapping column names to attribute types
    condition_columns : dict
        Dictionary mapping sensitive columns to their condition columns
    epochs : int, default=500
        Number of epochs to train each model for
        
    Returns:
    --------
    dict
        Dictionary mapping column names to trained CGAN models
    """
    models = {}
    
    for column, attr_type in attribute_types.items():
        print(f"\nTraining model for {column} (type: {attr_type})...")
        
        # Get condition column
        condition_column = condition_columns[column]
        
        if condition_column is None:
            # If no condition column, use a dummy condition
            conditions = pd.Series(['default'] * len(df))
        else:
            conditions = df[condition_column]
        
        # Train the model
        try:
            model = train_cgan_model(df[column], conditions, attr_type, epochs=epochs)
            models[column] = model
            print(f"Successfully trained model for {column}")
        except Exception as e:
            print(f"Error training model for {column}: {e}")
    
    return models


def save_models(models, output_dir):
    """
    Save trained models to disk.
    
    Parameters:
    -----------
    models : dict
        Dictionary mapping column names to trained CGAN models
    output_dir : str
        Directory to save models to
    """
    import pickle
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    for column, model in models.items():
        try:
            # Save model
            model_path = os.path.join(output_dir, f"{column}_model.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            print(f"Saved model for {column} to {model_path}")
        except Exception as e:
            print(f"Error saving model for {column}: {e}")


def load_models(model_dir):
    """
    Load trained models from disk.
    
    Parameters:
    -----------
    model_dir : str
        Directory containing saved models
        
    Returns:
    --------
    dict
        Dictionary mapping column names to loaded CGAN models
    """
    import pickle
    
    models = {}
    
    # Check if model directory exists
    if not os.path.exists(model_dir):
        print(f"Model directory {model_dir} does not exist")
        return models
    
    # Load models
    for filename in os.listdir(model_dir):
        if filename.endswith('_model.pkl'):
            column = filename.replace('_model.pkl', '')
            try:
                model_path = os.path.join(model_dir, filename)
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                models[column] = model
                print(f"Loaded model for {column} from {model_path}")
            except Exception as e:
                print(f"Error loading model for {column}: {e}")
    
    return models


def generate_synthetic_data(df, models, condition_columns):
    """
    Generate synthetic data for sensitive attributes.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the data
    models : dict
        Dictionary mapping column names to trained CGAN models
    condition_columns : dict
        Dictionary mapping sensitive columns to their condition columns
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with synthetic data for sensitive attributes
    """
    synthetic_df = df.copy()
    
    for column, model in models.items():
        print(f"\nGenerating synthetic data for {column}...")
        
        # Get condition column
        condition_column = condition_columns[column]
        
        if condition_column is None:
            # If no condition column, use a dummy condition
            conditions = ['default'] * len(df)
        else:
            conditions = df[condition_column].tolist()
        
        # Generate synthetic data
        try:
            synthetic_data = generate_synthetic_data_with_cgan(model, conditions)
            synthetic_df[column] = synthetic_data
            print(f"Successfully generated synthetic data for {column}")
        except Exception as e:
            print(f"Error generating synthetic data for {column}: {e}")
    
    return synthetic_df


def evaluate_synthetic_data(original_df, synthetic_df, sensitive_columns):
    """
    Evaluate the quality of synthetic data.
    
    Parameters:
    -----------
    original_df : pandas.DataFrame
        Original DataFrame
    synthetic_df : pandas.DataFrame
        DataFrame with synthetic data
    sensitive_columns : list
        List of sensitive column names
    """
    for column in sensitive_columns:
        print(f"\nEvaluating synthetic data for {column}...")
        
        # For categorical columns, compare value distributions
        if original_df[column].dtype == 'object':
            try:
                # Get value counts
                original_counts = original_df[column].value_counts(normalize=True)
                synthetic_counts = synthetic_df[column].value_counts(normalize=True)
                
                # Print top 5 values
                print("Top 5 original values:")
                print(original_counts.head())
                print("\nTop 5 synthetic values:")
                print(synthetic_counts.head())
                
                # Plot distributions
                plt.figure(figsize=(12, 6))
                plt.subplot(1, 2, 1)
                original_counts.head(10).plot(kind='bar')
                plt.title(f"Original {column} Distribution (Top 10)")
                plt.subplot(1, 2, 2)
                synthetic_counts.head(10).plot(kind='bar')
                plt.title(f"Synthetic {column} Distribution (Top 10)")
                plt.tight_layout()
                plt.savefig(f"{column}_distribution_comparison.png")
                plt.close()
            except Exception as e:
                print(f"Error comparing distributions for {column}: {e}")
        
        # For numerical columns, compare statistics
        else:
            try:
                # Calculate statistics
                original_stats = original_df[column].describe()
                synthetic_stats = synthetic_df[column].describe()
                
                # Print statistics
                print("Original statistics:")
                print(original_stats)
                print("\nSynthetic statistics:")
                print(synthetic_stats)
                
                # Plot distributions
                plt.figure(figsize=(12, 6))
                plt.subplot(1, 2, 1)
                sns.histplot(original_df[column].dropna(), kde=True)
                plt.title(f"Original {column} Distribution")
                plt.subplot(1, 2, 2)
                sns.histplot(synthetic_df[column].dropna(), kde=True)
                plt.title(f"Synthetic {column} Distribution")
                plt.tight_layout()
                plt.savefig(f"{column}_distribution_comparison.png")
                plt.close()
            except Exception as e:
                print(f"Error comparing distributions for {column}: {e}")


def main():
    """
    Main function to train CGAN models and generate synthetic data.
    """
    parser = argparse.ArgumentParser(description='Train CGAN models for sensitive attributes')
    parser.add_argument('--input', type=str, required=True, help='Path to input CSV file')
    parser.add_argument('--output', type=str, required=True, help='Path to output CSV file')
    parser.add_argument('--model_dir', type=str, default='models', help='Directory to save/load models')
    parser.add_argument('--epochs', type=int, default=500, help='Number of epochs to train each model for')
    parser.add_argument('--load_models', action='store_true', help='Load models from disk instead of training')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate synthetic data quality')
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.input}...")
    df = load_data(args.input)
    if df is None:
        return
    
    print(f"Loaded {len(df)} rows with {len(df.columns)} columns")
    
    # Identify sensitive attributes
    print("\nIdentifying sensitive attributes...")
    sensitive_attributes = identify_sensitive_attributes(df)
    sensitive_columns = list(sensitive_attributes.keys())
    
    if not sensitive_columns:
        print("No sensitive attributes found in the data")
        return
    
    print(f"Found {len(sensitive_columns)} sensitive columns: {sensitive_columns}")
    
    # Identify attribute types
    print("\nIdentifying attribute types...")
    attribute_types = identify_attribute_types(df)
    print(f"Attribute types: {attribute_types}")
    
    # Identify condition columns
    print("\nIdentifying condition columns...")
    condition_columns = identify_condition_columns(df, sensitive_columns)
    print(f"Condition columns: {condition_columns}")
    
    # Train or load models
    if args.load_models:
        print(f"\nLoading models from {args.model_dir}...")
        models = load_models(args.model_dir)
    else:
        print(f"\nTraining models...")
        models = train_models_for_sensitive_attributes(df, attribute_types, condition_columns, epochs=args.epochs)
        
        # Save models
        print(f"\nSaving models to {args.model_dir}...")
        save_models(models, args.model_dir)
    
    # Generate synthetic data
    print("\nGenerating synthetic data...")
    synthetic_df = generate_synthetic_data(df, models, condition_columns)
    
    # Save synthetic data
    print(f"\nSaving synthetic data to {args.output}...")
    synthetic_df.to_csv(args.output, index=False)
    print(f"Saved synthetic data to {args.output}")
    
    # Evaluate synthetic data
    if args.evaluate:
        print("\nEvaluating synthetic data...")
        evaluate_synthetic_data(df, synthetic_df, sensitive_columns)


if __name__ == "__main__":
    main()
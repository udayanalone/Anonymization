import os
import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

# Import the enhanced anonymization module
from enhanced_anonymization import identify_sensitive_attributes, anonymize_dataset

# Import CGAN model functions if available
try:
    from cgan_model import CGAN, train_cgan_model, generate_synthetic_data_with_cgan
    has_cgan = True
except ImportError:
    has_cgan = False


def identify_potential_conditions(df, sensitive_columns):
    """
    Identify potential condition columns for each sensitive column.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame to analyze
    sensitive_columns : list
        List of sensitive column names
        
    Returns:
    --------
    dict
        Dictionary mapping sensitive columns to lists of potential condition columns
    """
    potential_conditions = {}
    
    for sensitive_column in sensitive_columns:
        potential_conditions[sensitive_column] = []
        
        # Look for categorical columns with few unique values
        for column in df.columns:
            if column not in sensitive_columns and column != sensitive_column:
                unique_values = df[column].nunique()
                if unique_values > 1 and unique_values <= 10:
                    potential_conditions[sensitive_column].append(column)
    
    return potential_conditions


def analyze_correlations(df, sensitive_column, condition_columns):
    """
    Analyze correlations between a sensitive column and potential condition columns.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame to analyze
    sensitive_column : str
        Name of the sensitive column
    condition_columns : list
        List of potential condition column names
        
    Returns:
    --------
    dict
        Dictionary with correlation metrics for each condition column
    """
    correlations = {}
    
    for condition_column in condition_columns:
        # For categorical sensitive columns, use chi-squared test or contingency table analysis
        if df[sensitive_column].dtype == 'object':
            # Create a contingency table
            contingency = pd.crosstab(df[sensitive_column], df[condition_column])
            # Use the shape of the table as a simple measure of association
            correlations[condition_column] = {
                'metric': 'contingency_size',
                'value': contingency.size
            }
        else:
            # For numerical sensitive columns, use mean differences between categories
            means = df.groupby(condition_column)[sensitive_column].mean()
            correlations[condition_column] = {
                'metric': 'mean_difference',
                'value': means.max() - means.min() if len(means) > 1 else 0
            }
    
    return correlations


def select_best_conditions(correlations, max_conditions=3):
    """
    Select the best condition columns based on correlation metrics.
    
    Parameters:
    -----------
    correlations : dict
        Dictionary with correlation metrics for each condition column
    max_conditions : int, default=3
        Maximum number of condition columns to select
        
    Returns:
    --------
    list
        List of selected condition column names
    """
    # Sort condition columns by correlation value (descending)
    sorted_conditions = sorted(correlations.items(), key=lambda x: x[1]['value'], reverse=True)
    
    # Select top N conditions
    selected_conditions = [condition for condition, _ in sorted_conditions[:max_conditions]]
    
    return selected_conditions


def create_combined_condition(df, condition_columns):
    """
    Create a combined condition column from multiple condition columns.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the condition columns
    condition_columns : list
        List of condition column names
        
    Returns:
    --------
    pandas.Series
        Combined condition column
    """
    if not condition_columns:
        return pd.Series(['default'] * len(df))
    
    # Combine condition columns into a single string column
    combined_condition = df[condition_columns[0]].astype(str)
    
    for column in condition_columns[1:]:
        combined_condition = combined_condition + '_' + df[column].astype(str)
    
    return combined_condition


def train_dynamic_cgan_models(df, sensitive_attributes):
    """
    Train CGAN models for each sensitive attribute with dynamic conditions.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the data
    sensitive_attributes : dict
        Dictionary mapping column names to attribute types
        
    Returns:
    --------
    dict
        Dictionary mapping column names to trained CGAN models
    dict
        Dictionary mapping column names to their condition columns
    """
    if not has_cgan:
        print("CGAN module not available. Cannot train models.")
        return {}, {}
    
    models = {}
    used_conditions = {}
    sensitive_columns = list(sensitive_attributes.keys())
    
    # Identify potential condition columns for each sensitive column
    potential_conditions = identify_potential_conditions(df, sensitive_columns)
    
    for column, attr_type in sensitive_attributes.items():
        print(f"\nAnalyzing conditions for {column} (type: {attr_type})...")
        
        # Analyze correlations
        correlations = analyze_correlations(df, column, potential_conditions[column])
        
        # Select best conditions
        selected_conditions = select_best_conditions(correlations)
        used_conditions[column] = selected_conditions
        
        if selected_conditions:
            print(f"Selected conditions for {column}: {selected_conditions}")
            
            # Create combined condition
            combined_condition = create_combined_condition(df, selected_conditions)
            
            # Train the model
            try:
                print(f"Training CGAN model for {column} with dynamic conditions...")
                model = train_cgan_model(df[column], combined_condition, attr_type, epochs=200)
                models[column] = model
                print(f"Successfully trained model for {column}")
            except Exception as e:
                print(f"Error training model for {column}: {e}")
        else:
            print(f"No suitable conditions found for {column}. Using unconditional generation.")
            try:
                # Train with dummy condition
                dummy_condition = pd.Series(['default'] * len(df))
                model = train_cgan_model(df[column], dummy_condition, attr_type, epochs=200)
                models[column] = model
                print(f"Successfully trained unconditional model for {column}")
            except Exception as e:
                print(f"Error training model for {column}: {e}")
    
    return models, used_conditions


def generate_dynamic_synthetic_data(df, models, used_conditions):
    """
    Generate synthetic data using trained CGAN models with dynamic conditions.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the data
    models : dict
        Dictionary mapping column names to trained CGAN models
    used_conditions : dict
        Dictionary mapping column names to their condition columns
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with synthetic data for sensitive attributes
    """
    if not has_cgan:
        print("CGAN module not available. Cannot generate synthetic data.")
        return df.copy()
    
    synthetic_df = df.copy()
    
    for column, model in models.items():
        print(f"\nGenerating synthetic data for {column}...")
        
        # Get condition columns
        condition_columns = used_conditions.get(column, [])
        
        if condition_columns:
            # Create combined condition
            combined_condition = create_combined_condition(df, condition_columns)
            conditions = combined_condition.tolist()
        else:
            # Use dummy condition
            conditions = ['default'] * len(df)
        
        # Generate synthetic data
        try:
            synthetic_data = generate_synthetic_data_with_cgan(model, conditions)
            synthetic_df[column] = synthetic_data
            print(f"Successfully generated synthetic data for {column}")
        except Exception as e:
            print(f"Error generating synthetic data for {column}: {e}")
    
    return synthetic_df


def visualize_conditional_distributions(original_df, synthetic_df, column, condition_column):
    """
    Visualize the distributions of original and synthetic data conditioned on a specific column.
    
    Parameters:
    -----------
    original_df : pandas.DataFrame
        Original DataFrame
    synthetic_df : pandas.DataFrame
        DataFrame with synthetic data
    column : str
        Name of the column to visualize
    condition_column : str
        Name of the condition column
    """
    # Get unique condition values
    condition_values = original_df[condition_column].unique()
    
    # Create a figure with subplots for each condition value
    n_conditions = len(condition_values)
    fig, axes = plt.subplots(n_conditions, 2, figsize=(12, 4 * n_conditions))
    
    for i, condition_value in enumerate(condition_values):
        # Filter data for this condition value
        original_filtered = original_df[original_df[condition_column] == condition_value][column]
        synthetic_filtered = synthetic_df[synthetic_df[condition_column] == condition_value][column]
        
        # Plot original data
        if original_df[column].dtype == 'object':
            # For categorical data, use bar plots
            original_counts = original_filtered.value_counts(normalize=True)
            original_counts.head(10).plot(kind='bar', ax=axes[i, 0])
        else:
            # For numerical data, use histograms
            sns.histplot(original_filtered, kde=True, ax=axes[i, 0])
        
        axes[i, 0].set_title(f"Original {column} for {condition_column}={condition_value}")
        
        # Plot synthetic data
        if synthetic_df[column].dtype == 'object':
            # For categorical data, use bar plots
            synthetic_counts = synthetic_filtered.value_counts(normalize=True)
            synthetic_counts.head(10).plot(kind='bar', ax=axes[i, 1])
        else:
            # For numerical data, use histograms
            sns.histplot(synthetic_filtered, kde=True, ax=axes[i, 1])
        
        axes[i, 1].set_title(f"Synthetic {column} for {condition_column}={condition_value}")
    
    plt.tight_layout()
    plt.savefig(f"{column}_by_{condition_column}_comparison.png")
    plt.close()


def main():
    """
    Main function to demonstrate dynamic condition-based anonymization.
    """
    parser = argparse.ArgumentParser(description='Dynamic condition-based anonymization with CGAN')
    parser.add_argument('--input', type=str, default=os.path.join('..', 'data', 'raw', 'synthetic_data.csv'),
                        help='Path to input CSV file')
    parser.add_argument('--output', type=str, default=os.path.join('..', 'data', 'processed', 'dynamic_anonymized_data.csv'),
                        help='Path to output CSV file')
    parser.add_argument('--visualize', action='store_true', help='Visualize conditional distributions')
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
    
    # For demonstration, add some synthetic sensitive attributes if they don't exist
    if 'FullName' not in df.columns:
        print("Adding synthetic sensitive attributes for demonstration...")
        
        # Add sensitive attributes
        import random
        import string
        from faker import Faker
        fake = Faker()
        
        df['FullName'] = [fake.name() for _ in range(len(df))]
        df['Email'] = [fake.email() for _ in range(len(df))]
        df['SSN'] = [f"{random.randint(100, 999)}-{random.randint(10, 99)}-{random.randint(1000, 9999)}" for _ in range(len(df))]
        
        # Add some condition columns if they don't exist
        if 'Gender' not in df.columns:
            df['Gender'] = [random.choice(['Male', 'Female']) for _ in range(len(df))]
        
        if 'AgeGroup' not in df.columns and 'Age' in df.columns:
            df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 30, 50, 100], labels=['Young', 'Middle', 'Senior'])
        elif 'AgeGroup' not in df.columns:
            df['AgeGroup'] = [random.choice(['Young', 'Middle', 'Senior']) for _ in range(len(df))]
        
        if 'Region' not in df.columns:
            df['Region'] = [random.choice(['North', 'South', 'East', 'West']) for _ in range(len(df))]
    
    # Identify sensitive attributes
    print("\nIdentifying sensitive attributes...")
    sensitive_attributes = identify_sensitive_attributes(df)
    
    if not sensitive_attributes:
        print("No sensitive attributes found in the data. Adding some for demonstration...")
        sensitive_attributes = {
            'FullName': 'name',
            'Email': 'email',
            'SSN': 'ssn'
        }
    
    print(f"Found {len(sensitive_attributes)} sensitive attributes: {sensitive_attributes}")
    
    # Train CGAN models with dynamic conditions
    if has_cgan:
        print("\nTraining CGAN models with dynamic conditions...")
        models, used_conditions = train_dynamic_cgan_models(df, sensitive_attributes)
        
        # Generate synthetic data
        if models:
            print("\nGenerating synthetic data with dynamic conditions...")
            synthetic_df = generate_dynamic_synthetic_data(df, models, used_conditions)
            
            # Save synthetic data
            print(f"\nSaving synthetic data to {args.output}...")
            synthetic_df.to_csv(args.output, index=False)
            print(f"Saved synthetic data to {args.output}")
            
            # Visualize conditional distributions
            if args.visualize:
                print("\nVisualizing conditional distributions...")
                for column in models.keys():
                    for condition_column in used_conditions.get(column, []):
                        print(f"Visualizing {column} by {condition_column}...")
                        visualize_conditional_distributions(df, synthetic_df, column, condition_column)
        else:
            print("No CGAN models were successfully trained. Using traditional anonymization.")
            synthetic_df = anonymize_dataset(df, sensitive_attributes)
            synthetic_df.to_csv(args.output, index=False)
    else:
        # Fall back to traditional anonymization
        print("\nUsing traditional anonymization methods...")
        synthetic_df = anonymize_dataset(df, sensitive_attributes)
        synthetic_df.to_csv(args.output, index=False)
    
    print("\nDynamic anonymization process completed!")


if __name__ == "__main__":
    main()
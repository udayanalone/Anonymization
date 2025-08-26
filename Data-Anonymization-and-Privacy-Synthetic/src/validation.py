import pandas as pd
import numpy as np
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


def calculate_information_loss(original_df, anonymized_df, numerical_columns=None):
    """
    Calculate information loss between original and anonymized datasets.
    
    Parameters:
    -----------
    original_df : pandas.DataFrame
        Original DataFrame before anonymization
    anonymized_df : pandas.DataFrame
        Anonymized DataFrame
    numerical_columns : list, optional
        List of numerical columns to consider. If None, all numerical columns are used.
        
    Returns:
    --------
    dict
        Dictionary containing various information loss metrics
    """
    # Ensure both DataFrames have the same columns
    common_columns = list(set(original_df.columns) & set(anonymized_df.columns))
    original_df = original_df[common_columns]
    anonymized_df = anonymized_df[common_columns]
    
    # Identify numerical columns if not provided
    if numerical_columns is None:
        numerical_columns = original_df.select_dtypes(include=[np.number]).columns.tolist()
        # Exclude ID columns
        numerical_columns = [col for col in numerical_columns if 'ID' not in col]
    
    # Calculate normalized average error
    nae_values = {}
    for col in numerical_columns:
        if col in original_df.columns and col in anonymized_df.columns:
            # Calculate range for normalization
            col_range = original_df[col].max() - original_df[col].min()
            if col_range == 0:  # Avoid division by zero
                nae_values[col] = 0
            else:
                # Calculate normalized average error
                nae = np.mean(np.abs(original_df[col] - anonymized_df[col])) / col_range
                nae_values[col] = nae
    
    # Calculate average information loss across all columns
    avg_info_loss = np.mean(list(nae_values.values())) if nae_values else 0
    
    return {
        'column_info_loss': nae_values,
        'average_info_loss': avg_info_loss
    }


def evaluate_utility_preservation(original_df, anonymized_df, target_column='Outcome'):
    """
    Evaluate how well the anonymized data preserves utility for machine learning tasks.
    
    Parameters:
    -----------
    original_df : pandas.DataFrame
        Original DataFrame before anonymization
    anonymized_df : pandas.DataFrame
        Anonymized DataFrame
    target_column : str, default='Outcome'
        Name of the target column for prediction
        
    Returns:
    --------
    dict
        Dictionary containing utility preservation metrics
    """
    # Ensure both DataFrames have the same columns
    common_columns = list(set(original_df.columns) & set(anonymized_df.columns))
    original_df = original_df[common_columns]
    anonymized_df = anonymized_df[common_columns]
    
    # Check if target column exists in both DataFrames
    if target_column not in original_df.columns or target_column not in anonymized_df.columns:
        raise ValueError(f"Target column '{target_column}' not found in both DataFrames")
    
    # Prepare features and target
    feature_columns = [col for col in common_columns if col != target_column and 'ID' not in col]
    
    X_orig = original_df[feature_columns]
    y_orig = original_df[target_column]
    
    X_anon = anonymized_df[feature_columns]
    y_anon = anonymized_df[target_column]
    
    # Split data
    X_orig_train, X_orig_test, y_orig_train, y_orig_test = train_test_split(
        X_orig, y_orig, test_size=0.3, random_state=42)
    
    X_anon_train, X_anon_test, y_anon_train, y_anon_test = train_test_split(
        X_anon, y_anon, test_size=0.3, random_state=42)
    
    # Train models
    model_orig = RandomForestClassifier(random_state=42)
    model_orig.fit(X_orig_train, y_orig_train)
    
    model_anon = RandomForestClassifier(random_state=42)
    model_anon.fit(X_anon_train, y_anon_train)
    
    # Make predictions
    y_orig_pred = model_orig.predict(X_orig_test)
    y_anon_pred = model_anon.predict(X_anon_test)
    
    # Calculate metrics
    orig_metrics = {
        'accuracy': accuracy_score(y_orig_test, y_orig_pred),
        'precision': precision_score(y_orig_test, y_orig_pred, zero_division=0),
        'recall': recall_score(y_orig_test, y_orig_pred, zero_division=0),
        'f1': f1_score(y_orig_test, y_orig_pred, zero_division=0)
    }
    
    anon_metrics = {
        'accuracy': accuracy_score(y_anon_test, y_anon_pred),
        'precision': precision_score(y_anon_test, y_anon_pred, zero_division=0),
        'recall': recall_score(y_anon_test, y_anon_pred, zero_division=0),
        'f1': f1_score(y_anon_test, y_anon_pred, zero_division=0)
    }
    
    # Calculate utility preservation (as percentage of original performance)
    utility_preservation = {}
    for metric in orig_metrics:
        if orig_metrics[metric] == 0:  # Avoid division by zero
            utility_preservation[metric] = 0
        else:
            utility_preservation[metric] = (anon_metrics[metric] / orig_metrics[metric]) * 100
    
    return {
        'original_metrics': orig_metrics,
        'anonymized_metrics': anon_metrics,
        'utility_preservation': utility_preservation
    }


def check_k_anonymity(df, quasi_identifiers, k=5):
    """
    Check if the dataset satisfies k-anonymity.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame to check
    quasi_identifiers : list
        List of column names considered as quasi-identifiers
    k : int, default=5
        The k value for k-anonymity
        
    Returns:
    --------
    dict
        Dictionary containing k-anonymity validation results
    """
    # Group by quasi-identifiers and count records in each group
    group_counts = df.groupby(quasi_identifiers).size()
    
    # Check if any group has fewer than k records
    violations = group_counts[group_counts < k]
    
    return {
        'satisfies_k_anonymity': len(violations) == 0,
        'violation_count': len(violations),
        'min_group_size': group_counts.min() if not group_counts.empty else 0,
        'max_group_size': group_counts.max() if not group_counts.empty else 0,
        'avg_group_size': group_counts.mean() if not group_counts.empty else 0
    }


def check_l_diversity(df, quasi_identifiers, sensitive_attribute, l=3):
    """
    Check if the dataset satisfies l-diversity.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame to check
    quasi_identifiers : list
        List of column names considered as quasi-identifiers
    sensitive_attribute : str
        Column name of the sensitive attribute
    l : int, default=3
        The l value for l-diversity
        
    Returns:
    --------
    dict
        Dictionary containing l-diversity validation results
    """
    # Group by quasi-identifiers
    groups = df.groupby(quasi_identifiers)
    
    # Check l-diversity for each group
    diversity_counts = {}
    for name, group in groups:
        diversity_counts[name] = group[sensitive_attribute].nunique()
    
    # Convert to Series for easier analysis
    diversity_series = pd.Series(diversity_counts)
    
    # Check violations
    violations = diversity_series[diversity_series < l]
    
    return {
        'satisfies_l_diversity': len(violations) == 0,
        'violation_count': len(violations),
        'min_diversity': diversity_series.min() if not diversity_series.empty else 0,
        'max_diversity': diversity_series.max() if not diversity_series.empty else 0,
        'avg_diversity': diversity_series.mean() if not diversity_series.empty else 0
    }


def validate_anonymization(original_df, anonymized_df, quasi_identifiers, sensitive_attribute):
    """
    Validate anonymization by checking privacy guarantees and utility preservation.
    
    Parameters:
    -----------
    original_df : pandas.DataFrame
        Original DataFrame before anonymization
    anonymized_df : pandas.DataFrame
        Anonymized DataFrame
    quasi_identifiers : list
        List of column names considered as quasi-identifiers
    sensitive_attribute : str
        Column name of the sensitive attribute
        
    Returns:
    --------
    dict
        Dictionary containing validation results
    """
    # Check k-anonymity (with k=5)
    k_anonymity_results = check_k_anonymity(anonymized_df, quasi_identifiers, k=5)
    
    # Check l-diversity (with l=3)
    l_diversity_results = check_l_diversity(anonymized_df, quasi_identifiers, sensitive_attribute, l=3)
    
    # Calculate information loss
    info_loss = calculate_information_loss(original_df, anonymized_df)
    
    # Evaluate utility preservation
    utility = evaluate_utility_preservation(original_df, anonymized_df, target_column=sensitive_attribute)
    
    return {
        'k_anonymity': k_anonymity_results,
        'l_diversity': l_diversity_results,
        'information_loss': info_loss,
        'utility_preservation': utility
    }


def main():
    """
    Main function to validate anonymization when script is run directly.
    """
    # Default paths
    original_path = os.path.join('..', 'data', 'processed', 'processed_data.csv')
    anonymized_path = os.path.join('..', 'data', 'processed', 'anonymized_data.csv')
    
    # Load data
    original_df = pd.read_csv(original_path)
    anonymized_df = pd.read_csv(anonymized_path)
    
    # Define quasi-identifiers and sensitive attribute
    quasi_identifiers = ['Age', 'Gender_Male']
    sensitive_attribute = 'Outcome'
    
    # Validate anonymization
    results = validate_anonymization(original_df, anonymized_df, quasi_identifiers, sensitive_attribute)
    
    # Print results
    print("\nAnonymization Validation Results:")
    print("==================================")
    
    print("\nK-Anonymity:")
    print(f"  Satisfies k-anonymity: {results['k_anonymity']['satisfies_k_anonymity']}")
    print(f"  Violation count: {results['k_anonymity']['violation_count']}")
    print(f"  Min group size: {results['k_anonymity']['min_group_size']}")
    
    print("\nL-Diversity:")
    print(f"  Satisfies l-diversity: {results['l_diversity']['satisfies_l_diversity']}")
    print(f"  Violation count: {results['l_diversity']['violation_count']}")
    print(f"  Min diversity: {results['l_diversity']['min_diversity']}")
    
    print("\nInformation Loss:")
    print(f"  Average information loss: {results['information_loss']['average_info_loss']:.4f}")
    
    print("\nUtility Preservation:")
    print(f"  Accuracy preservation: {results['utility_preservation']['utility_preservation']['accuracy']:.2f}%")
    print(f"  F1-score preservation: {results['utility_preservation']['utility_preservation']['f1']:.2f}%")


if __name__ == "__main__":
    main()
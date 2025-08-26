import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import KBinsDiscretizer


def k_anonymize(df, sensitive_columns, k=5):
    """
    Apply k-anonymity to the dataset.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame to anonymize
    sensitive_columns : list
        List of column names to be considered as quasi-identifiers
    k : int, default=5
        The k value for k-anonymity (minimum group size)
        
    Returns:
    --------
    pandas.DataFrame
        k-anonymized DataFrame
    """
    # Create a copy to avoid modifying the original DataFrame
    df_anonymized = df.copy()
    
    # Apply binning/generalization to quasi-identifiers
    for col in sensitive_columns:
        if df_anonymized[col].dtype in [np.float64, np.int64]:
            # For numerical columns, use binning
            n_bins = max(3, len(df_anonymized) // (k * 5))  # Determine number of bins
            
            # Create discretizer
            discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
            
            # Fit and transform
            binned_values = discretizer.fit_transform(df_anonymized[[col]])
            
            # Replace original values with bin indices
            df_anonymized[col] = binned_values
        else:
            # For categorical columns, use generalization if needed
            # Check if we need to generalize (if any category has fewer than k records)
            value_counts = df_anonymized[col].value_counts()
            rare_categories = value_counts[value_counts < k].index.tolist()
            
            if rare_categories:
                # Replace rare categories with 'Other'
                df_anonymized.loc[df_anonymized[col].isin(rare_categories), col] = 'Other'
    
    # Verify k-anonymity
    group_counts = df_anonymized.groupby(sensitive_columns).size()
    if (group_counts < k).any():
        print(f"Warning: k-anonymity with k={k} not fully achieved. Some groups still have fewer than {k} records.")
    
    return df_anonymized


def l_diversity(df, sensitive_columns, sensitive_attribute, l=3):
    """
    Apply l-diversity to the dataset.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame to anonymize
    sensitive_columns : list
        List of column names to be considered as quasi-identifiers
    sensitive_attribute : str
        Column name of the sensitive attribute
    l : int, default=3
        The l value for l-diversity (minimum number of distinct values in each group)
        
    Returns:
    --------
    pandas.DataFrame
        l-diverse DataFrame
    """
    # First apply k-anonymity
    df_diverse = k_anonymize(df, sensitive_columns)
    
    # Check l-diversity for each group
    groups = df_diverse.groupby(sensitive_columns)
    
    # Identify groups that don't satisfy l-diversity
    non_diverse_groups = []
    for name, group in groups:
        distinct_values = group[sensitive_attribute].nunique()
        if distinct_values < l:
            if isinstance(name, tuple):
                non_diverse_groups.extend(list(group.index))
            else:
                non_diverse_groups.extend(list(group.index))
    
    # Remove records from non-diverse groups
    if non_diverse_groups:
        print(f"Removing {len(non_diverse_groups)} records to achieve l-diversity.")
        df_diverse = df_diverse.drop(non_diverse_groups)
    
    return df_diverse


def t_closeness(df, sensitive_columns, sensitive_attribute, t=0.2):
    """
    Apply t-closeness to the dataset.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame to anonymize
    sensitive_columns : list
        List of column names to be considered as quasi-identifiers
    sensitive_attribute : str
        Column name of the sensitive attribute
    t : float, default=0.2
        The t value for t-closeness (maximum allowed distance)
        
    Returns:
    --------
    pandas.DataFrame
        t-close DataFrame
    """
    # First apply k-anonymity
    df_tclose = k_anonymize(df, sensitive_columns)
    
    # Calculate global distribution of sensitive attribute
    global_dist = df_tclose[sensitive_attribute].value_counts(normalize=True)
    
    # Check t-closeness for each group
    groups = df_tclose.groupby(sensitive_columns)
    
    # Identify groups that don't satisfy t-closeness
    non_tclose_groups = []
    for name, group in groups:
        # Calculate group distribution
        group_dist = group[sensitive_attribute].value_counts(normalize=True)
        
        # Align distributions (fill missing categories with 0)
        group_dist = group_dist.reindex(global_dist.index, fill_value=0)
        
        # Calculate Earth Mover's Distance (simplified as absolute difference sum)
        distance = sum(abs(global_dist - group_dist)) / 2
        
        if distance > t:
            if isinstance(name, tuple):
                non_tclose_groups.extend(list(group.index))
            else:
                non_tclose_groups.extend(list(group.index))
    
    # Remove records from non-t-close groups
    if non_tclose_groups:
        print(f"Removing {len(non_tclose_groups)} records to achieve t-closeness.")
        df_tclose = df_tclose.drop(non_tclose_groups)
    
    return df_tclose


def differential_privacy(df, epsilon=1.0):
    """
    Apply differential privacy to the dataset by adding Laplace noise.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame to anonymize
    epsilon : float, default=1.0
        Privacy parameter (smaller values provide stronger privacy)
        
    Returns:
    --------
    pandas.DataFrame
        Differentially private DataFrame
    """
    # Create a copy to avoid modifying the original DataFrame
    df_private = df.copy()
    
    # Add Laplace noise to numerical columns
    for col in df_private.select_dtypes(include=[np.number]).columns:
        if col != 'PatientID':  # Skip ID column
            # Calculate sensitivity (assuming normalized data, sensitivity is often 1/n)
            sensitivity = 1.0 / len(df_private)
            
            # Calculate scale for Laplace noise
            scale = sensitivity / epsilon
            
            # Add noise
            noise = np.random.laplace(0, scale, size=len(df_private))
            df_private[col] = df_private[col] + noise
    
    return df_private


def anonymize_data(df, method='k_anonymity', **kwargs):
    """
    Anonymize data using the specified method.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame to anonymize
    method : str, default='k_anonymity'
        Anonymization method to use ('k_anonymity', 'l_diversity', 't_closeness', 'differential_privacy')
    **kwargs : dict
        Additional parameters for the specific anonymization method
        
    Returns:
    --------
    pandas.DataFrame
        Anonymized DataFrame
    """
    if method == 'k_anonymity':
        sensitive_columns = kwargs.get('sensitive_columns', ['Age', 'Gender_Male'])
        k = kwargs.get('k', 5)
        return k_anonymize(df, sensitive_columns, k)
    
    elif method == 'l_diversity':
        sensitive_columns = kwargs.get('sensitive_columns', ['Age', 'Gender_Male'])
        sensitive_attribute = kwargs.get('sensitive_attribute', 'Outcome')
        l = kwargs.get('l', 3)
        return l_diversity(df, sensitive_columns, sensitive_attribute, l)
    
    elif method == 't_closeness':
        sensitive_columns = kwargs.get('sensitive_columns', ['Age', 'Gender_Male'])
        sensitive_attribute = kwargs.get('sensitive_attribute', 'Outcome')
        t = kwargs.get('t', 0.2)
        return t_closeness(df, sensitive_columns, sensitive_attribute, t)
    
    elif method == 'differential_privacy':
        epsilon = kwargs.get('epsilon', 1.0)
        return differential_privacy(df, epsilon)
    
    else:
        raise ValueError(f"Unknown anonymization method: {method}")


def main():
    """
    Main function to anonymize data when script is run directly.
    """
    # Default paths
    input_path = os.path.join('..', 'data', 'processed', 'processed_data.csv')
    output_path = os.path.join('..', 'data', 'processed', 'anonymized_data.csv')
    
    # Load data
    df = pd.read_csv(input_path)
    
    # Anonymize data using k-anonymity
    df_anonymized = anonymize_data(df, method='k_anonymity', 
                                  sensitive_columns=['Age', 'Gender_Male'], 
                                  k=5)
    
    # Save anonymized data
    df_anonymized.to_csv(output_path, index=False)
    print(f"Anonymized data saved successfully to {output_path}!")


if __name__ == "__main__":
    main()
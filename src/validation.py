import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

class PrivacyValidator:
    """
    Class for validating privacy protection of anonymized data.
    """
    
    def __init__(self):
        """Initialize the privacy validator."""
        pass
    
    def calculate_k_anonymity(self, df, quasi_identifier_cols, k_values=[2, 5, 10, 20]):
        """
        Calculate k-anonymity for different values of k.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Dataset to evaluate
        quasi_identifier_cols : list
            List of quasi-identifier column names
        k_values : list
            List of k values to test
            
        Returns:
        --------
        dict
            Dictionary containing k-anonymity results
        """
        print("Calculating k-anonymity...")
        
        results = {}
        
        for k in k_values:
            # Group by quasi-identifiers and count occurrences
            groups = df[quasi_identifier_cols].groupby(quasi_identifier_cols).size()
            
            # Check if all groups have at least k members
            min_group_size = groups.min()
            k_anon = min_group_size >= k
            
            results[f'k={k}'] = {
                'achieved': k_anon,
                'min_group_size': min_group_size,
                'total_groups': len(groups),
                'groups_below_k': len(groups[groups < k])
            }
            
            print(f"  k={k}: {'✓' if k_anon else '✗'} (min group size: {min_group_size})")
        
        return results
    
    def calculate_l_diversity(self, df, quasi_identifier_cols, sensitive_cols, l_values=[2, 3, 5]):
        """
        Calculate l-diversity for different values of l.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Dataset to evaluate
        quasi_identifier_cols : list
            List of quasi-identifier column names
        sensitive_cols : list
            List of sensitive attribute column names
        l_values : list
            List of l values to test
            
        Returns:
        --------
        dict
            Dictionary containing l-diversity results
        """
        print("Calculating l-diversity...")
        
        results = {}
        
        for l in l_values:
            # Group by quasi-identifiers
            groups = df.groupby(quasi_identifier_cols)
            
            l_diverse_groups = 0
            total_groups = len(groups)
            
            for name, group in groups:
                # Check if all sensitive attributes have at least l distinct values
                group_l_diverse = True
                for sensitive_col in sensitive_cols:
                    if sensitive_col in group.columns:
                        distinct_values = group[sensitive_col].nunique()
                        if distinct_values < l:
                            group_l_diverse = False
                            break
                
                if group_l_diverse:
                    l_diverse_groups += 1
            
            l_diversity_ratio = l_diverse_groups / total_groups if total_groups > 0 else 0
            
            results[f'l={l}'] = {
                'achieved': l_diversity_ratio >= 0.8,  # Consider achieved if 80%+ groups are l-diverse
                'l_diverse_groups': l_diverse_groups,
                'total_groups': total_groups,
                'ratio': l_diversity_ratio
            }
            
            print(f"  l={l}: {'✓' if results[f'l={l}']['achieved'] else '✗'} "
                  f"({l_diverse_groups}/{total_groups} groups, {l_diversity_ratio:.2%})")
        
        return results
    
    def calculate_t_closeness(self, df, quasi_identifier_cols, sensitive_cols, t_values=[0.1, 0.2, 0.3]):
        """
        Calculate t-closeness for different values of t.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Dataset to evaluate
        quasi_identifier_cols : list
            List of quasi-identifier column names
        sensitive_cols : list
            List of sensitive attribute column names
        t_values : list
            List of t values to test
            
        Returns:
        --------
        dict
            Dictionary containing t-closeness results
        """
        print("Calculating t-closeness...")
        
        results = {}
        
        for t in t_values:
            # Group by quasi-identifiers
            groups = df.groupby(quasi_identifier_cols)
            
            t_close_groups = 0
            total_groups = len(groups)
            
            for name, group in groups:
                # Check if distribution of sensitive attributes is close to overall distribution
                group_t_close = True
                for sensitive_col in sensitive_cols:
                    if sensitive_col in group.columns:
                        # Calculate distribution difference
                        overall_dist = df[sensitive_col].value_counts(normalize=True)
                        group_dist = group[sensitive_col].value_counts(normalize=True)
                        
                        # Fill missing values with 0
                        for val in overall_dist.index:
                            if val not in group_dist.index:
                                group_dist[val] = 0
                        
                        # Reorder to match overall distribution
                        group_dist = group_dist.reindex(overall_dist.index, fill_value=0)
                        
                        # Calculate total variation distance
                        tv_distance = 0.5 * np.sum(np.abs(overall_dist.values - group_dist.values))
                        
                        if tv_distance > t:
                            group_t_close = False
                            break
                
                if group_t_close:
                    t_close_groups += 1
            
            t_closeness_ratio = t_close_groups / total_groups if total_groups > 0 else 0
            
            results[f't={t}'] = {
                'achieved': t_closeness_ratio >= 0.8,  # Consider achieved if 80%+ groups are t-close
                't_close_groups': t_close_groups,
                'total_groups': total_groups,
                'ratio': t_closeness_ratio
            }
            
            print(f"  t={t}: {'✓' if results[f't={t}']['achieved'] else '✗'} "
                  f"({t_close_groups}/{total_groups} groups, {t_closeness_ratio:.2%})")
        
        return results
    
    def calculate_reidentification_risk(self, df, quasi_identifier_cols, original_df=None):
        """
        Calculate re-identification risk based on uniqueness of quasi-identifiers.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Anonymized dataset to evaluate
        quasi_identifier_cols : list
            List of quasi-identifier column names
        original_df : pandas.DataFrame, optional
            Original dataset for comparison
            
        Returns:
        --------
        dict
            Dictionary containing re-identification risk metrics
        """
        print("Calculating re-identification risk...")
        
        # Count unique combinations of quasi-identifiers
        unique_combinations = df[quasi_identifier_cols].drop_duplicates()
        total_records = len(df)
        
        # Calculate risk metrics
        risk_metrics = {
            'total_records': total_records,
            'unique_combinations': len(unique_combinations),
            'uniqueness_ratio': len(unique_combinations) / total_records,
            'risk_level': 'LOW'
        }
        
        # Determine risk level
        if risk_metrics['uniqueness_ratio'] > 0.8:
            risk_metrics['risk_level'] = 'HIGH'
        elif risk_metrics['uniqueness_ratio'] > 0.5:
            risk_metrics['risk_level'] = 'MEDIUM'
        
        print(f"  Re-identification risk: {risk_metrics['risk_level']}")
        print(f"  Unique combinations: {risk_metrics['unique_combinations']}/{total_records} "
              f"({risk_metrics['uniqueness_ratio']:.2%})")
        
        return risk_metrics

class UtilityValidator:
    """
    Class for validating data utility of anonymized data.
    """
    
    def __init__(self):
        """Initialize the utility validator."""
        pass
    
    def train_test_ml_models(self, original_df, synthetic_df, target_col, test_size=0.3):
        """
        Train ML models on both original and synthetic data to compare utility.
        
        Parameters:
        -----------
        original_df : pandas.DataFrame
            Original dataset
        synthetic_df : pandas.DataFrame
            Synthetic dataset
        target_col : str
            Target column for prediction
        test_size : float
            Proportion of data to use for testing
            
        Returns:
        --------
        dict
            Dictionary containing utility comparison results
        """
        print("Training ML models to compare utility...")
        
        if target_col not in original_df.columns or target_col not in synthetic_df.columns:
            print(f"Target column '{target_col}' not found in datasets")
            return None
        
        # Prepare features (exclude target and non-numerical columns)
        exclude_cols = [target_col] + [col for col in original_df.columns 
                                      if original_df[col].dtype == 'object']
        
        feature_cols = [col for col in original_df.columns if col not in exclude_cols]
        
        if not feature_cols:
            print("No suitable feature columns found")
            return None
        
        results = {}
        
        # Test with Random Forest
        results['random_forest'] = self._compare_model_performance(
            original_df, synthetic_df, feature_cols, target_col, 
            RandomForestClassifier(n_estimators=100, random_state=42),
            test_size, 'Random Forest'
        )
        
        # Test with Logistic Regression
        results['logistic_regression'] = self._compare_model_performance(
            original_df, synthetic_df, feature_cols, target_col,
            LogisticRegression(random_state=42, max_iter=1000),
            test_size, 'Logistic Regression'
        )
        
        return results
    
    def _compare_model_performance(self, original_df, synthetic_df, feature_cols, target_col, 
                                  model, test_size, model_name):
        """
        Compare model performance between original and synthetic data.
        
        Parameters:
        -----------
        original_df : pandas.DataFrame
            Original dataset
        synthetic_df : pandas.DataFrame
            Synthetic dataset
        feature_cols : list
            List of feature column names
        target_col : str
            Target column name
        model : sklearn estimator
            Model to train and test
        test_size : float
            Proportion of data to use for testing
        model_name : str
            Name of the model for reporting
            
        Returns:
        --------
        dict
            Dictionary containing performance comparison
        """
        print(f"  Testing {model_name}...")
        
        # Prepare original data
        X_orig = original_df[feature_cols].fillna(0)
        y_orig = original_df[target_col]
        
        # Split original data
        X_orig_train, X_orig_test, y_orig_train, y_orig_test = train_test_split(
            X_orig, y_orig, test_size=test_size, random_state=42
        )
        
        # Prepare synthetic data
        X_synth = synthetic_df[feature_cols].fillna(0)
        y_synth = synthetic_df[target_col]
        
        # Split synthetic data
        X_synth_train, X_synth_test, y_synth_train, y_synth_test = train_test_split(
            X_synth, y_synth, test_size=test_size, random_state=42
        )
        
        # Train on original data
        model_orig = model.__class__(**model.get_params())
        model_orig.fit(X_orig_train, y_orig_train)
        
        # Train on synthetic data
        model_synth = model.__class__(**model.get_params())
        model_synth.fit(X_synth_train, y_synth_train)
        
        # Test on original test data
        y_pred_orig = model_orig.predict(X_orig_test)
        y_pred_synth = model_synth.predict(X_orig_test)
        
        # Calculate metrics
        accuracy_orig = accuracy_score(y_orig_test, y_pred_orig)
        accuracy_synth = accuracy_score(y_orig_test, y_pred_synth)
        
        # Calculate AUC if binary classification
        auc_orig = None
        auc_synth = None
        if len(np.unique(y_orig)) == 2:
            try:
                auc_orig = roc_auc_score(y_orig_test, model_orig.predict_proba(X_orig_test)[:, 1])
                auc_synth = roc_auc_score(y_orig_test, model_synth.predict_proba(X_orig_test)[:, 1])
            except:
                pass
        
        results = {
            'model_name': model_name,
            'accuracy_original': accuracy_orig,
            'accuracy_synthetic': accuracy_synth,
            'accuracy_difference': accuracy_synth - accuracy_orig,
            'auc_original': auc_orig,
            'auc_synthetic': auc_synth,
            'utility_preserved': abs(accuracy_synth - accuracy_orig) < 0.1  # Consider preserved if difference < 10%
        }
        
        print(f"    Original accuracy: {accuracy_orig:.4f}")
        print(f"    Synthetic accuracy: {accuracy_synth:.4f}")
        print(f"    Difference: {results['accuracy_difference']:.4f}")
        
        return results
    
    def calculate_statistical_similarity(self, original_df, synthetic_df, numerical_cols=None):
        """
        Calculate statistical similarity between original and synthetic data.
        
        Parameters:
        -----------
        original_df : pandas.DataFrame
            Original dataset
        synthetic_df : pandas.DataFrame
            Synthetic dataset
        numerical_cols : list, optional
            List of numerical columns to compare
            
        Returns:
        --------
        dict
            Dictionary containing statistical similarity metrics
        """
        print("Calculating statistical similarity...")
        
        if numerical_cols is None:
            numerical_cols = [col for col in original_df.columns 
                             if original_df[col].dtype in [np.number]]
        
        similarity_metrics = {}
        
        for col in numerical_cols:
            if col in original_df.columns and col in synthetic_df.columns:
                # Calculate basic statistics
                orig_mean = original_df[col].mean()
                orig_std = original_df[col].std()
                synth_mean = synthetic_df[col].mean()
                synth_std = synthetic_df[col].std()
                
                # Calculate relative differences
                mean_diff = abs(synth_mean - orig_mean) / (abs(orig_mean) + 1e-8)
                std_diff = abs(synth_std - orig_std) / (abs(orig_std) + 1e-8)
                
                similarity_metrics[col] = {
                    'mean_original': orig_mean,
                    'mean_synthetic': synth_mean,
                    'mean_relative_diff': mean_diff,
                    'std_original': orig_std,
                    'std_synthetic': synth_std,
                    'std_relative_diff': std_diff,
                    'similarity_score': 1.0 - (mean_diff + std_diff) / 2
                }
        
        # Overall similarity score
        if similarity_metrics:
            overall_similarity = np.mean([metrics['similarity_score'] 
                                        for metrics in similarity_metrics.values()])
            similarity_metrics['overall'] = {
                'similarity_score': overall_similarity,
                'columns_compared': len(similarity_metrics)
            }
        
        print(f"  Overall similarity score: {overall_similarity:.4f}")
        return similarity_metrics

def validate_anonymization(original_file, anonymized_file, output_dir='results/validation'):
    """
    Complete validation pipeline for anonymized data.
    
    Parameters:
    -----------
    original_file : str
        Path to original dataset
    anonymized_file : str
        Path to anonymized dataset
    output_dir : str
        Directory to save validation results
        
    Returns:
    --------
    dict
        Dictionary containing all validation results
    """
    print("=== ANONYMIZATION VALIDATION PIPELINE ===")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load datasets
    print("Loading datasets...")
    original_df = pd.read_csv(original_file)
    anonymized_df = pd.read_csv(anonymized_file)
    
    print(f"Original dataset: {original_df.shape}")
    print(f"Anonymized dataset: {anonymized_df.shape}")
    
    # Initialize validators
    privacy_validator = PrivacyValidator()
    utility_validator = UtilityValidator()
    
    # Define field types for validation
    quasi_identifier_cols = ['AGE', 'ZIP', 'RACE', 'ETHNICITY', 'CITY', 'STATE']
    sensitive_cols = ['GENDER', 'HEALTHCARE_EXPENSES', 'HEALTHCARE_COVERAGE']
    
    # Ensure columns exist
    quasi_identifier_cols = [col for col in quasi_identifier_cols if col in anonymized_df.columns]
    sensitive_cols = [col for col in sensitive_cols if col in anonymized_df.columns]
    
    print(f"Quasi-identifiers: {quasi_identifier_cols}")
    print(f"Sensitive attributes: {sensitive_cols}")
    
    # Privacy validation
    print("\n--- PRIVACY VALIDATION ---")
    privacy_results = {}
    
    privacy_results['k_anonymity'] = privacy_validator.calculate_k_anonymity(
        anonymized_df, quasi_identifier_cols
    )
    
    privacy_results['l_diversity'] = privacy_validator.calculate_l_diversity(
        anonymized_df, quasi_identifier_cols, sensitive_cols
    )
    
    privacy_results['t_closeness'] = privacy_validator.calculate_t_closeness(
        anonymized_df, quasi_identifier_cols, sensitive_cols
    )
    
    privacy_results['reidentification_risk'] = privacy_validator.calculate_reidentification_risk(
        anonymized_df, quasi_identifier_cols
    )
    
    # Utility validation
    print("\n--- UTILITY VALIDATION ---")
    utility_results = {}
    
    # Find a suitable target column for ML testing
    target_candidates = ['GENDER', 'RACE', 'ETHNICITY']
    target_col = None
    for candidate in target_candidates:
        if candidate in anonymized_df.columns and candidate in original_df.columns:
            target_col = candidate
            break
    
    if target_col:
        utility_results['ml_performance'] = utility_validator.train_test_ml_models(
            original_df, anonymized_df, target_col
        )
    else:
        print("No suitable target column found for ML testing")
    
    utility_results['statistical_similarity'] = utility_validator.calculate_statistical_similarity(
        original_df, anonymized_df
    )
    
    # Combine results
    validation_results = {
        'privacy': privacy_results,
        'utility': utility_results,
        'dataset_info': {
            'original_shape': original_df.shape,
            'anonymized_shape': anonymized_df.shape,
            'quasi_identifiers': quasi_identifier_cols,
            'sensitive_attributes': sensitive_cols
        }
    }
    
    # Save results
    results_file = f"{output_dir}/validation_results.json"
    with open(results_file, 'w') as f:
        json.dump(validation_results, f, indent=2, default=str)
    
    # Generate summary report
    summary_file = f"{output_dir}/validation_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("ANONYMIZATION VALIDATION SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("PRIVACY METRICS:\n")
        f.write("-" * 20 + "\n")
        for metric, results in privacy_results.items():
            f.write(f"{metric}:\n")
            for key, value in results.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")
        
        f.write("UTILITY METRICS:\n")
        f.write("-" * 20 + "\n")
        for metric, results in utility_results.items():
            f.write(f"{metric}:\n")
            if isinstance(results, dict):
                for key, value in results.items():
                    f.write(f"  {key}: {value}\n")
            f.write("\n")
    
    print(f"\nValidation completed! Results saved to {output_dir}")
    return validation_results

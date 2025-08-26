import unittest
import pandas as pd
import numpy as np
import os
import sys

# Add the src directory to the path so we can import the modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.anonymization import k_anonymize, l_diversity, t_closeness, differential_privacy, anonymize_data


class TestAnonymization(unittest.TestCase):
    
    def setUp(self):
        """Set up test data."""
        # Create a sample DataFrame for testing
        np.random.seed(42)
        self.test_data = pd.DataFrame({
            'PatientID': range(1, 101),
            'Age': np.random.randint(21, 81, size=100),
            'Gender': np.random.choice(['Male', 'Female'], size=100),
            'Glucose': np.round(np.random.uniform(70, 200, size=100), 1),
            'BloodPressure': np.round(np.random.uniform(50, 120, size=100), 1),
            'Outcome': np.random.choice([0, 1], size=100)
        })
    
    def test_k_anonymize(self):
        """Test k-anonymity implementation."""
        # Apply k-anonymity
        sensitive_columns = ['Age', 'Gender']
        k = 5
        df_anonymized = k_anonymize(self.test_data, sensitive_columns, k)
        
        # Check that the anonymized DataFrame has the same number of rows
        self.assertEqual(len(df_anonymized), len(self.test_data))
        
        # Check that the anonymized DataFrame has the same columns
        self.assertListEqual(list(df_anonymized.columns), list(self.test_data.columns))
        
        # Check that the sensitive columns have been modified
        for col in sensitive_columns:
            if self.test_data[col].dtype in [np.float64, np.int64]:
                # For numerical columns, check that the number of unique values has decreased
                self.assertLess(df_anonymized[col].nunique(), self.test_data[col].nunique())
        
        # Check k-anonymity property: each combination of quasi-identifiers appears at least k times
        group_counts = df_anonymized.groupby(sensitive_columns).size()
        self.assertTrue((group_counts >= k).all() or len(group_counts) == 0)
    
    def test_l_diversity(self):
        """Test l-diversity implementation."""
        # Apply l-diversity
        sensitive_columns = ['Age', 'Gender']
        sensitive_attribute = 'Outcome'
        l = 2  # Using a small l for testing
        df_diverse = l_diversity(self.test_data, sensitive_columns, sensitive_attribute, l)
        
        # Check that the diverse DataFrame has rows
        self.assertGreater(len(df_diverse), 0)
        
        # Check that the diverse DataFrame has the same columns
        self.assertListEqual(list(df_diverse.columns), list(self.test_data.columns))
        
        # Check l-diversity property: each group has at least l distinct values of the sensitive attribute
        groups = df_diverse.groupby(sensitive_columns)
        for name, group in groups:
            distinct_values = group[sensitive_attribute].nunique()
            self.assertGreaterEqual(distinct_values, l)
    
    def test_t_closeness(self):
        """Test t-closeness implementation."""
        # Apply t-closeness
        sensitive_columns = ['Age', 'Gender']
        sensitive_attribute = 'Outcome'
        t = 0.5  # Using a large t for testing
        df_tclose = t_closeness(self.test_data, sensitive_columns, sensitive_attribute, t)
        
        # Check that the t-close DataFrame has rows
        self.assertGreater(len(df_tclose), 0)
        
        # Check that the t-close DataFrame has the same columns
        self.assertListEqual(list(df_tclose.columns), list(self.test_data.columns))
        
        # Full t-closeness verification would be complex, so we just check basic properties
        # Calculate global distribution
        global_dist = df_tclose[sensitive_attribute].value_counts(normalize=True)
        
        # Check that each group's distribution is not too far from the global distribution
        groups = df_tclose.groupby(sensitive_columns)
        for name, group in groups:
            group_dist = group[sensitive_attribute].value_counts(normalize=True)
            group_dist = group_dist.reindex(global_dist.index, fill_value=0)
            distance = sum(abs(global_dist - group_dist)) / 2
            self.assertLessEqual(distance, t)
    
    def test_differential_privacy(self):
        """Test differential privacy implementation."""
        # Apply differential privacy
        epsilon = 1.0
        df_private = differential_privacy(self.test_data, epsilon)
        
        # Check that the private DataFrame has the same shape
        self.assertEqual(df_private.shape, self.test_data.shape)
        
        # Check that the private DataFrame has the same columns
        self.assertListEqual(list(df_private.columns), list(self.test_data.columns))
        
        # Check that numerical columns have been modified
        for col in self.test_data.select_dtypes(include=[np.number]).columns:
            if col != 'PatientID':  # Skip ID column
                # Check that values have changed
                self.assertFalse(np.array_equal(df_private[col].values, self.test_data[col].values))
    
    def test_anonymize_data(self):
        """Test the anonymize_data function with different methods."""
        # Test k-anonymity method
        df_k = anonymize_data(self.test_data, method='k_anonymity', 
                             sensitive_columns=['Age', 'Gender'], k=5)
        self.assertEqual(len(df_k), len(self.test_data))
        
        # Test l-diversity method
        df_l = anonymize_data(self.test_data, method='l_diversity', 
                             sensitive_columns=['Age', 'Gender'], 
                             sensitive_attribute='Outcome', l=2)
        self.assertGreater(len(df_l), 0)
        
        # Test t-closeness method
        df_t = anonymize_data(self.test_data, method='t_closeness', 
                             sensitive_columns=['Age', 'Gender'], 
                             sensitive_attribute='Outcome', t=0.5)
        self.assertGreater(len(df_t), 0)
        
        # Test differential privacy method
        df_dp = anonymize_data(self.test_data, method='differential_privacy', epsilon=1.0)
        self.assertEqual(len(df_dp), len(self.test_data))
        
        # Test with invalid method
        with self.assertRaises(ValueError):
            anonymize_data(self.test_data, method='invalid_method')


if __name__ == '__main__':
    unittest.main()
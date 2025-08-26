import unittest
import pandas as pd
import numpy as np
import os
import sys

# Add the src directory to the path so we can import the modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.validation import calculate_information_loss, evaluate_utility_preservation, check_k_anonymity, check_l_diversity, validate_anonymization


class TestValidation(unittest.TestCase):
    
    def setUp(self):
        """Set up test data."""
        # Create sample DataFrames for testing
        np.random.seed(42)
        
        # Original data
        self.original_data = pd.DataFrame({
            'PatientID': range(1, 101),
            'Age': np.random.randint(21, 81, size=100),
            'Gender': np.random.choice(['Male', 'Female'], size=100),
            'Glucose': np.round(np.random.uniform(70, 200, size=100), 1),
            'BloodPressure': np.round(np.random.uniform(50, 120, size=100), 1),
            'Outcome': np.random.choice([0, 1], size=100)
        })
        
        # Anonymized data (simulated)
        self.anonymized_data = self.original_data.copy()
        # Modify Age (binning)
        self.anonymized_data['Age'] = pd.cut(self.anonymized_data['Age'], bins=5, labels=False)
        # Add some noise to numerical columns
        self.anonymized_data['Glucose'] = self.anonymized_data['Glucose'] + np.random.normal(0, 5, size=100)
        self.anonymized_data['BloodPressure'] = self.anonymized_data['BloodPressure'] + np.random.normal(0, 3, size=100)
    
    def test_calculate_information_loss(self):
        """Test calculation of information loss."""
        # Calculate information loss
        info_loss = calculate_information_loss(self.original_data, self.anonymized_data)
        
        # Check that the result is a dictionary with the expected keys
        self.assertIsInstance(info_loss, dict)
        self.assertIn('column_info_loss', info_loss)
        self.assertIn('average_info_loss', info_loss)
        
        # Check that the average information loss is a float between 0 and 1
        self.assertIsInstance(info_loss['average_info_loss'], float)
        self.assertGreaterEqual(info_loss['average_info_loss'], 0)
        self.assertLessEqual(info_loss['average_info_loss'], 1)
        
        # Check with specific numerical columns
        numerical_columns = ['Glucose', 'BloodPressure']
        info_loss_specific = calculate_information_loss(self.original_data, self.anonymized_data, numerical_columns)
        self.assertEqual(set(info_loss_specific['column_info_loss'].keys()), set(numerical_columns))
    
    def test_evaluate_utility_preservation(self):
        """Test evaluation of utility preservation."""
        # Evaluate utility preservation
        utility = evaluate_utility_preservation(self.original_data, self.anonymized_data, target_column='Outcome')
        
        # Check that the result is a dictionary with the expected keys
        self.assertIsInstance(utility, dict)
        self.assertIn('original_metrics', utility)
        self.assertIn('anonymized_metrics', utility)
        self.assertIn('utility_preservation', utility)
        
        # Check that the metrics are dictionaries with the expected keys
        for metrics in [utility['original_metrics'], utility['anonymized_metrics']]:
            self.assertIn('accuracy', metrics)
            self.assertIn('precision', metrics)
            self.assertIn('recall', metrics)
            self.assertIn('f1', metrics)
        
        # Check that the utility preservation values are percentages
        for metric, value in utility['utility_preservation'].items():
            self.assertIsInstance(value, float)
            # Allow for values > 100% as anonymized data could perform better in some cases
            self.assertGreaterEqual(value, 0)
    
    def test_check_k_anonymity(self):
        """Test checking of k-anonymity."""
        # Create a DataFrame that satisfies k-anonymity with k=2
        k_anon_data = pd.DataFrame({
            'Age': [30, 30, 40, 40, 50, 50],
            'Gender': ['Male', 'Male', 'Female', 'Female', 'Male', 'Male'],
            'Outcome': [0, 1, 0, 1, 0, 1]
        })
        
        # Check k-anonymity
        result = check_k_anonymity(k_anon_data, quasi_identifiers=['Age', 'Gender'], k=2)
        
        # Check that the result is a dictionary with the expected keys
        self.assertIsInstance(result, dict)
        self.assertIn('satisfies_k_anonymity', result)
        self.assertIn('violation_count', result)
        self.assertIn('min_group_size', result)
        self.assertIn('max_group_size', result)
        self.assertIn('avg_group_size', result)
        
        # Check that k-anonymity is satisfied
        self.assertTrue(result['satisfies_k_anonymity'])
        self.assertEqual(result['violation_count'], 0)
        self.assertEqual(result['min_group_size'], 2)
        
        # Test with k=3 (should not be satisfied)
        result_k3 = check_k_anonymity(k_anon_data, quasi_identifiers=['Age', 'Gender'], k=3)
        self.assertFalse(result_k3['satisfies_k_anonymity'])
        self.assertGreater(result_k3['violation_count'], 0)
    
    def test_check_l_diversity(self):
        """Test checking of l-diversity."""
        # Create a DataFrame that satisfies l-diversity with l=2
        l_div_data = pd.DataFrame({
            'Age': [30, 30, 40, 40, 50, 50],
            'Gender': ['Male', 'Male', 'Female', 'Female', 'Male', 'Male'],
            'Outcome': [0, 1, 0, 1, 0, 1]
        })
        
        # Check l-diversity
        result = check_l_diversity(l_div_data, quasi_identifiers=['Age', 'Gender'], 
                                  sensitive_attribute='Outcome', l=2)
        
        # Check that the result is a dictionary with the expected keys
        self.assertIsInstance(result, dict)
        self.assertIn('satisfies_l_diversity', result)
        self.assertIn('violation_count', result)
        self.assertIn('min_diversity', result)
        self.assertIn('max_diversity', result)
        self.assertIn('avg_diversity', result)
        
        # Check that l-diversity is satisfied
        self.assertTrue(result['satisfies_l_diversity'])
        self.assertEqual(result['violation_count'], 0)
        self.assertEqual(result['min_diversity'], 2)
        
        # Test with l=3 (should not be satisfied)
        result_l3 = check_l_diversity(l_div_data, quasi_identifiers=['Age', 'Gender'], 
                                     sensitive_attribute='Outcome', l=3)
        self.assertFalse(result_l3['satisfies_l_diversity'])
        self.assertGreater(result_l3['violation_count'], 0)
    
    def test_validate_anonymization(self):
        """Test the complete anonymization validation."""
        # Validate anonymization
        result = validate_anonymization(self.original_data, self.anonymized_data, 
                                       quasi_identifiers=['Age', 'Gender'], 
                                       sensitive_attribute='Outcome')
        
        # Check that the result is a dictionary with the expected keys
        self.assertIsInstance(result, dict)
        self.assertIn('k_anonymity', result)
        self.assertIn('l_diversity', result)
        self.assertIn('information_loss', result)
        self.assertIn('utility_preservation', result)
        
        # Check that each component has the expected structure
        self.assertIn('satisfies_k_anonymity', result['k_anonymity'])
        self.assertIn('satisfies_l_diversity', result['l_diversity'])
        self.assertIn('average_info_loss', result['information_loss'])
        self.assertIn('utility_preservation', result['utility_preservation'])


if __name__ == '__main__':
    unittest.main()
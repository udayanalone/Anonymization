import unittest
import pandas as pd
import numpy as np
import os
import sys

# Add the src directory to the path so we can import the modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.enhanced_anonymization import identify_sensitive_attributes, generate_synthetic_data_for_column, anonymize_dataset


class TestEnhancedAnonymization(unittest.TestCase):
    """Test cases for the enhanced anonymization module."""
    
    def setUp(self):
        """Set up test data."""
        # Create a test DataFrame with various sensitive attributes
        self.test_data = pd.DataFrame({
            'PatientID': range(1, 11),
            'Age': np.random.randint(20, 80, 10),
            'FullName': ['John Smith', 'Jane Doe', 'Robert Johnson', 'Emily Davis', 'Michael Brown',
                        'Sarah Wilson', 'David Miller', 'Jennifer Taylor', 'James Anderson', 'Lisa Thomas'],
            'Email': ['john.smith@example.com', 'jane.doe@example.com', 'robert.j@example.com',
                     'emily.davis@example.com', 'michael.b@example.com', 'sarah.w@example.com',
                     'david.m@example.com', 'jennifer.t@example.com', 'james.a@example.com', 'lisa.t@example.com'],
            'SSN': ['123-45-6789', '234-56-7890', '345-67-8901', '456-78-9012', '567-89-0123',
                   '678-90-1234', '789-01-2345', '890-12-3456', '901-23-4567', '012-34-5678'],
            'Phone': ['(123) 456-7890', '(234) 567-8901', '(345) 678-9012', '(456) 789-0123', '(567) 890-1234',
                     '(678) 901-2345', '(789) 012-3456', '(890) 123-4567', '(901) 234-5678', '(012) 345-6789'],
            'Address': ['123 Main Street, City, State', '456 Oak Avenue, Town, State', '789 Pine Road, Village, State',
                       '321 Elm Drive, Suburb, State', '654 Maple Boulevard, District, State',
                       '987 Cedar Lane, County, State', '246 Birch Path, Region, State',
                       '135 Willow Way, Area, State', '864 Spruce Court, Zone, State', '975 Fir Circle, Sector, State'],
            'BirthDate': ['1980-01-15', '1975-03-22', '1990-05-10', '1985-07-18', '1970-09-25',
                         '1995-11-05', '1982-02-12', '1978-04-30', '1992-06-20', '1988-08-08'],
            'CreditCard': ['1234-5678-9012-3456', '2345-6789-0123-4567', '3456-7890-1234-5678',
                          '4567-8901-2345-6789', '5678-9012-3456-7890', '6789-0123-4567-8901',
                          '7890-1234-5678-9012', '8901-2345-6789-0123', '9012-3456-7890-1234', '0123-4567-8901-2345'],
            'Outcome': np.random.choice([0, 1], 10)
        })
    
    def test_identify_sensitive_attributes(self):
        """Test the identify_sensitive_attributes function."""
        sensitive_attrs = identify_sensitive_attributes(self.test_data)
        
        # Check that all sensitive attributes are correctly identified
        self.assertIn('FullName', sensitive_attrs)
        self.assertEqual(sensitive_attrs['FullName'], 'name')
        
        self.assertIn('Email', sensitive_attrs)
        self.assertEqual(sensitive_attrs['Email'], 'email')
        
        self.assertIn('SSN', sensitive_attrs)
        self.assertEqual(sensitive_attrs['SSN'], 'ssn')
        
        self.assertIn('Phone', sensitive_attrs)
        self.assertEqual(sensitive_attrs['Phone'], 'phone')
        
        self.assertIn('Address', sensitive_attrs)
        self.assertEqual(sensitive_attrs['Address'], 'address')
        
        self.assertIn('BirthDate', sensitive_attrs)
        self.assertEqual(sensitive_attrs['BirthDate'], 'birthdate')
        
        self.assertIn('CreditCard', sensitive_attrs)
        self.assertEqual(sensitive_attrs['CreditCard'], 'credit_card')
        
        # Check that non-sensitive attributes are not identified
        self.assertNotIn('PatientID', sensitive_attrs)
        self.assertNotIn('Age', sensitive_attrs)
        self.assertNotIn('Outcome', sensitive_attrs)
    
    def test_generate_synthetic_data_for_column(self):
        """Test the generate_synthetic_data_for_column function."""
        # Test name generation
        synthetic_names = generate_synthetic_data_for_column(self.test_data, 'FullName', 'name')
        self.assertEqual(len(synthetic_names), len(self.test_data))
        self.assertNotEqual(set(synthetic_names), set(self.test_data['FullName']))
        
        # Test email generation
        synthetic_emails = generate_synthetic_data_for_column(self.test_data, 'Email', 'email')
        self.assertEqual(len(synthetic_emails), len(self.test_data))
        self.assertNotEqual(set(synthetic_emails), set(self.test_data['Email']))
        
        # Test SSN generation
        synthetic_ssns = generate_synthetic_data_for_column(self.test_data, 'SSN', 'ssn')
        self.assertEqual(len(synthetic_ssns), len(self.test_data))
        self.assertNotEqual(set(synthetic_ssns), set(self.test_data['SSN']))
        
        # Verify SSN format
        for ssn in synthetic_ssns:
            self.assertTrue(ssn.count('-') == 2)
            parts = ssn.split('-')
            self.assertEqual(len(parts), 3)
            self.assertEqual(len(parts[0]), 3)
            self.assertEqual(len(parts[1]), 2)
            self.assertEqual(len(parts[2]), 4)
    
    def test_anonymize_dataset(self):
        """Test the anonymize_dataset function."""
        # Define sensitive attributes
        sensitive_attrs = {
            'FullName': 'name',
            'Email': 'email',
            'SSN': 'ssn',
            'Phone': 'phone',
            'Address': 'address',
            'BirthDate': 'birthdate',
            'CreditCard': 'credit_card'
        }
        
        # Anonymize the dataset
        anonymized_df = anonymize_dataset(self.test_data, sensitive_attrs)
        
        # Check that the anonymized dataset has the same shape
        self.assertEqual(anonymized_df.shape, self.test_data.shape)
        
        # Check that sensitive attributes have been replaced
        for col, attr_type in sensitive_attrs.items():
            self.assertNotEqual(set(anonymized_df[col]), set(self.test_data[col]))
        
        # Check that non-sensitive attributes remain unchanged
        pd.testing.assert_series_equal(anonymized_df['PatientID'], self.test_data['PatientID'])
        pd.testing.assert_series_equal(anonymized_df['Age'], self.test_data['Age'])
        pd.testing.assert_series_equal(anonymized_df['Outcome'], self.test_data['Outcome'])


if __name__ == '__main__':
    unittest.main()
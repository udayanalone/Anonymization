import unittest
import pandas as pd
import numpy as np
import os
import sys
import tempfile

# Add the src directory to the path so we can import the modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_processing import load_data, clean_data, preprocess_data, save_processed_data


class TestDataProcessing(unittest.TestCase):
    
    def setUp(self):
        """Set up test data."""
        # Create a sample DataFrame for testing
        self.test_data = pd.DataFrame({
            'PatientID': range(1, 11),
            'Age': [25, 35, 45, 55, 65, 75, np.nan, 40, 50, 60],
            'Gender': ['Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', np.nan],
            'Glucose': [100, 120, 140, 160, 180, 200, 90, np.nan, 130, 150],
            'BloodPressure': [70, 80, 90, 100, 110, 120, 65, 75, np.nan, 95],
            'Outcome': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        })
        
        # Create a temporary file for testing load_data and save_processed_data
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as temp_file:
            self.temp_path = temp_file.name
            self.test_data.to_csv(self.temp_path, index=False)
    
    def tearDown(self):
        """Clean up after tests."""
        # Remove temporary file
        if os.path.exists(self.temp_path):
            os.remove(self.temp_path)
    
    def test_load_data(self):
        """Test loading data from a CSV file."""
        # Load data from the temporary file
        df = load_data(self.temp_path)
        
        # Check that the loaded DataFrame has the correct shape and columns
        self.assertEqual(len(df), len(self.test_data))
        self.assertListEqual(list(df.columns), list(self.test_data.columns))
        
        # Test loading non-existent file
        with self.assertRaises(FileNotFoundError):
            load_data('non_existent_file.csv')
    
    def test_clean_data(self):
        """Test cleaning data by handling missing values and outliers."""
        # Clean the test data
        df_clean = clean_data(self.test_data)
        
        # Check that the cleaned DataFrame has the same shape as the original
        self.assertEqual(len(df_clean), len(self.test_data))
        self.assertListEqual(list(df_clean.columns), list(self.test_data.columns))
        
        # Check that there are no missing values in the cleaned DataFrame
        self.assertEqual(df_clean.isna().sum().sum(), 0)
        
        # Check that the missing values were replaced with appropriate values
        self.assertFalse(np.isnan(df_clean.loc[6, 'Age']))
        self.assertFalse(np.isnan(df_clean.loc[9, 'Gender']))
        self.assertFalse(np.isnan(df_clean.loc[7, 'Glucose']))
        self.assertFalse(np.isnan(df_clean.loc[8, 'BloodPressure']))
    
    def test_preprocess_data(self):
        """Test preprocessing data for analysis and modeling."""
        # Clean the test data first to handle missing values
        df_clean = clean_data(self.test_data)
        
        # Preprocess the cleaned data
        df_processed = preprocess_data(df_clean)
        
        # Check that the processed DataFrame has the correct shape
        self.assertEqual(len(df_processed), len(self.test_data))
        
        # Check that categorical variables were one-hot encoded
        self.assertIn('Gender_Male', df_processed.columns)
        
        # Check that numerical features were normalized
        for col in ['Age', 'Glucose', 'BloodPressure']:
            self.assertAlmostEqual(df_processed[col].mean(), 0, places=10)
            self.assertAlmostEqual(df_processed[col].std(), 1, places=10)
    
    def test_save_processed_data(self):
        """Test saving processed data to a CSV file."""
        # Create a temporary file for saving processed data
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as temp_file:
            output_path = temp_file.name
        
        try:
            # Save the test data to the temporary file
            save_processed_data(self.test_data, output_path)
            
            # Check that the file exists
            self.assertTrue(os.path.exists(output_path))
            
            # Check that the file contains the correct data
            df_loaded = pd.read_csv(output_path)
            pd.testing.assert_frame_equal(df_loaded, self.test_data)
        finally:
            # Clean up
            if os.path.exists(output_path):
                os.remove(output_path)
    
    def test_end_to_end_processing(self):
        """Test the entire data processing pipeline."""
        # Create a temporary file for saving processed data
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as temp_file:
            output_path = temp_file.name
        
        try:
            # Load, clean, preprocess, and save data
            df = load_data(self.temp_path)
            df_clean = clean_data(df)
            df_processed = preprocess_data(df_clean)
            save_processed_data(df_processed, output_path)
            
            # Check that the file exists
            self.assertTrue(os.path.exists(output_path))
            
            # Check that the file contains the correct data
            df_loaded = pd.read_csv(output_path)
            pd.testing.assert_frame_equal(df_loaded, df_processed)
        finally:
            # Clean up
            if os.path.exists(output_path):
                os.remove(output_path)


if __name__ == '__main__':
    unittest.main()
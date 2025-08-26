import unittest
import pandas as pd
import numpy as np
import os
import sys
import tempfile

# Add the src directory to the path so we can import the modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_generation import generate_synthetic_data


class TestDataGeneration(unittest.TestCase):
    
    def test_generate_synthetic_data_shape(self):
        """Test that the generated data has the correct shape."""
        n_records = 100
        df = generate_synthetic_data(n_records=n_records, seed=42)
        
        # Check that the DataFrame has the correct number of rows
        self.assertEqual(len(df), n_records)
        
        # Check that the DataFrame has the correct columns
        expected_columns = [
            'PatientID', 'Age', 'Gender', 'Pregnancies', 'Glucose', 
            'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 
            'DiabetesPedigreeFunction', 'Outcome'
        ]
        self.assertListEqual(list(df.columns), expected_columns)
    
    def test_generate_synthetic_data_types(self):
        """Test that the generated data has the correct data types."""
        df = generate_synthetic_data(n_records=50, seed=42)
        
        # Check data types
        self.assertTrue(pd.api.types.is_integer_dtype(df['PatientID']))
        self.assertTrue(pd.api.types.is_integer_dtype(df['Age']))
        self.assertTrue(pd.api.types.is_object_dtype(df['Gender']))
        self.assertTrue(pd.api.types.is_integer_dtype(df['Pregnancies']))
        self.assertTrue(pd.api.types.is_float_dtype(df['Glucose']))
        self.assertTrue(pd.api.types.is_float_dtype(df['BloodPressure']))
        self.assertTrue(pd.api.types.is_float_dtype(df['SkinThickness']))
        self.assertTrue(pd.api.types.is_float_dtype(df['Insulin']))
        self.assertTrue(pd.api.types.is_float_dtype(df['BMI']))
        self.assertTrue(pd.api.types.is_float_dtype(df['DiabetesPedigreeFunction']))
        self.assertTrue(pd.api.types.is_integer_dtype(df['Outcome']))
    
    def test_generate_synthetic_data_values(self):
        """Test that the generated data has values within expected ranges."""
        df = generate_synthetic_data(n_records=200, seed=42)
        
        # Check value ranges
        self.assertTrue(all(df['PatientID'] >= 1))
        self.assertTrue(all(df['Age'] >= 21) and all(df['Age'] <= 80))
        self.assertTrue(all(df['Gender'].isin(['Male', 'Female'])))
        self.assertTrue(all(df['Pregnancies'] >= 0) and all(df['Pregnancies'] <= 15))
        self.assertTrue(all(df['Glucose'] >= 70) and all(df['Glucose'] <= 200))
        self.assertTrue(all(df['BloodPressure'] >= 50) and all(df['BloodPressure'] <= 120))
        self.assertTrue(all(df['SkinThickness'] >= 10) and all(df['SkinThickness'] <= 50))
        self.assertTrue(all(df['Insulin'] >= 15) and all(df['Insulin'] <= 276))
        self.assertTrue(all(df['BMI'] >= 18) and all(df['BMI'] <= 45))
        self.assertTrue(all(df['DiabetesPedigreeFunction'] >= 0.1) and all(df['DiabetesPedigreeFunction'] <= 2.5))
        self.assertTrue(all(df['Outcome'].isin([0, 1])))
    
    def test_generate_synthetic_data_reproducibility(self):
        """Test that the generated data is reproducible with the same seed."""
        df1 = generate_synthetic_data(n_records=100, seed=42)
        df2 = generate_synthetic_data(n_records=100, seed=42)
        
        # Check that the DataFrames are identical
        pd.testing.assert_frame_equal(df1, df2)
    
    def test_generate_synthetic_data_different_seeds(self):
        """Test that different seeds produce different data."""
        df1 = generate_synthetic_data(n_records=100, seed=42)
        df2 = generate_synthetic_data(n_records=100, seed=43)
        
        # Check that the DataFrames are different
        self.assertFalse(df1.equals(df2))
    
    def test_generate_synthetic_data_save_to_file(self):
        """Test that the generated data can be saved to a file."""
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            # Generate data and save to file
            df = generate_synthetic_data(n_records=50, seed=42, output_path=temp_path)
            
            # Check that the file exists
            self.assertTrue(os.path.exists(temp_path))
            
            # Check that the file contains the correct data
            df_loaded = pd.read_csv(temp_path)
            pd.testing.assert_frame_equal(df, df_loaded)
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)


if __name__ == '__main__':
    unittest.main()
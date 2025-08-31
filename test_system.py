#!/usr/bin/env python3
"""
Test script for the Data Anonymization System.

This script tests the core components without running the full pipeline.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

def test_imports():
    """Test if all modules can be imported."""
    print("Testing module imports...")
    
    try:
        from preprocessing import load_patient_data, identify_field_types
        print("✓ Preprocessing module imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import preprocessing module: {e}")
        return False
    
    try:
        from pseudonymization import Pseudonymizer
        print("✓ Pseudonymization module imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import pseudonymization module: {e}")
        return False
    
    try:
        from gan_model import ConditionalGAN
        print("✓ GAN model module imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import GAN model module: {e}")
        return False
    
    try:
        from validation import PrivacyValidator, UtilityValidator
        print("✓ Validation module imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import validation module: {e}")
        return False
    
    return True

def test_data_loading():
    """Test data loading functionality."""
    print("\nTesting data loading...")
    
    try:
        from preprocessing import load_patient_data
        
        # Test with a small number of records
        df = load_patient_data('data/raw/patient.csv', n_records=100)
        
        if df is not None and len(df) > 0:
            print(f"✓ Data loaded successfully: {df.shape}")
            print(f"  Columns: {list(df.columns)}")
            return True
        else:
            print("✗ Data loading failed")
            return False
            
    except Exception as e:
        print(f"✗ Data loading error: {e}")
        return False

def test_field_identification():
    """Test field type identification."""
    print("\nTesting field identification...")
    
    try:
        from preprocessing import identify_field_types
        
        # Load a small dataset
        df = load_patient_data('data/raw/patient.csv', n_records=100)
        if df is None:
            return False
        
        field_types = identify_field_types(df)
        
        print("✓ Field types identified:")
        for field_type, fields in field_types.items():
            if fields:
                print(f"  {field_type}: {len(fields)} fields")
        
        return True
        
    except Exception as e:
        print(f"✗ Field identification error: {e}")
        return False

def test_pseudonymization():
    """Test pseudonymization functionality."""
    print("\nTesting pseudonymization...")
    
    try:
        from pseudonymization import Pseudonymizer
        
        # Create test data
        test_data = {
            'Id': ['patient_001', 'patient_002'],
            'SSN': ['123-45-6789', '987-65-4321'],
            'FIRST': ['John', 'Jane'],
            'LAST': ['Doe', 'Smith'],
            'ADDRESS': ['123 Main St', '456 Oak Ave'],
            'CITY': ['Boston', 'New York'],
            'STATE': ['MA', 'NY'],
            'ZIP': ['02101', '10001']
        }
        
        df = pd.DataFrame(test_data)
        
        # Test pseudonymizer
        pseudonymizer = Pseudonymizer()
        df_pseudo = pseudonymizer.pseudonymize_dataset(df)
        
        print("✓ Pseudonymization completed successfully")
        print(f"  Original IDs: {list(df['Id'])}")
        print(f"  Pseudonymized IDs: {list(df_pseudo['Id'])}")
        
        return True
        
    except Exception as e:
        print(f"✗ Pseudonymization error: {e}")
        return False

def test_cgan_model():
    """Test CGAN model creation."""
    print("\nTesting CGAN model...")
    
    try:
        from gan_model import ConditionalGAN
        
        # Create CGAN instance
        cgan = ConditionalGAN(noise_dim=50)
        
        # Test model building
        cgan.build_models(condition_dim=3, target_dim=5)
        
        print("✓ CGAN model created successfully")
        print(f"  Generator: {cgan.generator}")
        print(f"  Discriminator: {cgan.discriminator}")
        
        return True
        
    except Exception as e:
        print(f"✗ CGAN model error: {e}")
        return False

def test_validation():
    """Test validation functionality."""
    print("\nTesting validation...")
    
    try:
        from validation import PrivacyValidator, UtilityValidator
        
        # Create validators
        privacy_validator = PrivacyValidator()
        utility_validator = UtilityValidator()
        
        print("✓ Validation modules created successfully")
        
        # Test with sample data
        test_df = pd.DataFrame({
            'AGE': [25, 30, 25, 35, 30, 25],
            'ZIP': ['02101', '02101', '02101', '10001', '10001', '10001'],
            'GENDER': ['M', 'F', 'M', 'F', 'M', 'F']
        })
        
        # Test k-anonymity
        k_results = privacy_validator.calculate_k_anonymity(test_df, ['AGE', 'ZIP'])
        print(f"  K-anonymity test completed: {len(k_results)} results")
        
        return True
        
    except Exception as e:
        print(f"✗ Validation error: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 50)
    print("DATA ANONYMIZATION SYSTEM - COMPONENT TESTS")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_data_loading,
        test_field_identification,
        test_pseudonymization,
        test_cgan_model,
        test_validation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    print("=" * 50)
    
    if passed == total:
        print("🎉 All tests passed! The system is ready to use.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

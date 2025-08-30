import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import KBinsDiscretizer
import re
import random
import string
import pickle
import importlib.util

# Check if CGAN module exists, otherwise fallback to Faker
try:
    from cgan_model import CGAN, generate_synthetic_data_with_cgan
    has_cgan = True
except ImportError:
    from faker import Faker
    fake = Faker()
    has_cgan = False

# Dictionary to store trained CGAN models
cgan_models = {}


def identify_sensitive_attributes(df):
    sensitive_attrs = {}
    
    for col in df.columns:
        # Convert column to string for pattern matching
        sample_values = df[col].astype(str).dropna().tolist()[:100]  # Check first 100 non-null values
        
        # Skip if no samples
        if not sample_values:
            continue
            
        # Check for names (simple heuristic: capitalized words without numbers)
        if any(re.match(r'^[A-Z][a-z]+ [A-Z][a-z]+$', str(val)) for val in sample_values):
            sensitive_attrs[col] = 'name'
            
        # Check for SSNs (format: XXX-XX-XXXX or XXXXXXXXX)
        elif any(re.match(r'^\d{3}-\d{2}-\d{4}$', str(val)) or re.match(r'^\d{9}$', str(val)) for val in sample_values):
            sensitive_attrs[col] = 'ssn'
            
        # Check for emails
        elif any(re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', str(val)) for val in sample_values):
            sensitive_attrs[col] = 'email'
            
        # Check for phone numbers
        elif any(re.match(r'^\(\d{3}\)\s?\d{3}-\d{4}$', str(val)) or 
                re.match(r'^\d{3}-\d{3}-\d{4}$', str(val)) or 
                re.match(r'^\d{10}$', str(val)) for val in sample_values):
            sensitive_attrs[col] = 'phone'
            
        # Check for addresses (simple heuristic)
        elif any('street' in str(val).lower() or 'avenue' in str(val).lower() or 
                'road' in str(val).lower() or 'drive' in str(val).lower() for val in sample_values):
            sensitive_attrs[col] = 'address'
            
        # Check for birthdates
        elif any(re.match(r'^\d{1,2}/\d{1,2}/\d{2,4}$', str(val)) or 
                re.match(r'^\d{4}-\d{2}-\d{2}$', str(val)) for val in sample_values):
            sensitive_attrs[col] = 'birthdate'
            
        # Check for credit card numbers
        elif any(re.match(r'^\d{4}-\d{4}-\d{4}-\d{4}$', str(val)) or 
                re.match(r'^\d{16}$', str(val)) for val in sample_values):
            sensitive_attrs[col] = 'credit_card'
    
    return sensitive_attrs


def load_or_train_cgan_model(df, column, attr_type, condition_column=None):
    """
    Load a pre-trained CGAN model or train a new one if not available.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the column
    column : str
        Name of the column to generate data for
    attr_type : str
        Type of the attribute ('name', 'ssn', 'email', etc.)
    condition_column : str, optional
        Name of the column to use as condition for generation
        
    Returns:
    --------
    CGAN or None
        Trained CGAN model or None if not available
    """
    global cgan_models
    
    # If CGAN is not available, return None
    if not has_cgan:
        return None
    
    # Check if model is already loaded
    if column in cgan_models:
        return cgan_models[column]
    
    # Try to load pre-trained model
    model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
    model_path = os.path.join(model_dir, f"{column}_model.pkl")
    
    if os.path.exists(model_path):
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            cgan_models[column] = model
            print(f"Loaded pre-trained CGAN model for {column} from {model_path}")
            return model
        except Exception as e:
            print(f"Error loading CGAN model for {column}: {e}")
    
    # If no pre-trained model, train a new one if we have enough data
    if len(df) >= 100:  # Only train if we have enough data
        try:
            from cgan_model import train_cgan_model
            
            # Determine conditions
            if condition_column is not None and condition_column in df.columns:
                conditions = df[condition_column]
            else:
                # Find a suitable condition column (categorical with few unique values)
                potential_conditions = []
                for col in df.columns:
                    if col != column and df[col].nunique() <= 10 and df[col].nunique() > 1:
                        potential_conditions.append(col)
                
                if potential_conditions:
                    # Use the column with the fewest unique values as condition
                    condition_column = min(potential_conditions, key=lambda c: df[c].nunique())
                    conditions = df[condition_column]
                    print(f"Using {condition_column} as condition for {column}")
                else:
                    # No suitable condition column found, use a dummy condition
                    conditions = pd.Series(['default'] * len(df))
            
            # Train the model
            print(f"Training new CGAN model for {column} (type: {attr_type})...")
            model = train_cgan_model(df[column], conditions, attr_type, epochs=200)
            
            # Save the model
            os.makedirs(model_dir, exist_ok=True)
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            cgan_models[column] = model
            print(f"Trained and saved new CGAN model for {column}")
            return model
        except Exception as e:
            print(f"Error training CGAN model for {column}: {e}")
    
    return None


def generate_synthetic_data_for_column(df, column, attr_type, preserve_distribution=True):
    """
    Generate synthetic data for a specific column based on its attribute type.
    Uses CGAN if available, otherwise falls back to Faker.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the column
    column : str
        Name of the column to anonymize
    attr_type : str
        Type of the attribute ('name', 'ssn', 'email', etc.)
    preserve_distribution : bool, default=True
        Whether to preserve the statistical distribution of the data
        
    Returns:
    --------
    pandas.Series
        Series with synthetic data for the specified column
    """
    n_records = len(df)
    
    # Try to use CGAN model if available
    if has_cgan:
        # Find a suitable condition column
        condition_column = None
        for col in df.columns:
            if col != column and df[col].nunique() <= 10 and df[col].nunique() > 1:
                condition_column = col
                break
        
        # Load or train CGAN model
        model = load_or_train_cgan_model(df, column, attr_type, condition_column)
        
        if model is not None:
            try:
                # Get conditions for generation
                if condition_column is not None:
                    conditions = df[condition_column].tolist()
                else:
                    conditions = ['default'] * n_records
                
                # Generate synthetic data
                synthetic_data = generate_synthetic_data_with_cgan(model, conditions)
                print(f"Generated synthetic data for {column} using CGAN model")
                return synthetic_data
            except Exception as e:
                print(f"Error generating synthetic data with CGAN for {column}: {e}")
                print("Falling back to traditional methods...")
    
    # Fall back to Faker if CGAN is not available or failed
    if not has_cgan:
        if attr_type == 'name':
            return pd.Series([fake.name() for _ in range(n_records)])
            
        elif attr_type == 'ssn':
            return pd.Series([f"{random.randint(100, 999)}-{random.randint(10, 99)}-{random.randint(1000, 9999)}" for _ in range(n_records)])
            
        elif attr_type == 'email':
            return pd.Series([fake.email() for _ in range(n_records)])
            
        elif attr_type == 'phone':
            return pd.Series([fake.phone_number() for _ in range(n_records)])
            
        elif attr_type == 'address':
            return pd.Series([fake.address().replace('\n', ', ') for _ in range(n_records)])
            
        elif attr_type == 'birthdate':
            return pd.Series([fake.date_of_birth(minimum_age=18, maximum_age=90).strftime('%Y-%m-%d') for _ in range(n_records)])
            
        elif attr_type == 'credit_card':
            return pd.Series([fake.credit_card_number(card_type=None) for _ in range(n_records)])
    else:
        # Simple fallback methods if CGAN failed but module is available
        if attr_type == 'name':
            return pd.Series([f"Person-{i}" for i in range(n_records)])
            
        elif attr_type == 'ssn':
            return pd.Series([f"{random.randint(100, 999)}-{random.randint(10, 99)}-{random.randint(1000, 9999)}" for _ in range(n_records)])
            
        elif attr_type == 'email':
            return pd.Series([f"user{i}@example.com" for i in range(n_records)])
            
        elif attr_type == 'phone':
            return pd.Series([f"({random.randint(100, 999)}) {random.randint(100, 999)}-{random.randint(1000, 9999)}" for _ in range(n_records)])
            
        elif attr_type == 'address':
            return pd.Series([f"{random.randint(1, 9999)} Main St, City, State {random.randint(10000, 99999)}" for _ in range(n_records)])
            
        elif attr_type == 'birthdate':
            return pd.Series([f"{random.randint(1950, 2000)}-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}" for _ in range(n_records)])
            
        elif attr_type == 'credit_card':
            return pd.Series([f"{random.randint(1000, 9999)}-{random.randint(1000, 9999)}-{random.randint(1000, 9999)}-{random.randint(1000, 9999)}" for _ in range(n_records)])
    
    # For unrecognized types, return the original column
    return df[column]


def anonymize_dataset(df, sensitive_attributes=None):
    """
    Anonymize a dataset by replacing sensitive attributes with synthetic data.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame to anonymize
    sensitive_attributes : dict, optional
        Dictionary with column names as keys and their identified type as values.
        If None, will attempt to automatically identify sensitive attributes.
        
    Returns:
    --------
    pandas.DataFrame
        Anonymized DataFrame
    """
    # Create a copy to avoid modifying the original DataFrame
    df_anonymized = df.copy()
    
    # Identify sensitive attributes if not provided
    if sensitive_attributes is None:
        sensitive_attributes = identify_sensitive_attributes(df)
    
    # Replace each sensitive attribute with synthetic data
    for column, attr_type in sensitive_attributes.items():
        if column in df_anonymized.columns:
            df_anonymized[column] = generate_synthetic_data_for_column(df, column, attr_type)
    
    return df_anonymized


def main():
    """
    Main function to demonstrate the enhanced anonymization process.
    """
    # Default paths
    input_path = os.path.join('..', 'data', 'raw', 'synthetic_data.csv')
    output_path = os.path.join('..', 'data', 'processed', 'enhanced_anonymized_data.csv')
    
    # Load data
    df = pd.read_csv(input_path)
    
    # For demonstration, let's add some synthetic sensitive attributes
    if not has_cgan:
        # Use Faker if CGAN is not available
        df['FullName'] = [fake.name() for _ in range(len(df))]
        df['Email'] = [fake.email() for _ in range(len(df))]
        df['SSN'] = [f"{random.randint(100, 999)}-{random.randint(10, 99)}-{random.randint(1000, 9999)}" for _ in range(len(df))]
        df['Phone'] = [fake.phone_number() for _ in range(len(df))]
        df['Address'] = [fake.address().replace('\n', ', ') for _ in range(len(df))]
        df['BirthDate'] = [fake.date_of_birth(minimum_age=18, maximum_age=90).strftime('%Y-%m-%d') for _ in range(len(df))]
        df['CreditCard'] = [fake.credit_card_number(card_type=None) for _ in range(len(df))]
    else:
        # Use simple placeholders that will be replaced by CGAN
        df['FullName'] = [f"Person-{i}" for i in range(len(df))]
        df['Email'] = [f"user{i}@example.com" for i in range(len(df))]
        df['SSN'] = [f"{random.randint(100, 999)}-{random.randint(10, 99)}-{random.randint(1000, 9999)}" for i in range(len(df))]
        df['Phone'] = [f"({random.randint(100, 999)}) {random.randint(100, 999)}-{random.randint(1000, 9999)}" for i in range(len(df))]
        df['Address'] = [f"{random.randint(1, 9999)} Main St, City, State {random.randint(10000, 99999)}" for i in range(len(df))]
        df['BirthDate'] = [f"{random.randint(1950, 2000)}-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}" for i in range(len(df))]
        df['CreditCard'] = [f"{random.randint(1000, 9999)}-{random.randint(1000, 9999)}-{random.randint(1000, 9999)}-{random.randint(1000, 9999)}" for i in range(len(df))]
    
    # Add a condition column for demonstration
    df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 30, 50, 100], labels=['Young', 'Middle', 'Senior'])
    
    # Identify sensitive attributes
    sensitive_attrs = identify_sensitive_attributes(df)
    print(f"Identified sensitive attributes: {sensitive_attrs}")
    
    # Anonymize data
    df_anonymized = anonymize_dataset(df, sensitive_attrs)
    
    # Save anonymized data
    df_anonymized.to_csv(output_path, index=False)
    print(f"Enhanced anonymized data saved successfully to {output_path}!")
    
    # If CGAN is available, show which models were used
    if has_cgan and cgan_models:
        print("\nCGAN models used for anonymization:")
        for column, model in cgan_models.items():
            print(f"- {column}: {model.attribute_type} model")


if __name__ == "__main__":
    main()
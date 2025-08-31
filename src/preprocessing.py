import pandas as pd
import numpy as np
import os
from pathlib import Path
import warnings
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib
import json
warnings.filterwarnings('ignore')

def load_patient_data(file_path, n_records=2000):
    """
    Load patient data from CSV file, limiting to specified number of records.
    
    Parameters:
    -----------
    file_path : str
        Path to the patient CSV file
    n_records : int, default=2000
        Number of records to load
        
    Returns:
    --------
    pandas.DataFrame
        Loaded patient data
    """
    try:
        print(f"Loading patient data from {file_path}...")
        print(f"Limiting to first {n_records} records...")
        
        # Load data with chunking to handle large files efficiently
        chunk_size = min(1000, n_records)
        chunks = []
        total_loaded = 0
        
        for chunk in pd.read_csv(file_path, chunksize=chunk_size):
            if total_loaded >= n_records:
                break
            chunks.append(chunk)
            total_loaded += len(chunk)
        
        # Combine chunks and limit to exact number requested
        df = pd.concat(chunks, ignore_index=True)
        df = df.head(n_records)
        
        print(f"Successfully loaded {len(df)} records")
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {', '.join(df.columns)}")
        
        return df
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def analyze_data_structure(df):
    """
    Analyze the structure of the patient dataset.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Patient dataset
        
    Returns:
    --------
    dict
        Dictionary containing data structure information
    """
    print("\n=== DATA STRUCTURE ANALYSIS ===")
    
    # Basic info
    info = {
        'total_records': len(df),
        'total_columns': len(df.columns),
        'memory_usage': df.memory_usage(deep=True).sum() / 1024 / 1024,  # MB
        'missing_values': df.isnull().sum().sum(),
        'duplicate_records': df.duplicated().sum()
    }
    
    print(f"Total records: {info['total_records']:,}")
    print(f"Total columns: {info['total_columns']}")
    print(f"Memory usage: {info['memory_usage']:.2f} MB")
    print(f"Missing values: {info['missing_values']:,}")
    print(f"Duplicate records: {info['duplicate_records']:,}")
    
    # Column types
    print(f"\nColumn types:")
    print(df.dtypes.value_counts())
    
    # Missing values per column
    missing_cols = df.columns[df.isnull().any()].tolist()
    if missing_cols:
        print(f"\nColumns with missing values:")
        for col in missing_cols:
            missing_count = df[col].isnull().sum()
            missing_pct = (missing_count / len(df)) * 100
            print(f"  {col}: {missing_count:,} ({missing_pct:.1f}%)")
    else:
        print("\nNo missing values found in any column.")
    
    return info

def identify_column_categories(df):
    """
    Identify different categories of columns in the dataset.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Patient dataset
        
    Returns:
    --------
    dict
        Dictionary mapping column categories to column names
    """
    print("\n=== COLUMN CATEGORIZATION ===")
    
    # Identify different types of columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    # Further categorize based on content
    identifier_cols = []
    sensitive_cols = []
    clinical_cols = []
    demographic_cols = []
    
    for col in df.columns:
        col_lower = col.lower()
        
        # Identifiers
        if any(keyword in col_lower for keyword in ['id', 'uuid', 'code']):
            identifier_cols.append(col)
        # Sensitive personal info
        elif any(keyword in col_lower for keyword in ['name', 'address', 'phone', 'email', 'ssn', 'birth']):
            sensitive_cols.append(col)
        # Clinical/medical data
        elif any(keyword in col_lower for keyword in ['diagnosis', 'condition', 'medication', 'procedure', 'test', 'result']):
            clinical_cols.append(col)
        # Demographics
        elif any(keyword in col_lower for keyword in ['age', 'gender', 'race', 'ethnicity', 'marital']):
            demographic_cols.append(col)
    
    # Print categorization
    print(f"Numerical columns ({len(numerical_cols)}): {', '.join(numerical_cols)}")
    print(f"Categorical columns ({len(categorical_cols)}): {', '.join(categorical_cols)}")
    print(f"Datetime columns ({len(datetime_cols)}): {', '.join(datetime_cols)}")
    print(f"Identifier columns ({len(identifier_cols)}): {', '.join(identifier_cols)}")
    print(f"Sensitive columns ({len(sensitive_cols)}): {', '.join(sensitive_cols)}")
    print(f"Clinical columns ({len(clinical_cols)}): {', '.join(clinical_cols)}")
    print(f"Demographic columns ({len(demographic_cols)}): {', '.join(demographic_cols)}")
    
    return {
        'numerical': numerical_cols,
        'categorical': categorical_cols,
        'datetime': datetime_cols,
        'identifiers': identifier_cols,
        'sensitive': sensitive_cols,
        'clinical': clinical_cols,
        'demographic': demographic_cols
    }

def identify_field_types(df):
    """
    Identify different types of fields in the dataset for anonymization strategy.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Patient dataset
        
    Returns:
    --------
    dict
        Dictionary categorizing fields by type
    """
    print("\n=== FIELD TYPE IDENTIFICATION ===")
    
    field_types = {
        'identifiers': [],      # Direct identifiers to pseudonymize
        'quasi_identifiers': [], # Quasi-identifiers for synthetic generation
        'critical_features': [], # Critical features to preserve
        'sensitive_attributes': [], # Other sensitive attributes
        'numerical_features': [],  # Numerical features
        'categorical_features': [] # Categorical features
    }
    
    # Identify field types based on column names and content
    for col in df.columns:
        col_lower = col.lower()
        
        # Direct identifiers
        if any(id_type in col_lower for id_type in ['id', 'ssn', 'drivers', 'passport', 'first', 'last', 'address']):
            field_types['identifiers'].append(col)
        
        # Quasi-identifiers
        elif any(qi_type in col_lower for qi_type in ['birthdate', 'zip', 'race', 'ethnicity', 'city', 'state', 'county']):
            field_types['quasi_identifiers'].append(col)
        
        # Critical features
        elif any(cf_type in col_lower for cf_type in ['gender', 'healthcare_expenses', 'healthcare_coverage']):
            field_types['critical_features'].append(col)
        
        # Sensitive attributes
        elif any(sa_type in col_lower for sa_type in ['lat', 'lon', 'maiden', 'marital']):
            field_types['sensitive_attributes'].append(col)
        
        # Numerical features
        if df[col].dtype in ['int64', 'float64']:
            field_types['numerical_features'].append(col)
        
        # Categorical features
        elif df[col].dtype == 'object':
            field_types['categorical_features'].append(col)
    
    # Print categorization
    for field_type, fields in field_types.items():
        if fields:
            print(f"{field_type.replace('_', ' ').title()}: {', '.join(fields)}")
    
    return field_types

def convert_birthdate_to_features(df):
    """
    Convert birthdate to age and month-year features to reduce identifiability.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Patient dataset with BIRTHDATE column
        
    Returns:
    --------
    pandas.DataFrame
        Dataset with age and month-year features
    """
    print("Converting birthdate to age and month-year features...")
    
    if 'BIRTHDATE' not in df.columns:
        print("BIRTHDATE column not found")
        return df
    
    df_processed = df.copy()
    
    # Convert BIRTHDATE to datetime
    df_processed['BIRTHDATE'] = pd.to_datetime(df_processed['BIRTHDATE'], errors='coerce')
    
    # Calculate age
    current_date = datetime.now()
    df_processed['AGE'] = (current_date - df_processed['BIRTHDATE']).dt.days // 365
    
    # Extract month and year (reduce precision)
    df_processed['BIRTH_MONTH'] = df_processed['BIRTHDATE'].dt.month
    df_processed['BIRTH_YEAR'] = df_processed['BIRTHDATE'].dt.year
    
    # Create age groups for additional privacy
    df_processed['AGE_GROUP'] = pd.cut(df_processed['AGE'], 
                                      bins=[0, 18, 30, 50, 70, 100], 
                                      labels=['0-18', '19-30', '31-50', '51-70', '70+'])
    
    # Drop original BIRTHDATE column
    df_processed = df_processed.drop('BIRTHDATE', axis=1)
    
    print("Birthdate conversion completed")
    return df_processed

def clean_patient_data(df):
    """
    Clean the patient dataset by handling missing values, duplicates, and data quality issues.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Patient dataset
        
    Returns:
    --------
    pandas.DataFrame
        Cleaned dataset
    """
    print("\n=== DATA CLEANING ===")
    
    df_clean = df.copy()
    initial_shape = df_clean.shape
    
    # Remove duplicate records
    initial_duplicates = df_clean.duplicated().sum()
    df_clean = df_clean.drop_duplicates()
    print(f"Removed {initial_duplicates} duplicate records")
    
    # Handle missing values
    missing_before = df_clean.isnull().sum().sum()
    
    # For numerical columns, fill with median
    numerical_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        if df_clean[col].isnull().any():
            median_val = df_clean[col].median()
            df_clean[col].fillna(median_val, inplace=True)
    
    # For categorical columns, fill with mode
    categorical_cols = df_clean.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df_clean[col].isnull().any():
            mode_val = df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'Unknown'
            df_clean[col].fillna(mode_val, inplace=True)
    
    missing_after = df_clean.isnull().sum().sum()
    print(f"Handled missing values: {missing_before:,} → {missing_after:,}")
    
    # Remove columns with too many missing values (>50%)
    high_missing_cols = []
    for col in df_clean.columns:
        missing_pct = (df_clean[col].isnull().sum() / len(df_clean)) * 100
        if missing_pct > 50:
            high_missing_cols.append(col)
    
    if high_missing_cols:
        print(f"Removing columns with >50% missing values: {', '.join(high_missing_cols)}")
        df_clean = df_clean.drop(columns=high_missing_cols)
    
    # Remove rows with too many missing values (>80% of columns)
    row_missing_threshold = len(df_clean.columns) * 0.8
    rows_to_drop = df_clean.isnull().sum(axis=1) > row_missing_threshold
    if rows_to_drop.any():
        print(f"Removing {rows_to_drop.sum()} rows with >80% missing values")
        df_clean = df_clean[~rows_to_drop]
    
    final_shape = df_clean.shape
    print(f"Data cleaning completed: {initial_shape} → {final_shape}")
    
    return df_clean

def normalize_numerical_features(df, numerical_cols, method='minmax'):
    """
    Normalize numerical features to 0-1 range.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Patient dataset
    numerical_cols : list
        List of numerical column names
    method : str
        Normalization method: 'minmax' or 'standard'
        
    Returns:
    --------
    tuple
        (normalized_dataframe, scaler_object)
    """
    print(f"Normalizing numerical features using {method} scaling...")
    
    df_normalized = df.copy()
    scaler = None
    
    if method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'standard':
        scaler = StandardScaler()
    else:
        raise ValueError("Method must be 'minmax' or 'standard'")
    
    # Fit and transform numerical columns
    df_normalized[numerical_cols] = scaler.fit_transform(df_normalized[numerical_cols])
    
    print(f"Normalized {len(numerical_cols)} numerical features")
    return df_normalized, scaler

def encode_categorical_features(df, categorical_cols, method='label'):
    """
    Encode categorical variables using label encoding or one-hot encoding.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Patient dataset
    categorical_cols : list
        List of categorical column names
    method : str
        Encoding method: 'label' or 'onehot'
        
    Returns:
    --------
    tuple
        (encoded_dataframe, encoders_dict)
    """
    print(f"Encoding categorical features using {method} encoding...")
    
    df_encoded = df.copy()
    encoders = {}
    
    if method == 'label':
        for col in categorical_cols:
            if col in df_encoded.columns:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                encoders[col] = le
                print(f"Label encoded: {col}")
    
    elif method == 'onehot':
        for col in categorical_cols:
            if col in df_encoded.columns:
                # Get dummies and drop original column
                dummies = pd.get_dummies(df_encoded[col], prefix=col)
                df_encoded = pd.concat([df_encoded, dummies], axis=1)
                df_encoded = df_encoded.drop(col, axis=1)
                print(f"One-hot encoded: {col} -> {len(dummies.columns)} columns")
    
    print(f"Encoded {len(categorical_cols)} categorical features")
    return df_encoded, encoders

def split_features_for_cgan(df, field_types):
    """
    Split features into condition vector (critical features) and target vector (quasi-identifiers).
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Processed patient dataset
    field_types : dict
        Dictionary categorizing fields by type
        
    Returns:
    --------
    tuple
        (condition_features, target_features, feature_names)
    """
    print("Splitting features for CGAN training...")
    
    # Condition vector: critical features to preserve
    condition_cols = field_types['critical_features'].copy()
    
    # Add age group if available
    if 'AGE_GROUP' in df.columns:
        condition_cols.append('AGE_GROUP')
    
    # Target vector: quasi-identifiers to generate synthetically
    target_cols = field_types['quasi_identifiers'].copy()
    
    # Add age if available (as it's derived from birthdate)
    if 'AGE' in df.columns:
        target_cols.append('AGE')
    
    # Ensure all columns exist in the dataset
    condition_cols = [col for col in condition_cols if col in df.columns]
    target_cols = [col for col in target_cols if col in df.columns]
    
    print(f"Condition features ({len(condition_cols)}): {', '.join(condition_cols)}")
    print(f"Target features ({len(target_cols)}): {', '.join(target_cols)}")
    
    # Extract feature matrices
    condition_features = df[condition_cols].values
    target_features = df[target_cols].values
    
    return condition_features, target_features, {
        'condition': condition_cols,
        'target': target_cols
    }

def prepare_data_for_cgan(df, field_types, output_dir='data/processed'):
    """
    Complete data preparation pipeline for CGAN training.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Raw patient dataset
    field_types : dict
        Dictionary categorizing fields by type
    output_dir : str
        Directory to save processed data
        
    Returns:
    --------
    dict
        Dictionary containing processed data and metadata
    """
    print("\n=== COMPLETE DATA PREPARATION PIPELINE ===")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Step 1: Convert birthdate to age features
    df_processed = convert_birthdate_to_features(df)
    
    # Step 2: Handle missing values
    print("Handling missing values...")
    df_processed = df_processed.fillna(method='ffill').fillna(method='bfill')
    
    # Step 3: Normalize numerical features
    numerical_cols = field_types['numerical_features']
    if numerical_cols:
        df_processed, scaler = normalize_numerical_features(df_processed, numerical_cols)
        # Save scaler for later use
        joblib.dump(scaler, f"{output_dir}/numerical_scaler.pkl")
    
    # Step 4: Encode categorical features
    categorical_cols = field_types['categorical_features']
    if categorical_cols:
        df_processed, encoders = encode_categorical_features(df_processed, categorical_cols, method='label')
        # Save encoders for later use
        joblib.dump(encoders, f"{output_dir}/categorical_encoders.pkl")
    
    # Step 5: Split features for CGAN
    condition_features, target_features, feature_names = split_features_for_cgan(df_processed, field_types)
    
    # Step 6: Save processed data
    df_processed.to_csv(f"{output_dir}/patient_processed.csv", index=False)
    
    # Save feature matrices
    np.save(f"{output_dir}/condition_features.npy", condition_features)
    np.save(f"{output_dir}/target_features.npy", target_features)
    
    # Save feature names
    with open(f"{output_dir}/feature_names.json", 'w') as f:
        json.dump(feature_names, f, indent=2)
    
    print(f"Data preparation completed! Files saved to {output_dir}")
    
    return {
        'processed_dataframe': df_processed,
        'condition_features': condition_features,
        'target_features': target_features,
        'feature_names': feature_names,
        'output_dir': output_dir
    }

def load_processed_data(data_dir='data/processed'):
    """
    Load processed data for CGAN training.
    
    Parameters:
    -----------
    data_dir : str
        Directory containing processed data
        
    Returns:
    --------
    dict
        Dictionary containing loaded data
    """
    print(f"Loading processed data from {data_dir}...")
    
    try:
        # Load feature matrices
        condition_features = np.load(f"{data_dir}/condition_features.npy")
        target_features = np.load(f"{data_dir}/target_features.npy")
        
        # Load feature names
        with open(f"{data_dir}/feature_names.json", 'r') as f:
            feature_names = json.load(f)
        
        # Load processed dataframe
        df_processed = pd.read_csv(f"{data_dir}/patient_processed.csv")
        
        print(f"Successfully loaded processed data:")
        print(f"  Condition features: {condition_features.shape}")
        print(f"  Target features: {target_features.shape}")
        print(f"  Processed dataframe: {df_processed.shape}")
        
        return {
            'condition_features': condition_features,
            'target_features': target_features,
            'feature_names': feature_names,
            'processed_dataframe': df_processed
        }
        
    except Exception as e:
        print(f"Error loading processed data: {e}")
        return None

def preprocess_patient_data(file_path, n_records=2000, output_path=None):
    """
    Main preprocessing function for patient data.
    
    Parameters:
    -----------
    file_path : str
        Path to the patient CSV file
    n_records : int, default=2000
        Number of records to process
    output_path : str, optional
        Path to save the preprocessed data
        
    Returns:
    --------
    pandas.DataFrame
        Preprocessed patient dataset
    """
    print("=" * 60)
    print("PATIENT DATA PREPROCESSING")
    print("=" * 60)
    
    # Step 1: Load data
    df = load_patient_data(file_path, n_records)
    if df is None:
        return None
    
    # Step 2: Analyze structure
    structure_info = analyze_data_structure(df)
    
    # Step 3: Categorize columns
    column_categories = identify_column_categories(df)
    
    # Step 4: Clean data
    df_clean = clean_patient_data(df)
    
    # Step 5: Save preprocessed data
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_clean.to_csv(output_path, index=False)
        print(f"\nPreprocessed data saved to: {output_path}")
    
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETED SUCCESSFULLY")
    print("=" * 60)
    
    return df_clean

if __name__ == "__main__":
    # Example usage
    input_file = "data/raw/patient.csv"
    output_file = "data/processed/patient_processed.csv"
    
    # Preprocess first 2000 records
    df_processed = preprocess_patient_data(input_file, n_records=2000, output_path=output_file)
    
    if df_processed is not None:
        print(f"\nFinal processed dataset shape: {df_processed.shape}")
        print("Preprocessing completed successfully!")
    else:
        print("Preprocessing failed!")

import pandas as pd
import numpy as np
import os

def generate_synthetic_data(n_records=2000, seed=42, output_path=None):
    """
    Generate synthetic patient data for diabetes research.
    
    Parameters:
    -----------
    n_records : int, default=2000
        Number of synthetic records to generate
    seed : int, default=42
        Random seed for reproducibility
    output_path : str, optional
        Path to save the generated data. If None, data is only returned as DataFrame
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing the synthetic patient data
    """
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Generate synthetic patient data
    data = {
        'PatientID': range(1, n_records+1),
        'Age': np.random.randint(21, 81, size=n_records),
        'Gender': np.random.choice(['Male', 'Female'], size=n_records),
        'Pregnancies': np.random.randint(0, 16, size=n_records),
        'Glucose': np.round(np.random.uniform(70, 200, size=n_records), 1),
        'BloodPressure': np.round(np.random.uniform(50, 120, size=n_records), 1),
        'SkinThickness': np.round(np.random.uniform(10, 50, size=n_records), 1),
        'Insulin': np.round(np.random.uniform(15, 276, size=n_records), 1),
        'BMI': np.round(np.random.uniform(18, 45, size=n_records), 1),
        'DiabetesPedigreeFunction': np.round(np.random.uniform(0.1, 2.5, size=n_records), 2),
        'Outcome': np.random.choice([0, 1], size=n_records)
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV if output_path is provided
    if output_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Synthetic patient data created successfully and saved to {output_path}!")
    
    return df


def main():
    """
    Main function to generate synthetic data when script is run directly.
    """
    # Default path for saving the synthetic data
    default_path = os.path.join('..', 'data', 'raw', 'synthetic_data.csv')
    
    # Generate and save synthetic data
    generate_synthetic_data(output_path=default_path)


if __name__ == "__main__":
    main()
import pandas as pd
import numpy as np
import torch
from pathlib import Path
import json
import joblib
from src.gan_model import ConditionalGAN
from src.preprocessing import load_processed_data
import matplotlib.pyplot as plt
import seaborn as sns

class SyntheticDataGenerator:
    """
    Class for generating synthetic patient data using trained CGAN.
    """
    
    def __init__(self, models_dir='models', processed_data_dir='data/processed'):
        """
        Initialize the synthetic data generator.
        
        Parameters:
        -----------
        models_dir : str
            Directory containing trained CGAN models
        processed_data_dir : str
            Directory containing processed data
        """
        self.models_dir = Path(models_dir)
        self.processed_data_dir = Path(processed_data_dir)
        self.cgan = None
        self.processed_data = None
        self.feature_names = None
        
        # Load processed data
        self._load_processed_data()
    
    def _load_processed_data(self):
        """Load processed data for synthetic generation."""
        try:
            self.processed_data = load_processed_data(str(self.processed_data_dir))
            if self.processed_data:
                self.feature_names = self.processed_data['feature_names']
                print("Processed data loaded successfully")
            else:
                print("Could not load processed data")
        except Exception as e:
            print(f"Error loading processed data: {e}")
    
    def load_trained_cgan(self, model_name='latest'):
        """
        Load a trained CGAN model.
        
        Parameters:
        -----------
        model_name : str
            Name of the model to load ('latest' or specific epoch)
        """
        print(f"Loading trained CGAN model: {model_name}")
        
        if model_name == 'latest':
            # Find the latest model
            model_dirs = list(self.models_dir.glob('cgan_epoch_*'))
            if not model_dirs:
                raise FileNotFoundError("No trained models found")
            model_dir = max(model_dirs, key=lambda x: int(x.name.split('_')[-1]))
        else:
            model_dir = self.models_dir / model_name
        
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
        
        # Initialize CGAN
        self.cgan = ConditionalGAN(noise_dim=100)
        
        # Build models with correct dimensions
        condition_dim = self.processed_data['condition_features'].shape[1]
        target_dim = self.processed_data['target_features'].shape[1]
        
        self.cgan.build_models(condition_dim, target_dim)
        
        # Load trained weights
        self.cgan.load_models(str(model_dir))
        
        print(f"CGAN model loaded from {model_dir}")
    
    def generate_synthetic_dataset(self, num_samples=None, output_file=None):
        """
        Generate synthetic patient dataset.
        
        Parameters:
        -----------
        num_samples : int, optional
            Number of synthetic samples to generate
        output_file : str, optional
            Path to save synthetic dataset
            
        Returns:
        --------
        pandas.DataFrame
            Synthetic patient dataset
        """
        if self.cgan is None:
            raise ValueError("CGAN model not loaded. Call load_trained_cgan() first.")
        
        if self.processed_data is None:
            raise ValueError("Processed data not loaded.")
        
        print("Generating synthetic dataset...")
        
        # Get condition features
        condition_features = self.processed_data['condition_features']
        
        # Generate synthetic quasi-identifiers
        synthetic_target = self.cgan.generate_synthetic_data(
            condition_features, num_samples
        )
        
        # Create synthetic dataset
        synthetic_df = self._create_synthetic_dataframe(synthetic_target)
        
        # Save if output file specified
        if output_file:
            synthetic_df.to_csv(output_file, index=False)
            print(f"Synthetic dataset saved to {output_file}")
        
        return synthetic_df
    
    def _create_synthetic_dataframe(self, synthetic_target):
        """
        Create synthetic dataframe from generated features.
        
        Parameters:
        -----------
        synthetic_target : np.ndarray
            Generated synthetic quasi-identifiers
            
        Returns:
        --------
        pandas.DataFrame
            Synthetic patient dataset
        """
        print("Creating synthetic dataframe...")
        
        # Get feature names
        target_cols = self.feature_names['target']
        condition_cols = self.feature_names['condition']
        
        # Create synthetic dataframe
        synthetic_df = pd.DataFrame(synthetic_target, columns=target_cols)
        
        # Add condition features (preserved from original)
        condition_features = self.processed_data['condition_features']
        for i, col in enumerate(condition_cols):
            synthetic_df[col] = condition_features[:len(synthetic_df), i]
        
        # Add pseudonymized identifiers
        synthetic_df = self._add_pseudonymized_identifiers(synthetic_df)
        
        # Reorder columns to match original structure
        synthetic_df = self._reorder_columns(synthetic_df)
        
        print(f"Synthetic dataframe created: {synthetic_df.shape}")
        return synthetic_df
    
    def _add_pseudonymized_identifiers(self, df):
        """
        Add pseudonymized identifiers to synthetic dataset.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Synthetic dataset
            
        Returns:
        --------
        pandas.DataFrame
            Dataset with pseudonymized identifiers
        """
        print("Adding pseudonymized identifiers...")
        
        # Generate new pseudonymized IDs
        num_samples = len(df)
        
        # Patient ID
        df['Id'] = [f"synth_{i:06d}" for i in range(num_samples)]
        
        # SSN (synthetic format)
        df['SSN'] = [f"{np.random.randint(100, 999)}-{np.random.randint(10, 99)}-{np.random.randint(1000, 9999)}" 
                    for _ in range(num_samples)]
        
        # Names
        df['FIRST'] = [f"Synthetic_First_{i:03d}" for i in range(num_samples)]
        df['LAST'] = [f"Synthetic_Last_{i:03d}" for i in range(num_samples)]
        
        # Address (masked)
        df['ADDRESS'] = "Address_Redacted"
        df['CITY'] = df['CITY']  # Preserve from condition
        df['STATE'] = df['STATE']  # Preserve from condition
        df['ZIP'] = df['ZIP']  # Preserve from condition
        
        # Other sensitive fields
        sensitive_fields = ['DRIVERS', 'PASSPORT', 'MAIDEN', 'LAT', 'LON']
        for field in sensitive_fields:
            if field in df.columns:
                df[field] = f"{field}_Redacted"
        
        return df
    
    def _reorder_columns(self, df):
        """
        Reorder columns to match original dataset structure.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Synthetic dataset
            
        Returns:
        --------
        pandas.DataFrame
            Reordered dataset
        """
        # Define column order (matching original patient.csv)
        original_order = [
            'Id', 'BIRTHDATE', 'DEATHDATE', 'SSN', 'DRIVERS', 'PASSPORT',
            'PREFIX', 'FIRST', 'LAST', 'SUFFIX', 'MAIDEN', 'MARITAL',
            'RACE', 'ETHNICITY', 'GENDER', 'BIRTHPLACE', 'ADDRESS',
            'CITY', 'STATE', 'COUNTY', 'ZIP', 'LAT', 'LON',
            'HEALTHCARE_EXPENSES', 'HEALTHCARE_COVERAGE'
        ]
        
        # Add any missing columns
        for col in original_order:
            if col not in df.columns:
                df[col] = None
        
        # Reorder columns
        available_cols = [col for col in original_order if col in df.columns]
        df = df[available_cols]
        
        return df
    
    def evaluate_synthetic_quality(self, synthetic_df, original_df=None):
        """
        Evaluate the quality of synthetic data.
        
        Parameters:
        -----------
        synthetic_df : pandas.DataFrame
            Synthetic dataset
        original_df : pandas.DataFrame, optional
            Original dataset for comparison
            
        Returns:
        --------
        dict
            Dictionary containing evaluation metrics
        """
        print("Evaluating synthetic data quality...")
        
        metrics = {}
        
        # Basic statistics
        metrics['synthetic_shape'] = synthetic_df.shape
        metrics['synthetic_columns'] = list(synthetic_df.columns)
        
        # Missing values
        metrics['missing_values'] = synthetic_df.isnull().sum().sum()
        metrics['missing_percentage'] = (metrics['missing_values'] / synthetic_df.size) * 100
        
        # Data types
        metrics['data_types'] = synthetic_df.dtypes.value_counts().to_dict()
        
        # Numerical features statistics
        numerical_cols = synthetic_df.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            metrics['numerical_stats'] = synthetic_df[numerical_cols].describe().to_dict()
        
        # Categorical features statistics
        categorical_cols = synthetic_df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            metrics['categorical_stats'] = {}
            for col in categorical_cols:
                metrics['categorical_stats'][col] = synthetic_df[col].value_counts().to_dict()
        
        # Compare with original if available
        if original_df is not None:
            metrics['comparison'] = self._compare_with_original(synthetic_df, original_df)
        
        print("Quality evaluation completed!")
        return metrics
    
    def _compare_with_original(self, synthetic_df, original_df):
        """
        Compare synthetic dataset with original dataset.
        
        Parameters:
        -----------
        synthetic_df : pandas.DataFrame
            Synthetic dataset
        original_df : pandas.DataFrame
            Original dataset
            
        Returns:
        --------
        dict
            Comparison metrics
        """
        comparison = {}
        
        # Shape comparison
        comparison['shape_ratio'] = {
            'rows': len(synthetic_df) / len(original_df),
            'columns': len(synthetic_df.columns) / len(original_df.columns)
        }
        
        # Common columns
        common_cols = set(synthetic_df.columns) & set(original_df.columns)
        comparison['common_columns'] = len(common_cols)
        comparison['column_overlap'] = len(common_cols) / len(original_df.columns)
        
        # Numerical features comparison
        numerical_cols = [col for col in common_cols 
                         if synthetic_df[col].dtype in [np.number] and original_df[col].dtype in [np.number]]
        
        if numerical_cols:
            comparison['numerical_comparison'] = {}
            for col in numerical_cols[:5]:  # Limit to first 5 for readability
                comparison['numerical_comparison'][col] = {
                    'original_mean': float(original_df[col].mean()),
                    'synthetic_mean': float(synthetic_df[col].mean()),
                    'original_std': float(original_df[col].std()),
                    'synthetic_std': float(synthetic_df[col].std())
                }
        
        return comparison
    
    def plot_synthetic_vs_original(self, synthetic_df, original_df, output_dir='results/figures'):
        """
        Create comparison plots between synthetic and original data.
        
        Parameters:
        -----------
        synthetic_df : pandas.DataFrame
            Synthetic dataset
        original_df : pandas.DataFrame
            Original dataset
        output_dir : str
            Directory to save plots
        """
        print("Creating comparison plots...")
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Common numerical columns
        numerical_cols = [col for col in synthetic_df.columns 
                         if synthetic_df[col].dtype in [np.number] and col in original_df.columns]
        
        if not numerical_cols:
            print("No numerical columns found for comparison")
            return
        
        # Limit to first 6 columns for readability
        plot_cols = numerical_cols[:6]
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, col in enumerate(plot_cols):
            if i >= len(axes):
                break
            
            ax = axes[i]
            
            # Plot histograms
            ax.hist(original_df[col].dropna(), bins=30, alpha=0.7, label='Original', density=True)
            ax.hist(synthetic_df[col].dropna(), bins=30, alpha=0.7, label='Synthetic', density=True)
            
            ax.set_title(f'{col} Distribution')
            ax.set_xlabel(col)
            ax.set_ylabel('Density')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(plot_cols), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = f"{output_dir}/synthetic_vs_original_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plots saved to {plot_path}")
        
        plt.show()

def generate_synthetic_patient_data(input_file='data/raw/patient.csv', 
                                  output_file='data/synthetic/patient_synthetic.csv',
                                  models_dir='models',
                                  num_samples=None):
    """
    Complete pipeline for generating synthetic patient data.
    
    Parameters:
    -----------
    input_file : str
        Path to input patient data
    output_file : str
        Path to save synthetic data
    models_dir : str
        Directory containing trained CGAN models
    num_samples : int, optional
        Number of synthetic samples to generate
        
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    try:
        print("=== SYNTHETIC DATA GENERATION PIPELINE ===")
        
        # Initialize generator
        generator = SyntheticDataGenerator(models_dir, 'data/processed')
        
        # Load trained CGAN
        generator.load_trained_cgan()
        
        # Generate synthetic dataset
        synthetic_df = generator.generate_synthetic_dataset(num_samples, output_file)
        
        # Load original data for comparison
        original_df = pd.read_csv(input_file)
        
        # Evaluate quality
        metrics = generator.evaluate_synthetic_quality(synthetic_df, original_df)
        
        # Create comparison plots
        generator.plot_synthetic_vs_original(synthetic_df, original_df)
        
        print("Synthetic data generation completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error during synthetic data generation: {e}")
        return False

#!/usr/bin/env python3
"""
Main project runner for the Data Anonymization System using CGAN.

This script orchestrates the complete pipeline:
1. Data preprocessing and field identification
2. Pseudonymization of identifiers
3. CGAN training for synthetic data generation
4. Validation of privacy and utility metrics
"""

import os
import sys
from pathlib import Path
import argparse
import time

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from preprocessing import load_patient_data, identify_field_types, prepare_data_for_cgan
from pseudonymization import pseudonymize_patient_data
from gan_model import ConditionalGAN
from synthetic_generator import generate_synthetic_patient_data
from validation import validate_anonymization

def main():
    """Main function to run the complete anonymization pipeline."""
    parser = argparse.ArgumentParser(description='Data Anonymization Pipeline using CGAN')
    parser.add_argument('--input', default='data/raw/patient.csv', 
                       help='Input patient data file')
    parser.add_argument('--output-dir', default='data', 
                       help='Output directory for processed data')
    parser.add_argument('--n-records', type=int, default=2000,
                       help='Number of records to process')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of CGAN training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for CGAN training')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip CGAN training and use existing models')
    parser.add_argument('--validate-only', action='store_true',
                       help='Only run validation on existing data')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("DATA ANONYMIZATION PIPELINE USING CONDITIONAL GAN")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        # Step 1: Data Loading and Analysis
        print("\n1. LOADING AND ANALYZING PATIENT DATA")
        print("-" * 40)
        
        df = load_patient_data(args.input, args.n_records)
        if df is None:
            print("Failed to load patient data. Exiting.")
            return False
        
        # Analyze data structure
        from preprocessing import analyze_data_structure
        analyze_data_structure(df)
        
        # Identify field types
        field_types = identify_field_types(df)
        
        # Step 2: Data Preprocessing
        print("\n2. PREPROCESSING DATA FOR CGAN")
        print("-" * 40)
        
        processed_data = prepare_data_for_cgan(df, field_types, f"{args.output_dir}/processed")
        if processed_data is None:
            print("Failed to preprocess data. Exiting.")
            return False
        
        # Step 3: Pseudonymization
        print("\n3. PSEUDONYMIZING IDENTIFIERS")
        print("-" * 40)
        
        pseudonymized_file = f"{args.output_dir}/processed/patient_pseudonymized.csv"
        success = pseudonymize_patient_data(
            f"{args.output_dir}/processed/patient_processed.csv",
            pseudonymized_file
        )
        
        if not success:
            print("Failed to pseudonymize data. Exiting.")
            return False
        
        # Step 4: CGAN Training (if not skipped)
        if not args.skip_training and not args.validate_only:
            print("\n4. TRAINING CONDITIONAL GAN")
            print("-" * 40)
            
            # Initialize CGAN
            cgan = ConditionalGAN(noise_dim=100)
            
            # Build models
            condition_dim = processed_data['condition_features'].shape[1]
            target_dim = processed_data['target_features'].shape[1]
            cgan.build_models(condition_dim, target_dim)
            
            # Train CGAN
            cgan.train(
                processed_data['condition_features'],
                processed_data['target_features'],
                epochs=args.epochs,
                batch_size=args.batch_size
            )
            
            # Save models
            models_dir = Path('models')
            models_dir.mkdir(exist_ok=True)
            cgan.save_models(f"models/cgan_epoch_{args.epochs}")
            
            # Plot training history
            cgan.plot_training_history(f"results/figures/training_history.png")
        
        # Step 5: Synthetic Data Generation
        if not args.validate_only:
            print("\n5. GENERATING SYNTHETIC DATA")
            print("-" * 40)
            
            success = generate_synthetic_patient_data(
                input_file=args.input,
                output_file=f"{args.output_dir}/synthetic/patient_synthetic.csv",
                models_dir='models'
            )
            
            if not success:
                print("Failed to generate synthetic data. Exiting.")
                return False
        
        # Step 6: Validation
        print("\n6. VALIDATING ANONYMIZATION")
        print("-" * 40)
        
        # Validate pseudonymized data
        print("Validating pseudonymized data...")
        validate_anonymization(
            args.input,
            pseudonymized_file,
            'results/validation/pseudonymized'
        )
        
        # Validate synthetic data
        synthetic_file = f"{args.output_dir}/synthetic/patient_synthetic.csv"
        if os.path.exists(synthetic_file):
            print("Validating synthetic data...")
            validate_anonymization(
                args.input,
                synthetic_file,
                'results/validation/synthetic'
            )
        
        # Step 7: Generate Summary Report
        print("\n7. GENERATING SUMMARY REPORT")
        print("-" * 40)
        
        generate_summary_report(args, start_time)
        
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        total_time = time.time() - start_time
        print(f"Total execution time: {total_time:.2f} seconds")
        
        return True
        
    except Exception as e:
        print(f"\nERROR: Pipeline failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

def generate_summary_report(args, start_time):
    """Generate a summary report of the anonymization process."""
    
    # Create results directory
    results_dir = Path('results/reports')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate report
    report_file = results_dir / 'pipeline_summary.txt'
    
    with open(report_file, 'w') as f:
        f.write("DATA ANONYMIZATION PIPELINE SUMMARY REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("EXECUTION PARAMETERS:\n")
        f.write("-" * 25 + "\n")
        f.write(f"Input file: {args.input}\n")
        f.write(f"Output directory: {args.output_dir}\n")
        f.write(f"Number of records: {args.n_records}\n")
        f.write(f"CGAN epochs: {args.epochs}\n")
        f.write(f"Batch size: {args.batch_size}\n")
        f.write(f"Skip training: {args.skip_training}\n")
        f.write(f"Validate only: {args.validate_only}\n\n")
        
        f.write("OUTPUT FILES:\n")
        f.write("-" * 15 + "\n")
        f.write(f"Processed data: {args.output_dir}/processed/patient_processed.csv\n")
        f.write(f"Pseudonymized data: {args.output_dir}/processed/patient_pseudonymized.csv\n")
        f.write(f"Synthetic data: {args.output_dir}/synthetic/patient_synthetic.csv\n")
        f.write(f"CGAN models: models/\n")
        f.write(f"Validation results: results/validation/\n")
        f.write(f"Training plots: results/figures/\n\n")
        
        f.write("PRIVACY METRICS:\n")
        f.write("-" * 18 + "\n")
        f.write("• K-anonymity: Calculated for k=2,5,10,20\n")
        f.write("• L-diversity: Calculated for l=2,3,5\n")
        f.write("• T-closeness: Calculated for t=0.1,0.2,0.3\n")
        f.write("• Re-identification risk: Based on quasi-identifier uniqueness\n\n")
        
        f.write("UTILITY METRICS:\n")
        f.write("-" * 18 + "\n")
        f.write("• ML model performance comparison\n")
        f.write("• Statistical similarity analysis\n")
        f.write("• Feature distribution preservation\n\n")
        
        f.write("NEXT STEPS:\n")
        f.write("-" * 12 + "\n")
        f.write("1. Review validation results in results/validation/\n")
        f.write("2. Examine synthetic data quality in data/synthetic/\n")
        f.write("3. Adjust CGAN parameters if needed\n")
        f.write("4. Deploy anonymized data for analysis\n")
    
    print(f"Summary report generated: {report_file}")

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

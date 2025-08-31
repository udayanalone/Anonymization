#!/usr/bin/env python3
"""
Script to run feature analysis and selection on the processed patient data.
"""

import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from feature_analysis import generate_feature_analysis_report

def main():
    """Main function to run feature analysis."""
    
    # File paths
    input_file = "data/processed/patient_processed.csv"
    
    print("Starting feature analysis on processed patient data...")
    print(f"Input file: {input_file}")
    
    # Run feature analysis
    results = generate_feature_analysis_report(input_file)
    
    if results is not None:
        print(f"\n✅ Feature analysis completed successfully!")
        print(f"Selected {len(results['critical_features']['selected_features'])} critical features")
        print(f"Features removed: {len(results['dataset_info']['columns']) - len(results['critical_features']['selected_features'])}")
        
        print(f"\nSelected critical features:")
        for i, feature in enumerate(results['critical_features']['selected_features'], 1):
            importance_score = results['critical_features']['importance_scores'].get(feature, 0)
            print(f"  {i:2d}. {feature:<25} (Importance: {importance_score:.3f})")
    else:
        print("\n❌ Feature analysis failed!")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

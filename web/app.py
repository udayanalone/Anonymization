#!/usr/bin/env python3
"""
Flask web application for the Data Anonymization System.

Provides a web interface for:
- Uploading patient data
- Running the anonymization pipeline
- Viewing results and visualizations
- Downloading anonymized datasets
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for, flash
from werkzeug.utils import secure_filename
import plotly.express as px
import plotly.graph_objects as go
import plotly.utils
from datetime import datetime
import threading
import time

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.preprocessing import load_patient_data, identify_field_types, prepare_data_for_cgan
from src.pseudonymization import pseudonymize_patient_data
from src.validation import validate_anonymization

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change in production

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

# Ensure upload directory exists
Path(UPLOAD_FOLDER).mkdir(exist_ok=True)

# Global variables for pipeline status
pipeline_status = {
    'running': False,
    'current_step': '',
    'progress': 0,
    'logs': [],
    'error': None
}

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def log_message(message):
    """Add message to pipeline logs."""
    timestamp = datetime.now().strftime('%H:%M:%S')
    pipeline_status['logs'].append(f"[{timestamp}] {message}")
    if len(pipeline_status['logs']) > 100:  # Keep only last 100 logs
        pipeline_status['logs'] = pipeline_status['logs'][-100:]

def run_pipeline_thread(input_file, output_dir, n_records=2000):
    """Run the anonymization pipeline in a separate thread."""
    global pipeline_status
    
    try:
        pipeline_status['running'] = True
        pipeline_status['error'] = None
        pipeline_status['logs'] = []
        
        log_message("Starting anonymization pipeline...")
        
        # Step 1: Load and analyze data
        pipeline_status['current_step'] = 'Loading and analyzing data'
        pipeline_status['progress'] = 10
        log_message("Loading patient data...")
        
        df = load_patient_data(input_file, n_records)
        if df is None:
            raise Exception("Failed to load patient data")
        
        log_message(f"Loaded {len(df)} records successfully")
        
        # Step 2: Identify field types
        pipeline_status['progress'] = 20
        log_message("Identifying field types...")
        
        field_types = identify_field_types(df)
        log_message(f"Identified {len(field_types['identifiers'])} identifiers, "
                   f"{len(field_types['quasi_identifiers'])} quasi-identifiers, "
                   f"{len(field_types['critical_features'])} critical features")
        
        # Step 3: Preprocess data
        pipeline_status['current_step'] = 'Preprocessing data for CGAN'
        pipeline_status['progress'] = 30
        log_message("Preprocessing data...")
        
        processed_data = prepare_data_for_cgan(df, field_types, f"{output_dir}/processed")
        if processed_data is None:
            raise Exception("Failed to preprocess data")
        
        log_message("Data preprocessing completed")
        
        # Step 4: Pseudonymization
        pipeline_status['current_step'] = 'Pseudonymizing identifiers'
        pipeline_status['progress'] = 50
        log_message("Applying pseudonymization...")
        
        pseudonymized_file = f"{output_dir}/processed/patient_pseudonymized.csv"
        success = pseudonymize_patient_data(
            f"{output_dir}/processed/patient_processed.csv",
            pseudonymized_file
        )
        
        if not success:
            raise Exception("Failed to pseudonymize data")
        
        log_message("Pseudonymization completed")
        
        # Step 5: Validation
        pipeline_status['current_step'] = 'Validating results'
        pipeline_status['progress'] = 80
        log_message("Running validation...")
        
        validate_anonymization(
            input_file,
            pseudonymized_file,
            f"{output_dir}/validation"
        )
        
        log_message("Validation completed")
        
        # Step 6: Complete
        pipeline_status['current_step'] = 'Pipeline completed'
        pipeline_status['progress'] = 100
        log_message("Anonymization pipeline completed successfully!")
        
    except Exception as e:
        pipeline_status['error'] = str(e)
        log_message(f"ERROR: {str(e)}")
        log_message("Pipeline failed")
    
    finally:
        pipeline_status['running'] = False

@app.route('/')
def index():
    """Main page."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        # Get number of records to process
        n_records = request.form.get('n_records', 2000, type=int)
        
        # Start pipeline in background
        output_dir = f"data_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        thread = threading.Thread(
            target=run_pipeline_thread,
            args=(filepath, output_dir, n_records)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'message': f'File uploaded successfully. Processing {n_records} records.',
            'output_dir': output_dir
        })
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/status')
def get_status():
    """Get current pipeline status."""
    return jsonify(pipeline_status)

@app.route('/results/<output_dir>')
def view_results(output_dir):
    """View pipeline results."""
    try:
        # Check if results exist
        processed_dir = Path(output_dir) / 'processed'
        validation_dir = Path(output_dir) / 'validation'
        
        if not processed_dir.exists():
            flash('Results not found. Please run the pipeline first.', 'error')
            return redirect(url_for('index'))
        
        # Load datasets
        datasets = {}
        if (processed_dir / 'patient_processed.csv').exists():
            datasets['processed'] = pd.read_csv(processed_dir / 'patient_processed.csv')
        
        if (processed_dir / 'patient_pseudonymized.csv').exists():
            datasets['pseudonymized'] = pd.read_csv(processed_dir / 'patient_pseudonymized.csv')
        
        # Load validation results
        validation_results = {}
        if (validation_dir / 'validation_results.json').exists():
            with open(validation_dir / 'validation_results.json', 'r') as f:
                validation_results = json.load(f)
        
        # Generate visualizations
        visualizations = generate_visualizations(datasets, validation_results)
        
        return render_template('results.html', 
                             output_dir=output_dir,
                             datasets=datasets,
                             validation_results=validation_results,
                             visualizations=visualizations)
    
    except Exception as e:
        flash(f'Error loading results: {str(e)}', 'error')
        return redirect(url_for('index'))

def generate_visualizations(datasets, validation_results):
    """Generate Plotly visualizations for the results."""
    visualizations = {}
    
    try:
        if 'processed' in datasets and 'pseudonymized' in datasets:
            df_orig = datasets['processed']
            df_pseudo = datasets['pseudonymized']
            
            # Age distribution comparison
            if 'AGE' in df_orig.columns and 'AGE' in df_pseudo.columns:
                fig_age = go.Figure()
                fig_age.add_trace(go.Histogram(x=df_orig['AGE'], name='Original', opacity=0.7))
                fig_age.add_trace(go.Histogram(x=df_pseudo['AGE'], name='Pseudonymized', opacity=0.7))
                fig_age.update_layout(title='Age Distribution Comparison', 
                                   xaxis_title='Age', yaxis_title='Count')
                visualizations['age_distribution'] = json.dumps(fig_age, cls=plotly.utils.PlotlyJSONEncoder)
            
            # Gender distribution
            if 'GENDER' in df_orig.columns and 'GENDER' in df_pseudo.columns:
                gender_orig = df_orig['GENDER'].value_counts()
                gender_pseudo = df_pseudo['GENDER'].value_counts()
                
                fig_gender = go.Figure(data=[
                    go.Bar(name='Original', x=gender_orig.index, y=gender_orig.values),
                    go.Bar(name='Pseudonymized', x=gender_pseudo.index, y=gender_pseudo.values)
                ])
                fig_gender.update_layout(title='Gender Distribution Comparison', 
                                       xaxis_title='Gender', yaxis_title='Count')
                visualizations['gender_distribution'] = json.dumps(fig_gender, cls=plotly.utils.PlotlyJSONEncoder)
            
            # Healthcare expenses comparison
            if 'HEALTHCARE_EXPENSES' in df_orig.columns and 'HEALTHCARE_EXPENSES' in df_pseudo.columns:
                fig_expenses = go.Figure()
                fig_expenses.add_trace(go.Box(y=df_orig['HEALTHCARE_EXPENSES'], name='Original'))
                fig_expenses.add_trace(go.Box(y=df_pseudo['HEALTHCARE_EXPENSES'], name='Pseudonymized'))
                fig_expenses.update_layout(title='Healthcare Expenses Comparison', 
                                        yaxis_title='Expenses ($)')
                visualizations['expenses_comparison'] = json.dumps(fig_expenses, cls=plotly.utils.PlotlyJSONEncoder)
        
        # Privacy metrics visualization
        if 'privacy' in validation_results:
            privacy_data = validation_results['privacy']
            
            # K-anonymity results
            if 'k_anonymity' in privacy_data:
                k_values = []
                achieved = []
                for k, result in privacy_data['k_anonymity'].items():
                    k_values.append(int(k.split('=')[1]))
                    achieved.append(1 if result['achieved'] else 0)
                
                fig_k_anon = go.Figure(data=[
                    go.Bar(x=k_values, y=achieved, name='K-Anonymity Achievement')
                ])
                fig_k_anon.update_layout(title='K-Anonymity Results', 
                                       xaxis_title='K Value', yaxis_title='Achieved (1) / Not Achieved (0)')
                visualizations['k_anonymity'] = json.dumps(fig_k_anon, cls=plotly.utils.PlotlyJSONEncoder)
    
    except Exception as e:
        print(f"Error generating visualizations: {e}")
    
    return visualizations

@app.route('/download/<output_dir>/<file_type>')
def download_file(output_dir, file_type):
    """Download processed files."""
    try:
        if file_type == 'processed':
            filepath = Path(output_dir) / 'processed' / 'patient_processed.csv'
        elif file_type == 'pseudonymized':
            filepath = Path(output_dir) / 'processed' / 'patient_pseudonymized.csv'
        elif file_type == 'validation':
            filepath = Path(output_dir) / 'validation' / 'validation_results.json'
        else:
            return jsonify({'error': 'Invalid file type'}), 400
        
        if not filepath.exists():
            return jsonify({'error': 'File not found'}), 404
        
        return send_file(filepath, as_attachment=True)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/dataset_stats/<output_dir>')
def get_dataset_stats(output_dir):
    """Get dataset statistics for AJAX requests."""
    try:
        processed_dir = Path(output_dir) / 'processed'
        
        if not processed_dir.exists():
            return jsonify({'error': 'Results not found'}), 404
        
        stats = {}
        
        # Processed dataset stats
        if (processed_dir / 'patient_processed.csv').exists():
            df = pd.read_csv(processed_dir / 'patient_processed.csv')
            stats['processed'] = {
                'shape': df.shape,
                'columns': list(df.columns),
                'memory_usage': df.memory_usage(deep=True).sum() / 1024 / 1024,  # MB
                'missing_values': df.isnull().sum().sum()
            }
        
        # Pseudonymized dataset stats
        if (processed_dir / 'patient_pseudonymized.csv').exists():
            df = pd.read_csv(processed_dir / 'patient_pseudonymized.csv')
            stats['pseudonymized'] = {
                'shape': df.shape,
                'columns': list(df.columns),
                'memory_usage': df.memory_usage(deep=True).sum() / 1024 / 1024,  # MB
                'missing_values': df.isnull().sum().sum()
            }
        
        return jsonify(stats)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

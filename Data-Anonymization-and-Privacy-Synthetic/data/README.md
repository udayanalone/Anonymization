# Data Description

This directory contains the datasets used in the Data Anonymization and Privacy project.

## Directory Structure

- `raw/`: Contains the original synthetic dataset generated using NumPy
  - `synthetic_data.csv`: Raw synthetic patient data with various health metrics

- `processed/`: Contains the anonymized datasets after applying privacy techniques
  - `anonymized_data.csv`: Dataset after applying anonymization techniques

## Data Description

The synthetic dataset contains the following fields:
- PatientID: Unique identifier for each patient
- Age: Patient age in years
- Gender: Patient gender (Male/Female)
- Pregnancies: Number of pregnancies
- Glucose: Blood glucose level
- BloodPressure: Blood pressure measurement
- SkinThickness: Skin thickness measurement
- Insulin: Insulin level
- BMI: Body Mass Index
- DiabetesPedigreeFunction: Diabetes pedigree function
- Outcome: Binary indicator for diabetes diagnosis

## Preprocessing Steps

1. **Data Generation**: Synthetic data is generated using NumPy with controlled random seed for reproducibility
2. **Data Cleaning**: Basic cleaning to handle missing values and outliers
3. **Anonymization**: Application of various anonymization techniques:
   - k-anonymity
   - l-diversity
   - t-closeness
   - Differential privacy
4. **Validation**: Verification of anonymization effectiveness and utility preservation
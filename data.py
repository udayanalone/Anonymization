import pandas as pd
import numpy as np

# Number of records
N = 2000
np.random.seed(42)  # reproducibility

# Generate synthetic patient data
data = {
    'PatientID': range(1, N+1),
    'Age': np.random.randint(21, 81, size=N),
    'Gender': np.random.choice(['Male', 'Female'], size=N),
    'Pregnancies': np.random.randint(0, 16, size=N),
    'Glucose': np.round(np.random.uniform(70, 200, size=N), 1),
    'BloodPressure': np.round(np.random.uniform(50, 120, size=N), 1),
    'SkinThickness': np.round(np.random.uniform(10, 50, size=N), 1),
    'Insulin': np.round(np.random.uniform(15, 276, size=N), 1),
    'BMI': np.round(np.random.uniform(18, 45, size=N), 1),
    'DiabetesPedigreeFunction': np.round(np.random.uniform(0.1, 2.5, size=N), 2),
    'Outcome': np.random.choice([0, 1], size=N)
}

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv('synthetic_2000_patients.csv', index=False)

print("Synthetic patient data created successfully!")

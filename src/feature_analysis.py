import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

def load_processed_data(file_path):
    """
    Load the preprocessed patient data.
    
    Parameters:
    -----------
    file_path : str
        Path to the processed patient data
        
    Returns:
    --------
    pandas.DataFrame
        Loaded processed data
    """
    try:
        print(f"Loading processed data from {file_path}...")
        df = pd.read_csv(file_path)
        print(f"Successfully loaded {len(df)} records with {len(df.columns)} columns")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def analyze_feature_distributions(df):
    """
    Analyze distributions of numerical and categorical features.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Patient dataset
        
    Returns:
    --------
    dict
        Dictionary containing distribution analysis results
    """
    print("\n=== FEATURE DISTRIBUTION ANALYSIS ===")
    
    # Separate numerical and categorical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    analysis_results = {
        'numerical_stats': {},
        'categorical_stats': {},
        'skewness': {},
        'missing_patterns': {}
    }
    
    # Analyze numerical features
    if numerical_cols:
        print(f"\nNumerical Features ({len(numerical_cols)}):")
        for col in numerical_cols:
            stats = df[col].describe()
            skewness = df[col].skew()
            
            analysis_results['numerical_stats'][col] = {
                'mean': stats['mean'],
                'std': stats['std'],
                'min': stats['min'],
                'max': stats['max'],
                'median': stats['50%']
            }
            analysis_results['skewness'][col] = skewness
            
            print(f"  {col}:")
            print(f"    Mean: {stats['mean']:.2f}, Std: {stats['std']:.2f}")
            print(f"    Range: [{stats['min']:.2f}, {stats['max']:.2f}]")
            print(f"    Skewness: {skewness:.2f}")
    
    # Analyze categorical features
    if categorical_cols:
        print(f"\nCategorical Features ({len(categorical_cols)}):")
        for col in categorical_cols:
            unique_count = df[col].nunique()
            most_common = df[col].mode()[0] if not df[col].mode().empty else 'None'
            most_common_count = df[col].value_counts().iloc[0]
            most_common_pct = (most_common_count / len(df)) * 100
            
            analysis_results['categorical_stats'][col] = {
                'unique_values': unique_count,
                'most_common': most_common,
                'most_common_count': most_common_count,
                'most_common_pct': most_common_pct
            }
            
            print(f"  {col}:")
            print(f"    Unique values: {unique_count}")
            print(f"    Most common: '{most_common}' ({most_common_pct:.1f}%)")
    
    return analysis_results

def analyze_correlations(df, numerical_cols):
    """
    Analyze correlations between numerical features.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Patient dataset
    numerical_cols : list
        List of numerical column names
        
    Returns:
    --------
    pandas.DataFrame
        Correlation matrix
    """
    print("\n=== CORRELATION ANALYSIS ===")
    
    if len(numerical_cols) < 2:
        print("Not enough numerical columns for correlation analysis")
        return None
    
    # Calculate correlation matrix
    corr_matrix = df[numerical_cols].corr()
    
    # Find high correlations
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.7:  # High correlation threshold
                high_corr_pairs.append({
                    'feature1': corr_matrix.columns[i],
                    'feature2': corr_matrix.columns[j],
                    'correlation': corr_val
                })
    
    print(f"Found {len(high_corr_pairs)} high correlation pairs (|r| > 0.7):")
    for pair in high_corr_pairs:
        print(f"  {pair['feature1']} ↔ {pair['feature2']}: {pair['correlation']:.3f}")
    
    return corr_matrix

def perform_feature_importance_analysis(df, target_col=None):
    """
    Perform feature importance analysis using multiple methods.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Patient dataset
    target_col : str, optional
        Target column for supervised feature selection
        
    Returns:
    --------
    dict
        Dictionary containing feature importance results
    """
    print("\n=== FEATURE IMPORTANCE ANALYSIS ===")
    
    # Prepare data for analysis
    df_encoded = df.copy()
    
    # Encode categorical variables
    label_encoders = {}
    for col in df_encoded.select_dtypes(include=['object']).columns:
        if col != target_col:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            label_encoders[col] = le
    
    # If no target column specified, use healthcare expenses as proxy
    if target_col is None:
        if 'HEALTHCARE_EXPENSES' in df_encoded.columns:
            target_col = 'HEALTHCARE_EXPENSES'
        else:
            print("No target column specified and no suitable default found")
            return None
    
    # Remove target column from features
    feature_cols = [col for col in df_encoded.columns if col != target_col]
    X = df_encoded[feature_cols]
    y = df_encoded[target_col]
    
    # Method 1: Statistical tests (F-test for classification, mutual info for regression)
    if y.dtype == 'object' or y.nunique() < 10:
        # Classification problem
        selector_f = SelectKBest(score_func=f_classif, k='all')
        selector_f.fit(X, y)
        f_scores = selector_f.scores_
        f_pvalues = selector_f.pvalues_
        
        selector_mi = SelectKBest(score_func=mutual_info_classif, k='all')
        selector_mi.fit(X, y)
        mi_scores = selector_mi.scores_
        
        print("Using classification-based feature selection methods")
    else:
        # Regression problem - use mutual_info_regression for continuous targets
        from sklearn.feature_selection import mutual_info_regression
        
        selector_f = SelectKBest(score_func=f_classif, k='all')
        selector_f.fit(X, y)
        f_scores = selector_f.scores_
        f_pvalues = selector_f.pvalues_
        
        selector_mi = SelectKBest(score_func=mutual_info_regression, k='all')
        selector_mi.fit(X, y)
        mi_scores = selector_mi.scores_
        
        print("Using regression-based feature selection methods")
    
    # Method 2: Random Forest importance
    if y.dtype == 'object' or y.nunique() < 10:
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
    
    rf.fit(X, y)
    rf_importance = rf.feature_importances_
    
    # Compile results
    feature_importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'F_Score': f_scores,
        'F_PValue': f_pvalues,
        'Mutual_Info': mi_scores,
        'RF_Importance': rf_importance
    })
    
    # Normalize scores to 0-1 range
    for col in ['F_Score', 'Mutual_Info', 'RF_Importance']:
        if feature_importance_df[col].max() > 0:
            feature_importance_df[f'{col}_Normalized'] = feature_importance_df[col] / feature_importance_df[col].max()
        else:
            feature_importance_df[f'{col}_Normalized'] = 0
    
    # Calculate composite importance score
    feature_importance_df['Composite_Score'] = (
        feature_importance_df['F_Score_Normalized'] * 0.3 +
        feature_importance_df['Mutual_Info_Normalized'] * 0.3 +
        feature_importance_df['RF_Importance_Normalized'] * 0.4
    )
    
    # Sort by composite score
    feature_importance_df = feature_importance_df.sort_values('Composite_Score', ascending=False)
    
    print(f"\nTop 10 most important features:")
    for i, row in feature_importance_df.head(10).iterrows():
        print(f"  {row['Feature']}: {row['Composite_Score']:.3f}")
    
    return feature_importance_df

def perform_dimensionality_reduction(df, numerical_cols, n_components=0.95):
    """
    Perform PCA-based dimensionality reduction.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Patient dataset
    numerical_cols : list
        List of numerical column names
    n_components : float, default=0.95
        Fraction of variance to retain
        
    Returns:
    --------
    dict
        Dictionary containing PCA results
    """
    print("\n=== DIMENSIONALITY REDUCTION ANALYSIS ===")
    
    if len(numerical_cols) < 2:
        print("Not enough numerical columns for PCA")
        return None
    
    # Prepare numerical data
    X_numerical = df[numerical_cols].fillna(df[numerical_cols].median())
    
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_numerical)
    
    # Perform PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    # Calculate explained variance
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)
    
    print(f"Original dimensions: {X_scaled.shape[1]}")
    print(f"Reduced dimensions: {X_pca.shape[1]}")
    print(f"Variance retained: {cumulative_variance[-1]*100:.1f}%")
    
    # Show top components
    print(f"\nTop 5 principal components:")
    for i in range(min(5, len(explained_variance_ratio))):
        print(f"  PC{i+1}: {explained_variance_ratio[i]*100:.1f}% variance")
    
    # Feature contributions to principal components
    feature_contributions = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(pca.n_components_)],
        index=numerical_cols
    )
    
    return {
        'pca': pca,
        'explained_variance': explained_variance_ratio,
        'cumulative_variance': cumulative_variance,
        'feature_contributions': feature_contributions,
        'transformed_data': X_pca
    }

def select_critical_features(df, feature_importance_df, importance_threshold=0.1, max_features=15):
    """
    Select critical features based on importance analysis.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Patient dataset
    feature_importance_df : pandas.DataFrame
        Feature importance results
    importance_threshold : float, default=0.1
        Minimum importance score to consider
    max_features : int, default=15
        Maximum number of features to select
        
    Returns:
    --------
    dict
        Dictionary containing selected features and rationale
    """
    print("\n=== CRITICAL FEATURE SELECTION ===")
    
    # Filter features by importance threshold
    critical_features = feature_importance_df[
        feature_importance_df['Composite_Score'] >= importance_threshold
    ]
    
    # Limit to max_features
    if len(critical_features) > max_features:
        critical_features = critical_features.head(max_features)
    
    selected_features = critical_features['Feature'].tolist()
    
    print(f"Selected {len(selected_features)} critical features:")
    for i, feature in enumerate(selected_features, 1):
        importance_score = critical_features[
            critical_features['Feature'] == feature
        ]['Composite_Score'].iloc[0]
        print(f"  {i:2d}. {feature:<20} (Score: {importance_score:.3f})")
    
    # Categorize selected features
    feature_categories = {
        'identifiers': [],
        'sensitive': [],
        'demographics': [],
        'geographic': [],
        'financial': []
    }
    
    for feature in selected_features:
        feature_lower = feature.lower()
        
        if any(keyword in feature_lower for keyword in ['id', 'uuid']):
            feature_categories['identifiers'].append(feature)
        elif any(keyword in feature_lower for keyword in ['name', 'ssn', 'birth', 'address']):
            feature_categories['sensitive'].append(feature)
        elif any(keyword in feature_lower for keyword in ['age', 'gender', 'race', 'ethnicity', 'marital']):
            feature_categories['demographics'].append(feature)
        elif any(keyword in feature_lower for keyword in ['city', 'state', 'county', 'lat', 'lon', 'zip']):
            feature_categories['geographic'].append(feature)
        elif any(keyword in feature_lower for keyword in ['expense', 'coverage', 'cost']):
            feature_categories['financial'].append(feature)
    
    print(f"\nFeature categorization:")
    for category, features in feature_categories.items():
        if features:
            print(f"  {category.capitalize()}: {', '.join(features)}")
    
    return {
        'selected_features': selected_features,
        'feature_categories': feature_categories,
        'importance_scores': critical_features.set_index('Feature')['Composite_Score'].to_dict()
    }

def select_balanced_features(df, feature_importance_df, privacy_level='medium'):
    """
    Select features using a balanced approach considering privacy, utility, and research value.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Patient dataset
    feature_importance_df : pandas.DataFrame
        Feature importance results
    privacy_level : str, default='medium'
        Privacy level: 'high', 'medium', 'low'
        
    Returns:
    --------
    dict
        Dictionary containing selected features and rationale
    """
    print("\n=== BALANCED FEATURE SELECTION ===")
    
    # Define feature categories with privacy levels
    feature_categories = {
        'high_risk': ['Id', 'SSN', 'DRIVERS', 'PASSPORT', 'ADDRESS', 'BIRTHDATE'],
        'medium_risk': ['FIRST', 'LAST', 'MAIDEN', 'BIRTHPLACE', 'ZIP', 'LAT', 'LON'],
        'low_risk': ['PREFIX', 'SUFFIX', 'MARITAL', 'RACE', 'ETHNICITY', 'GENDER', 'CITY', 'STATE', 'COUNTY'],
        'research_valuable': ['HEALTHCARE_EXPENSES', 'HEALTHCARE_COVERAGE', 'DEATHDATE']
    }
    
    # Select features based on privacy level
    if privacy_level == 'high':
        # High privacy: remove all high and medium risk features
        features_to_remove = feature_categories['high_risk'] + feature_categories['medium_risk']
        selected_features = [col for col in df.columns if col not in features_to_remove]
        print("Privacy Level: HIGH - Removing all high and medium risk features")
        
    elif privacy_level == 'medium':
        # Medium privacy: remove high risk features, keep some medium risk
        features_to_remove = feature_categories['high_risk']
        # Keep only some medium risk features (geographic info for research)
        medium_keep = ['ZIP', 'LAT', 'LON', 'CITY', 'STATE', 'COUNTY']
        features_to_remove = [f for f in features_to_remove if f not in medium_keep]
        selected_features = [col for col in df.columns if col not in features_to_remove]
        print("Privacy Level: MEDIUM - Removing high risk features, keeping some geographic info")
        
    else:  # low privacy
        # Low privacy: remove only highest risk features
        features_to_remove = ['Id', 'SSN', 'DRIVERS', 'PASSPORT']
        selected_features = [col for col in df.columns if col not in features_to_remove]
        print("Privacy Level: LOW - Removing only highest risk identifiers")
    
    # Ensure we have enough features for analysis
    min_features = 8
    if len(selected_features) < min_features:
        # Add some important features back
        important_features = feature_importance_df.head(min_features - len(selected_features))['Feature'].tolist()
        for feature in important_features:
            if feature not in selected_features:
                selected_features.append(feature)
        print(f"Added {min_features - len(selected_features)} important features to meet minimum requirement")
    
    # Categorize selected features
    categorized_features = {
        'identifiers': [],
        'sensitive': [],
        'demographics': [],
        'geographic': [],
        'financial': [],
        'research': []
    }
    
    for feature in selected_features:
        feature_lower = feature.lower()
        
        if any(keyword in feature_lower for keyword in ['id', 'uuid']):
            categorized_features['identifiers'].append(feature)
        elif any(keyword in feature_lower for keyword in ['name', 'ssn', 'birth', 'address']):
            categorized_features['sensitive'].append(feature)
        elif any(keyword in feature_lower for keyword in ['age', 'gender', 'race', 'ethnicity', 'marital']):
            categorized_features['demographics'].append(feature)
        elif any(keyword in feature_lower for keyword in ['city', 'state', 'county', 'lat', 'lon', 'zip']):
            categorized_features['geographic'].append(feature)
        elif any(keyword in feature_lower for keyword in ['expense', 'coverage', 'cost']):
            categorized_features['financial'].append(feature)
        else:
            categorized_features['research'].append(feature)
    
    print(f"\nSelected {len(selected_features)} features:")
    for i, feature in enumerate(selected_features, 1):
        importance_score = feature_importance_df[
            feature_importance_df['Feature'] == feature
        ]['Composite_Score'].iloc[0] if feature in feature_importance_df['Feature'].values else 0.0
        print(f"  {i:2d}. {feature:<20} (Importance: {importance_score:.3f})")
    
    print(f"\nFeature categorization:")
    for category, features in categorized_features.items():
        if features:
            print(f"  {category.capitalize()}: {', '.join(features)}")
    
    return {
        'selected_features': selected_features,
        'feature_categories': categorized_features,
        'privacy_level': privacy_level,
        'features_removed': len(df.columns) - len(selected_features)
    }

def save_selected_features_dataset(df, selected_features, output_path):
    """
    Save a dataset containing only the selected features.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Original dataset
    selected_features : list
        List of selected feature names
    output_path : str
        Path to save the filtered dataset
        
    Returns:
    --------
    pandas.DataFrame
        Filtered dataset with selected features
    """
    print(f"\n=== SAVING SELECTED FEATURES DATASET ===")
    
    # Filter dataset to selected features
    df_selected = df[selected_features].copy()
    
    # Save to file
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_selected.to_csv(output_path, index=False)
    
    print(f"Selected features dataset saved to: {output_path}")
    print(f"Dataset shape: {df_selected.shape}")
    print(f"Features included: {', '.join(selected_features)}")
    
    return df_selected

def generate_feature_analysis_report(file_path, output_dir="results/reports"):
    """
    Generate comprehensive feature analysis report.
    
    Parameters:
    -----------
    file_path : str
        Path to the processed patient data file
    output_dir : str
        Directory to save the report
        
    Returns:
    --------
    dict
        Complete feature analysis results
    """
    print("=" * 60)
    print("COMPREHENSIVE FEATURE ANALYSIS")
    print("=" * 60)
    
    # Load the data first
    df = load_processed_data(file_path)
    if df is None:
        print("Failed to load data!")
        return None
    
    # Ensure output directory exists
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Distribution analysis
    distribution_results = analyze_feature_distributions(df)
    
    # Step 2: Correlation analysis
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    correlation_results = analyze_correlations(df, numerical_cols)
    
    # Step 3: Feature importance analysis
    feature_importance = perform_feature_importance_analysis(df)
    
    # Step 4: Dimensionality reduction
    pca_results = perform_dimensionality_reduction(df, numerical_cols)
    
    # Step 5: Critical feature selection
    critical_features = select_critical_features(df, feature_importance)
    
    # Step 6: Balanced feature selection
    balanced_features = select_balanced_features(df, feature_importance, privacy_level='medium')
    
    # Compile complete results
    complete_results = {
        'distribution_analysis': distribution_results,
        'correlation_analysis': correlation_results,
        'feature_importance': feature_importance,
        'pca_results': pca_results,
        'critical_features': critical_features,
        'balanced_features': balanced_features,
        'dataset_info': {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.to_dict()
        }
    }
    
    # Save results
    report_file = os.path.join(output_dir, "feature_analysis_report.txt")
    with open(report_file, 'w') as f:
        f.write("FEATURE ANALYSIS REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Dataset: {df.shape[0]} records, {df.shape[1]} columns\n\n")
        
        f.write("CRITICAL FEATURES (High Importance):\n")
        for feature in critical_features['selected_features']:
            f.write(f"- {feature}\n")
        
        f.write(f"\nBALANCED FEATURE SELECTION (Privacy Level: {balanced_features['privacy_level'].upper()}):\n")
        for feature in balanced_features['selected_features']:
            f.write(f"- {feature}\n")
        
        f.write(f"\nFeature Summary:\n")
        f.write(f"- Critical features: {len(critical_features['selected_features'])}\n")
        f.write(f"- Balanced selection: {len(balanced_features['selected_features'])}\n")
        f.write(f"- Features removed: {balanced_features['features_removed']}\n")
        f.write(f"- Privacy level: {balanced_features['privacy_level']}\n")
    
    print(f"\nFeature analysis report saved to: {report_file}")
    
    # Save the balanced features dataset for further processing
    balanced_dataset_path = os.path.join(os.path.dirname(output_dir), "..", "data", "processed", "patient_selected_features.csv")
    save_selected_features_dataset(df, balanced_features['selected_features'], balanced_dataset_path)
    
    print("\n" + "=" * 60)
    print("FEATURE ANALYSIS COMPLETED SUCCESSFULLY")
    print("=" * 60)
    
    return complete_results

if __name__ == "__main__":
    # Example usage
    input_file = "data/processed/patient_processed.csv"
    
    # Load data
    df = load_processed_data(input_file)
    if df is not None:
        # Perform complete feature analysis
        results = generate_feature_analysis_report(df)
        print(f"\nFeature analysis completed! Selected {len(results['critical_features']['selected_features'])} critical features.")
    else:
        print("Failed to load data!")

"""
Data Preprocessing Module
=========================

This script handles data cleaning and preprocessing operations:
- Handle missing/null values
- Remove duplicates
- Convert categorical data (LabelEncoder/OneHotEncoder)
- Normalize numerical features (StandardScaler)
- Split dataset into training and testing sets (80-20 ratio)

Author: ML Engineer
Date: 2025
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from pathlib import Path
import joblib


def clean_data(df, remove_duplicates=True, handle_missing='median'):
    """
    Clean the dataset by handling missing values and removing duplicates.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame to clean
    remove_duplicates : bool, default=True
        If True, remove duplicate rows
    handle_missing : str, default='median'
        Strategy for handling missing values:
        - 'drop': Drop rows with missing values
        - 'mean': Fill with mean (for numerical columns)
        - 'median': Fill with median (for numerical columns)
        - 'mode': Fill with mode (for categorical columns)
        - 'forward': Forward fill
        - 'backward': Backward fill
    
    Returns:
    --------
    pd.DataFrame
        Cleaned DataFrame
    """
    print(f"\n{'='*60}")
    print("DATA CLEANING")
    print(f"{'='*60}")
    print(f"Original shape: {df.shape}")
    
    # Create a copy to avoid modifying original
    df_clean = df.copy()
    
    # Check for missing values
    missing_count = df_clean.isnull().sum()
    total_missing = missing_count.sum()
    
    if total_missing > 0:
        print(f"\nMissing values found:")
        print("-" * 60)
        for col, count in missing_count[missing_count > 0].items():
            print(f"  {col}: {count} ({count/len(df_clean)*100:.2f}%)")
        
        # Handle missing values based on strategy
        print(f"\nHandling missing values using '{handle_missing}' strategy...")
        
        for col in df_clean.columns:
            if df_clean[col].isnull().sum() > 0:
                if df_clean[col].dtype in ['int64', 'float64']:
                    # Numerical columns
                    if handle_missing == 'mean':
                        df_clean[col].fillna(df_clean[col].mean(), inplace=True)
                    elif handle_missing == 'median':
                        df_clean[col].fillna(df_clean[col].median(), inplace=True)
                    elif handle_missing == 'drop':
                        df_clean = df_clean.dropna(subset=[col])
                    elif handle_missing == 'forward':
                        df_clean[col].ffill(inplace=True)
                    elif handle_missing == 'backward':
                        df_clean[col].bfill(inplace=True)
                else:
                    # Categorical columns
                    if handle_missing == 'mode':
                        df_clean[col].fillna(df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'Unknown', inplace=True)
                    elif handle_missing == 'drop':
                        df_clean = df_clean.dropna(subset=[col])
                    elif handle_missing == 'forward':
                        df_clean[col].ffill(inplace=True)
                    elif handle_missing == 'backward':
                        df_clean[col].bfill(inplace=True)
        
        print("Missing values handled.")
    else:
        print("\nNo missing values found.")
    
    # Remove duplicates
    if remove_duplicates:
        duplicates_before = df_clean.shape[0]
        df_clean = df_clean.drop_duplicates()
        duplicates_removed = duplicates_before - df_clean.shape[0]
        
        if duplicates_removed > 0:
            print(f"\nRemoved {duplicates_removed} duplicate rows.")
        else:
            print("\nNo duplicate rows found.")
    
    print(f"Cleaned shape: {df_clean.shape}")
    print(f"{'='*60}\n")
    
    return df_clean


def identify_columns(df, target_col=None):
    """
    Identify numerical and categorical columns in the dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    target_col : str, optional
        Name of the target column (to exclude from feature identification)
    
    Returns:
    --------
    tuple
        (numerical_cols, categorical_cols) - Lists of column names
    """
    if target_col and target_col in df.columns:
        df_features = df.drop(columns=[target_col])
    else:
        df_features = df
    
    numerical_cols = df_features.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df_features.select_dtypes(include=['object', 'category']).columns.tolist()
    
    return numerical_cols, categorical_cols


def encode_categorical(df, categorical_cols=None, encoding_type='label', target_col=None):
    """
    Encode categorical variables using LabelEncoder or OneHotEncoder.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    categorical_cols : list, optional
        List of categorical column names. If None, auto-detects.
    encoding_type : str, default='label'
        'label' for LabelEncoder (ordinal encoding)
        'onehot' for OneHotEncoder (one-hot encoding)
    target_col : str, optional
        Target column name (always uses LabelEncoder)
    
    Returns:
    --------
    tuple
        (encoded_df, encoders_dict) - Encoded DataFrame and dictionary of encoders
    """
    print(f"\n{'='*60}")
    print("CATEGORICAL ENCODING")
    print(f"{'='*60}")
    
    df_encoded = df.copy()
    encoders = {}
    
    # Identify categorical columns if not provided
    if categorical_cols is None:
        _, categorical_cols = identify_columns(df, target_col)
    
    # Encode target column with LabelEncoder (always)
    if target_col and target_col in df_encoded.columns:
        if df_encoded[target_col].dtype == 'object':
            print(f"Encoding target column '{target_col}' with LabelEncoder...")
            le = LabelEncoder()
            df_encoded[target_col] = le.fit_transform(df_encoded[target_col])
            encoders[target_col] = le
            print(f"  Classes: {len(le.classes_)} unique values")
    
    # Encode feature columns
    for col in categorical_cols:
        if col == target_col:
            continue
            
        if df_encoded[col].dtype == 'object':
            print(f"Encoding '{col}' with {encoding_type.upper()}...")
            
            if encoding_type == 'label':
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                encoders[col] = le
                print(f"  Classes: {len(le.classes_)} unique values")
            
            elif encoding_type == 'onehot':
                ohe = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
                encoded_array = ohe.fit_transform(df_encoded[[col]])
                
                # Create column names for one-hot encoded features
                feature_names = [f"{col}_{cat}" for cat in ohe.categories_[0][1:]]
                
                # Create DataFrame with one-hot encoded columns
                df_encoded_dummy = pd.DataFrame(encoded_array, columns=feature_names, index=df_encoded.index)
                
                # Drop original column and add encoded columns
                df_encoded = df_encoded.drop(columns=[col])
                df_encoded = pd.concat([df_encoded, df_encoded_dummy], axis=1)
                
                encoders[col] = ohe
                print(f"  Created {len(feature_names)} one-hot encoded columns")
    
    print(f"{'='*60}\n")
    
    return df_encoded, encoders


def scale_features(df, numerical_cols=None, scaler=None, fit=True, target_col=None):
    """
    Normalize numerical features using StandardScaler.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    numerical_cols : list, optional
        List of numerical column names to scale. If None, auto-detects.
    scaler : StandardScaler, optional
        Pre-fitted scaler. If None, creates a new one.
    fit : bool, default=True
        If True, fit the scaler on the data
    target_col : str, optional
        Name of target column to exclude from scaling
    
    Returns:
    --------
    tuple
        (scaled_df, scaler) - Scaled DataFrame and fitted scaler
    """
    print(f"\n{'='*60}")
    print("FEATURE SCALING (StandardScaler)")
    print(f"{'='*60}")
    
    df_scaled = df.copy()
    
    # Identify numerical columns if not provided
    if numerical_cols is None:
        numerical_cols, _ = identify_columns(df, target_col=target_col)
    
    # Exclude target column from scaling if provided
    if target_col and target_col in numerical_cols:
        numerical_cols = [col for col in numerical_cols if col != target_col]
        print(f"Excluding target column '{target_col}' from scaling.")
    
    if len(numerical_cols) == 0:
        print("No numerical columns found to scale.")
        return df_scaled, None
    
    print(f"Scaling {len(numerical_cols)} numerical columns: {numerical_cols}")
    
    # Create or use provided scaler
    if scaler is None:
        scaler = StandardScaler()
    
    # Scale the numerical columns
    if fit:
        df_scaled[numerical_cols] = scaler.fit_transform(df[numerical_cols])
        print("Scaler fitted and applied.")
    else:
        df_scaled[numerical_cols] = scaler.transform(df[numerical_cols])
        print("Scaler applied (using pre-fitted scaler).")
    
    print(f"{'='*60}\n")
    
    return df_scaled, scaler


def split_data(df, target_col, test_size=0.2, random_state=42):
    """
    Split dataset into training and testing sets.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    target_col : str
        Name of the target column
    test_size : float, default=0.2
        Proportion of dataset to include in the test split (20%)
    random_state : int, default=42
        Random seed for reproducibility
    
    Returns:
    --------
    tuple
        (X_train, X_test, y_train, y_test) - Split datasets
    """
    print(f"\n{'='*60}")
    print("TRAIN-TEST SPLIT")
    print(f"{'='*60}")
    
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame")
    
    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Test size: {test_size*100}%")
    print(f"Random state: {random_state}")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y if y.dtype == 'int64' or y.nunique() < 50 else None  # Stratify for classification
    )
    
    print(f"\nTraining set: {X_train.shape[0]} samples ({X_train.shape[0]/len(df)*100:.1f}%)")
    print(f"Test set: {X_test.shape[0]} samples ({X_test.shape[0]/len(df)*100:.1f}%)")
    print(f"{'='*60}\n")
    
    return X_train, X_test, y_train, y_test


def preprocess_pipeline(df, target_col, handle_missing='median', 
                       remove_duplicates=True, encoding_type='label',
                       scale_features_flag=True, test_size=0.2, random_state=42):
    """
    Complete preprocessing pipeline combining all steps.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input raw DataFrame
    target_col : str
        Name of the target column
    handle_missing : str, default='median'
        Strategy for handling missing values
    remove_duplicates : bool, default=True
        Whether to remove duplicate rows
    encoding_type : str, default='label'
        'label' or 'onehot' for categorical encoding
    scale_features_flag : bool, default=True
        Whether to scale numerical features
    test_size : float, default=0.2
        Proportion for test split
    random_state : int, default=42
        Random seed
    
    Returns:
    --------
    dict
        Dictionary containing all preprocessed data and objects
    """
    print("\n" + "="*60)
    print("COMPLETE PREPROCESSING PIPELINE")
    print("="*60)
    
    # Step 1: Clean data
    df_clean = clean_data(df, remove_duplicates=remove_duplicates, 
                         handle_missing=handle_missing)
    
    # Step 2: Encode categorical data
    df_encoded, encoders = encode_categorical(df_clean, encoding_type=encoding_type, 
                                             target_col=target_col)
    
    # Step 3: Scale numerical features (excluding target column)
    if scale_features_flag:
        df_scaled, scaler = scale_features(df_encoded, target_col=target_col)
    else:
        df_scaled = df_encoded
        scaler = None
    
    # Step 4: Split data
    X_train, X_test, y_train, y_test = split_data(df_scaled, target_col, 
                                                  test_size=test_size, 
                                                  random_state=random_state)
    
    # Prepare return dictionary
    result = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'encoders': encoders,
        'scaler': scaler,
        'df_processed': df_scaled
    }
    
    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE!")
    print("="*60 + "\n")
    
    return result


if __name__ == "__main__":
    """
    Example usage of the preprocessing module.
    """
    # Import load_data module
    import sys
    from pathlib import Path
    
    # Add scripts directory to path
    scripts_dir = Path(__file__).parent
    sys.path.append(str(scripts_dir))
    
    try:
        from load_data import load_dataset
        
        print("="*60)
        print("Data Preprocessing - Example Usage")
        print("="*60)
        
        # Load dataset
        df = load_dataset('crop_recommendation.csv', display_info=False)
        
        # Run complete preprocessing pipeline
        preprocessed = preprocess_pipeline(
            df=df,
            target_col='label',  # Adjust based on your dataset
            handle_missing='median',
            remove_duplicates=True,
            encoding_type='label',
            scale_features_flag=True,
            test_size=0.2,
            random_state=42
        )
        
        # Display summary
        print("\n" + "="*60)
        print("PREPROCESSING SUMMARY")
        print("="*60)
        print(f"Training features shape: {preprocessed['X_train'].shape}")
        print(f"Testing features shape: {preprocessed['X_test'].shape}")
        print(f"Training target shape: {preprocessed['y_train'].shape}")
        print(f"Testing target shape: {preprocessed['y_test'].shape}")
        print(f"Number of encoders saved: {len(preprocessed['encoders'])}")
        print(f"Scaler fitted: {preprocessed['scaler'] is not None}")
        
    except ImportError:
        print("Note: load_data module not found. Using standalone preprocessing functions.")
        print("Please import this module and use the functions directly in your scripts.")


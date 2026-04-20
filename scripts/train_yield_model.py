"""
Crop Yield Prediction Model Training
====================================

This script trains multiple regression models for crop yield prediction,
compares their performance, and saves the best model.

Models tested:
- RandomForestRegressor
- XGBRegressor
- LinearRegression

Author: ML Engineer
Date: 2025
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from pathlib import Path
import joblib
import warnings
from time import time

warnings.filterwarnings('ignore')

# Try to import XGBoost
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("[WARNING] XGBoost not available. Install with: pip install xgboost")


def get_project_paths():
    """Get paths to important project directories."""
    project_root = Path(__file__).parent.parent
    return {
        'models_dir': project_root / 'models',
        'data_dir': project_root / 'data',
        'results_dir': project_root / 'results',
        'metrics_dir': project_root / 'results' / 'metrics'
    }


def prepare_yield_dataset(df, target_col='yield'):
    """
    Prepare dataset for yield prediction. If yield column doesn't exist,
    calculate it from Production/Area if available.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    target_col : str, default='yield'
        Name of the target column
    
    Returns:
    --------
    pd.DataFrame
        Prepared DataFrame with yield column
    """
    df_prepared = df.copy()
    
    # If yield column doesn't exist, try to calculate from Production/Area
    if target_col not in df_prepared.columns:
        if 'Production' in df_prepared.columns and 'Area' in df_prepared.columns:
            print(f"Calculating yield from Production/Area...")
            # Filter out zero areas to avoid division by zero
            df_prepared = df_prepared[df_prepared['Area'] > 0].copy()
            df_prepared[target_col] = df_prepared['Production'] / df_prepared['Area']
            print(f"Yield calculated. Mean yield: {df_prepared[target_col].mean():.2f}")
        else:
            raise ValueError(f"Target column '{target_col}' not found and cannot be calculated.")
    
    # Remove rows with invalid yield values
    df_prepared = df_prepared[df_prepared[target_col].notna()].copy()
    df_prepared = df_prepared[df_prepared[target_col] > 0].copy()
    
    return df_prepared


def handle_outliers(df, numerical_cols, method='iqr'):
    """
    Handle outliers using IQR method or Z-score.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    numerical_cols : list
        List of numerical column names
    method : str, default='iqr'
        'iqr' for Interquartile Range method
        'zscore' for Z-score method
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with outliers handled
    """
    df_clean = df.copy()
    outliers_removed = 0
    
    print(f"\n[INFO] Handling outliers using {method.upper()} method...")
    
    for col in numerical_cols:
        if col not in df_clean.columns:
            continue
        
        before_count = len(df_clean)
        
        if method == 'iqr':
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
        
        elif method == 'zscore':
            try:
                from scipy import stats
                z_scores = np.abs(stats.zscore(df_clean[col]))
                df_clean = df_clean[z_scores < 3]
            except ImportError:
                print(f"  Warning: scipy not available, skipping Z-score for {col}")
                continue
        
        after_count = len(df_clean)
        removed = before_count - after_count
        outliers_removed += removed
        
        if removed > 0:
            print(f"  {col}: Removed {removed} outliers")
    
    print(f"[OK] Total outliers removed: {outliers_removed}")
    print(f"[OK] Dataset size after outlier removal: {len(df_clean)} rows")
    
    return df_clean


def load_and_preprocess_data(data_path='data/crop_production.csv', target_col='yield'):
    """
    Load and preprocess the crop yield dataset.
    
    Parameters:
    -----------
    data_path : str
        Path to dataset file
    target_col : str, default='yield'
        Name of target column
    
    Returns:
    --------
    dict
        Dictionary containing preprocessed data and preprocessing objects
    """
    print("="*70)
    print("LOADING AND PREPROCESSING DATA")
    print("="*70)
    
    paths = get_project_paths()
    # Handle both relative and absolute paths
    if Path(data_path).is_absolute():
        file_path = Path(data_path)
    else:
        # If it's a relative path, check if it already includes 'data/' or just the filename
        if data_path.startswith('data/'):
            file_path = paths['data_dir'].parent / data_path
        else:
            file_path = paths['data_dir'] / data_path
    
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")
    
    print(f"\n[INFO] Loading dataset: {file_path}")
    df = pd.read_csv(file_path)
    print(f"[OK] Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Prepare yield dataset
    df = prepare_yield_dataset(df, target_col=target_col)
    
    # Handle missing values
    print(f"\n[INFO] Handling missing values...")
    missing_before = df.isnull().sum().sum()
    if missing_before > 0:
        # Fill numerical columns with median
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for col in numerical_cols:
            if col != target_col:
                df[col].fillna(df[col].median(), inplace=True)
        
        # Fill categorical columns with mode
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        for col in categorical_cols:
            if col != target_col:
                mode_val = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
                df[col].fillna(mode_val, inplace=True)
        
        print(f"[OK] Handled {missing_before} missing values")
    else:
        print("[OK] No missing values found")
    
    # Remove duplicates
    duplicates_before = len(df)
    df = df.drop_duplicates()
    duplicates_removed = duplicates_before - len(df)
    if duplicates_removed > 0:
        print(f"[OK] Removed {duplicates_removed} duplicate rows")
    
    # Handle outliers for numerical columns (excluding target)
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numerical_cols:
        numerical_cols.remove(target_col)
    
    if len(numerical_cols) > 0:
        df = handle_outliers(df, numerical_cols, method='iqr')
    
    # Separate features and target
    feature_cols = [col for col in df.columns if col != target_col]
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    print(f"\n[INFO] Features: {len(feature_cols)} columns")
    print(f"[INFO] Target: {target_col}")
    print(f"[INFO] Features: {feature_cols}")
    
    # Encode categorical variables
    print(f"\n[INFO] Encoding categorical features...")
    encoders = {}
    X_encoded = X.copy()
    
    for col in X.columns:
        if X[col].dtype == 'object':
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X[col].astype(str))
            encoders[col] = le
            print(f"  {col}: Encoded {len(le.classes_)} unique values")
    
    # Scale numerical features
    print(f"\n[INFO] Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_encoded)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X_encoded.columns)
    print(f"[OK] Features scaled")
    
    # Split data (80/20)
    print(f"\n[INFO] Splitting data into train/test (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled_df, y,
        test_size=0.2,
        random_state=42
    )
    print(f"[OK] Training set: {X_train.shape[0]} samples")
    print(f"[OK] Test set: {X_test.shape[0]} samples")
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'encoders': encoders,
        'scaler': scaler,
        'feature_names': list(X_scaled_df.columns)
    }


def train_random_forest(X_train, y_train, **kwargs):
    """Train RandomForestRegressor."""
    print("\n[INFO] Training RandomForestRegressor...")
    model = RandomForestRegressor(
        n_estimators=kwargs.get('n_estimators', 100),
        random_state=kwargs.get('random_state', 42),
        n_jobs=-1,
        verbose=0,
        **{k: v for k, v in kwargs.items() if k not in ['n_estimators', 'random_state']}
    )
    start_time = time()
    model.fit(X_train, y_train)
    elapsed = time() - start_time
    print(f"[OK] Training completed in {elapsed:.2f} seconds")
    return model


def train_xgboost(X_train, y_train, **kwargs):
    """Train XGBRegressor."""
    if not XGBOOST_AVAILABLE:
        return None
    
    print("\n[INFO] Training XGBRegressor...")
    model = XGBRegressor(
        n_estimators=kwargs.get('n_estimators', 100),
        random_state=kwargs.get('random_state', 42),
        n_jobs=-1,
        eval_metric='rmse',
        **{k: v for k, v in kwargs.items() if k not in ['n_estimators', 'random_state']}
    )
    start_time = time()
    model.fit(X_train, y_train)
    elapsed = time() - start_time
    print(f"[OK] Training completed in {elapsed:.2f} seconds")
    return model


def train_linear_regression(X_train, y_train, **kwargs):
    """Train LinearRegression."""
    print("\n[INFO] Training LinearRegression...")
    model = LinearRegression(**kwargs)
    start_time = time()
    model.fit(X_train, y_train)
    elapsed = time() - start_time
    print(f"[OK] Training completed in {elapsed:.2f} seconds")
    return model


def evaluate_regression_model(model, X_test, y_test, model_name='Model'):
    """
    Evaluate a regression model.
    
    Returns:
    --------
    dict
        Dictionary containing evaluation metrics
    """
    print(f"\n[INFO] Evaluating {model_name}...")
    
    y_pred = model.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    metrics = {
        'r2_score': r2,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'y_pred': y_pred
    }
    
    print(f"  R² Score: {r2:.4f}")
    print(f"  MAE:      {mae:.4f}")
    print(f"  RMSE:     {rmse:.4f}")
    
    return metrics


def compare_models(models_dict, X_test, y_test):
    """
    Compare all trained models and find the best one.
    
    Returns:
    --------
    dict
        Best model information
    """
    print("\n" + "="*70)
    print("COMPARING MODELS")
    print("="*70)
    
    model_results = []
    
    for model_name, model in models_dict.items():
        if model is None:
            continue
        
        metrics = evaluate_regression_model(model, X_test, y_test, model_name)
        
        model_results.append({
            'name': model_name,
            'model': model,
            'r2_score': metrics['r2_score'],
            'mae': metrics['mae'],
            'rmse': metrics['rmse'],
            'metrics': metrics
        })
    
    # Sort by R² score (primary metric)
    model_results.sort(key=lambda x: x['r2_score'], reverse=True)
    
    # Display comparison
    print("\n" + "-"*70)
    print(f"{'Model':<30} {'R² Score':<15} {'MAE':<15} {'RMSE':<15}")
    print("-"*70)
    for result in model_results:
        print(f"{result['name']:<30} {result['r2_score']:<15.4f} "
              f"{result['mae']:<15.4f} {result['rmse']:<15.4f}")
    
    # Best model is the one with highest R² score
    best_model_info = model_results[0]
    
    print("\n" + "="*70)
    print("BEST MODEL SELECTED")
    print("="*70)
    print(f"\nModel: {best_model_info['name']}")
    print(f"R² Score: {best_model_info['r2_score']:.4f}")
    print(f"MAE:      {best_model_info['mae']:.4f}")
    print(f"RMSE:     {best_model_info['rmse']:.4f}")
    
    return best_model_info, model_results


def save_best_model(best_model_info, encoders, scaler, feature_names,
                   save_path='models/best_yield_model.pkl'):
    """
    Save the best model with preprocessing objects.
    """
    paths = get_project_paths()
    if save_path is None:
        save_path = paths['models_dir'] / 'best_yield_model.pkl'
    else:
        save_path = Path(save_path)
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    model_data = {
        'model': best_model_info['model'],
        'encoders': encoders,
        'scaler': scaler,
        'model_name': best_model_info['name'],
        'r2_score': best_model_info['r2_score'],
        'mae': best_model_info['mae'],
        'rmse': best_model_info['rmse'],
        'feature_names': feature_names
    }
    
    joblib.dump(model_data, save_path)
    print(f"\n[OK] Best model saved to: {save_path}")


def save_evaluation_metrics(best_model_info, all_results, save_path=None):
    """
    Save evaluation metrics to a text file.
    """
    paths = get_project_paths()
    if save_path is None:
        save_path = paths['metrics_dir'] / 'yield_model_evaluation.txt'
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("CROP YIELD PREDICTION - MODEL COMPARISON RESULTS\n")
        f.write("="*70 + "\n\n")
        
        f.write("BEST MODEL\n")
        f.write("-"*70 + "\n")
        f.write(f"Model Name: {best_model_info['name']}\n")
        f.write(f"R² Score:   {best_model_info['r2_score']:.4f}\n")
        f.write(f"MAE:        {best_model_info['mae']:.4f}\n")
        f.write(f"RMSE:       {best_model_info['rmse']:.4f}\n\n")
        
        f.write("ALL MODELS COMPARISON\n")
        f.write("-"*70 + "\n")
        f.write(f"{'Model':<30} {'R² Score':<15} {'MAE':<15} {'RMSE':<15}\n")
        f.write("-"*70 + "\n")
        
        for result in all_results:
            f.write(f"{result['name']:<30} {result['r2_score']:<15.4f} "
                   f"{result['mae']:<15.4f} {result['rmse']:<15.4f}\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("Note: Models are ranked by R² Score (higher is better)\n")
        f.write("="*70 + "\n")
    
    print(f"[OK] Evaluation metrics saved to: {save_path}")


def print_final_metrics(best_model_info, all_results):
    """Print final evaluation metrics clearly."""
    print("\n" + "="*70)
    print("FINAL EVALUATION METRICS")
    print("="*70)
    
    print(f"\n[WINNER] Best Model: {best_model_info['name']}")
    print("-"*70)
    print(f"R² Score (Coefficient of Determination): {best_model_info['r2_score']:.4f}")
    print(f"  -> {best_model_info['r2_score']*100:.2f}% of variance in yield is explained by the model")
    print(f"\nMAE (Mean Absolute Error):              {best_model_info['mae']:.4f}")
    print(f"  -> Average absolute difference between actual and predicted yield")
    print(f"\nRMSE (Root Mean Squared Error):        {best_model_info['rmse']:.4f}")
    print(f"  -> Standard deviation of prediction errors (in same units as target)")
    
    print("\n" + "-"*70)
    print("ALL MODELS PERFORMANCE")
    print("-"*70)
    print(f"\n{'Model':<30} {'R² Score':<15} {'MAE':<15} {'RMSE':<15}")
    print("-"*70)
    for result in all_results:
        print(f"{result['name']:<30} {result['r2_score']:<15.4f} "
              f"{result['mae']:<15.4f} {result['rmse']:<15.4f}")
    
    print("="*70 + "\n")


def main():
    """
    Main training pipeline for crop yield prediction models.
    """
    print("\n" + "="*70)
    print("CROP YIELD PREDICTION - MULTI-MODEL TRAINING")
    print("="*70)
    
    start_time = time()
    
    try:
        # Load and preprocess data
        preprocessed = load_and_preprocess_data(
            data_path='data/crop_production.csv',
            target_col='yield'
        )
        
        X_train = preprocessed['X_train']
        X_test = preprocessed['X_test']
        y_train = preprocessed['y_train']
        y_test = preprocessed['y_test']
        encoders = preprocessed['encoders']
        scaler = preprocessed['scaler']
        feature_names = preprocessed['feature_names']
        
        # Train multiple models
        print("\n" + "="*70)
        print("TRAINING MODELS")
        print("="*70)
        
        models = {}
        
        # RandomForestRegressor
        models['RandomForestRegressor'] = train_random_forest(
            X_train, y_train,
            n_estimators=100,
            random_state=42
        )
        
        # XGBRegressor
        if XGBOOST_AVAILABLE:
            models['XGBRegressor'] = train_xgboost(
                X_train, y_train,
                n_estimators=100,
                random_state=42
            )
        
        # LinearRegression
        models['LinearRegression'] = train_linear_regression(X_train, y_train)
        
        # Compare models and select best
        best_model_info, all_results = compare_models(models, X_test, y_test)
        
        # Save best model
        print("\n" + "="*70)
        print("SAVING RESULTS")
        print("="*70)
        save_best_model(best_model_info, encoders, scaler, feature_names)
        save_evaluation_metrics(best_model_info, all_results)
        
        # Print final metrics
        print_final_metrics(best_model_info, all_results)
        
        elapsed_time = time() - start_time
        print(f"Total training time: {elapsed_time/60:.2f} minutes ({elapsed_time:.2f} seconds)")
        print("\n" + "="*70)
        print("TRAINING COMPLETE!")
        print("="*70 + "\n")
        
    except FileNotFoundError as e:
        print(f"\n[ERROR] File not found: {str(e)}")
        print("Please ensure the dataset file exists in the data/ directory.")
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


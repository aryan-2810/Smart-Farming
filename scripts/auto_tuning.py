"""
Automated Model Tuning and Selection
====================================

This script automatically tunes and compares multiple ML models to find
the best performing model for crop recommendation.

Models tested:
- RandomForestClassifier
- DecisionTreeClassifier
- SVC (Support Vector Classifier)
- XGBClassifier (XGBoost)

Author: ML Engineer
Date: 2025
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib
from pathlib import Path
import time
import warnings

warnings.filterwarnings('ignore')

# Try to import XGBoost
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("[WARNING] XGBoost not available. Install with: pip install xgboost")


def load_and_preprocess_data(data_path='data/crop_recommendation.csv'):
    """
    Load and preprocess the crop recommendation dataset.
    
    Parameters:
    -----------
    data_path : str
        Path to the dataset file
    
    Returns:
    --------
    tuple
        (X_train, X_test, y_train, y_test, feature_names, label_encoder, scaler)
    """
    print("="*70)
    print("LOADING AND PREPROCESSING DATA")
    print("="*70)
    
    # Load dataset
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    
    print(f"\n[INFO] Loading dataset: {data_path}")
    df = pd.read_csv(data_path)
    print(f"[OK] Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Separate features and target
    feature_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    target_col = 'label'
    
    # Check if columns exist (handle case sensitivity)
    available_cols = [col for col in df.columns if col.lower() in [f.lower() for f in feature_cols]]
    if len(available_cols) != len(feature_cols):
        # Try to find target column (might be 'label' or 'Label' or 'crop')
        if 'label' in df.columns:
            target_col = 'label'
        elif 'Label' in df.columns:
            target_col = 'Label'
        elif 'crop' in df.columns:
            target_col = 'crop'
        else:
            raise ValueError(f"Could not find target column. Available columns: {df.columns.tolist()}")
        
        # Use actual column names from dataset
        feature_cols = [col for col in df.columns if col != target_col]
    
    print(f"\n[INFO] Features: {feature_cols}")
    print(f"[INFO] Target: {target_col}")
    
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    # Encode target labels
    print("\n[INFO] Encoding target labels...")
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    print(f"[OK] Encoded {len(label_encoder.classes_)} classes")
    
    # Scale features
    print("\n[INFO] Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols)
    print(f"[OK] Features scaled")
    
    # Split data
    print("\n[INFO] Splitting data into train/test (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled_df, y_encoded, 
        test_size=0.2, 
        random_state=42,
        stratify=y_encoded
    )
    print(f"[OK] Training set: {X_train.shape[0]} samples")
    print(f"[OK] Test set: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test, feature_cols, label_encoder, scaler


def tune_random_forest(X_train, y_train, cv=5, n_jobs=-1):
    """
    Tune RandomForestClassifier using GridSearchCV.
    
    Returns:
    --------
    GridSearchCV object
    """
    print("\n[INFO] Tuning Random Forest...")
    
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    model = RandomForestClassifier(random_state=42, n_jobs=n_jobs)
    grid_search = GridSearchCV(
        model, param_grid, 
        cv=cv, 
        scoring='accuracy',
        n_jobs=n_jobs,
        verbose=0
    )
    
    grid_search.fit(X_train, y_train)
    print("[OK] Random Forest tuning complete")
    
    return grid_search


def tune_decision_tree(X_train, y_train, cv=5, n_jobs=-1):
    """
    Tune DecisionTreeClassifier using GridSearchCV.
    
    Returns:
    --------
    GridSearchCV object
    """
    print("\n[INFO] Tuning Decision Tree...")
    
    param_grid = {
        'max_depth': [5, 10, 15, 20, None],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 4, 8],
        'criterion': ['gini', 'entropy']
    }
    
    model = DecisionTreeClassifier(random_state=42)
    grid_search = GridSearchCV(
        model, param_grid,
        cv=cv,
        scoring='accuracy',
        n_jobs=n_jobs,
        verbose=0
    )
    
    grid_search.fit(X_train, y_train)
    print("[OK] Decision Tree tuning complete")
    
    return grid_search


def tune_svc(X_train, y_train, cv=5, n_jobs=-1):
    """
    Tune SVC (Support Vector Classifier) using GridSearchCV.
    
    Returns:
    --------
    GridSearchCV object
    """
    print("\n[INFO] Tuning SVC (Support Vector Classifier)...")
    print("[INFO] This may take a while...")
    
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
        'kernel': ['rbf', 'poly', 'sigmoid']
    }
    
    model = SVC(random_state=42, probability=True)
    grid_search = GridSearchCV(
        model, param_grid,
        cv=cv,
        scoring='accuracy',
        n_jobs=n_jobs,
        verbose=0
    )
    
    grid_search.fit(X_train, y_train)
    print("[OK] SVC tuning complete")
    
    return grid_search


def tune_xgboost(X_train, y_train, cv=5, n_jobs=-1):
    """
    Tune XGBClassifier using GridSearchCV.
    
    Returns:
    --------
    GridSearchCV object or None if XGBoost not available
    """
    if not XGBOOST_AVAILABLE:
        print("\n[SKIP] XGBoost not available")
        return None
    
    print("\n[INFO] Tuning XGBoost...")
    
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7, 10],
        'learning_rate': [0.01, 0.1, 0.3],
        'subsample': [0.8, 1.0]
    }
    
    model = XGBClassifier(random_state=42, n_jobs=n_jobs, eval_metric='mlogloss')
    grid_search = GridSearchCV(
        model, param_grid,
        cv=cv,
        scoring='accuracy',
        n_jobs=n_jobs,
        verbose=0
    )
    
    grid_search.fit(X_train, y_train)
    print("[OK] XGBoost tuning complete")
    
    return grid_search


def compare_models(results_dict, X_test, y_test):
    """
    Compare all tuned models and find the best one.
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary with model names as keys and GridSearchCV objects as values
    X_test : array-like
        Test features
    y_test : array-like
        Test labels
    
    Returns:
    --------
    dict
        Best model information
    """
    print("\n" + "="*70)
    print("COMPARING MODELS")
    print("="*70)
    
    model_comparison = []
    
    for model_name, grid_search in results_dict.items():
        if grid_search is None:
            continue
        
        best_score = grid_search.best_score_
        best_params = grid_search.best_params_
        
        # Evaluate on test set
        y_pred = grid_search.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)
        
        model_comparison.append({
            'name': model_name,
            'cv_score': best_score,
            'test_accuracy': test_accuracy,
            'best_params': best_params,
            'grid_search': grid_search
        })
        
        print(f"\n{model_name}:")
        print(f"  CV Accuracy: {best_score:.4f}")
        print(f"  Test Accuracy: {test_accuracy:.4f}")
        print(f"  Best Params: {best_params}")
    
    # Find best model based on CV score
    best_model_info = max(model_comparison, key=lambda x: x['cv_score'])
    
    print("\n" + "="*70)
    print("BEST MODEL")
    print("="*70)
    print(f"\nModel: {best_model_info['name']}")
    print(f"CV Accuracy: {best_model_info['cv_score']:.4f}")
    print(f"Test Accuracy: {best_model_info['test_accuracy']:.4f}")
    print(f"Best Parameters:")
    for param, value in best_model_info['best_params'].items():
        print(f"  {param}: {value}")
    
    return best_model_info, model_comparison


def save_results(best_model_info, all_results, label_encoder, scaler, 
                 save_model_path='models/best_crop_model.pkl',
                 save_summary_path='results/best_params.txt'):
    """
    Save the best model and summary results.
    
    Parameters:
    -----------
    best_model_info : dict
        Information about the best model
    all_results : list
        List of all model results
    label_encoder : LabelEncoder
        Encoder for labels
    scaler : StandardScaler
        Feature scaler
    save_model_path : str
        Path to save the model
    save_summary_path : str
        Path to save the summary
    """
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)
    
    # Save best model
    save_model_path = Path(save_model_path)
    save_model_path.parent.mkdir(parents=True, exist_ok=True)
    
    model_data = {
        'model': best_model_info['grid_search'].best_estimator_,
        'label_encoder': label_encoder,
        'scaler': scaler,
        'model_name': best_model_info['name'],
        'best_params': best_model_info['best_params'],
        'cv_score': best_model_info['cv_score'],
        'test_accuracy': best_model_info['test_accuracy']
    }
    
    joblib.dump(model_data, save_model_path)
    print(f"[OK] Best model saved to: {save_model_path}")
    
    # Save summary
    save_summary_path = Path(save_summary_path)
    save_summary_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_summary_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("AUTOMATED MODEL TUNING RESULTS - CROP RECOMMENDATION\n")
        f.write("="*70 + "\n\n")
        
        f.write("BEST MODEL SUMMARY\n")
        f.write("-"*70 + "\n")
        f.write(f"Model Name: {best_model_info['name']}\n")
        f.write(f"Cross-Validation Accuracy: {best_model_info['cv_score']:.4f}\n")
        f.write(f"Test Set Accuracy: {best_model_info['test_accuracy']:.4f}\n\n")
        
        f.write("Best Hyperparameters:\n")
        for param, value in best_model_info['best_params'].items():
            f.write(f"  {param}: {value}\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("ALL MODELS COMPARISON\n")
        f.write("="*70 + "\n\n")
        
        # Sort by CV score
        sorted_results = sorted(all_results, key=lambda x: x['cv_score'], reverse=True)
        
        for idx, result in enumerate(sorted_results, 1):
            f.write(f"{idx}. {result['name']}\n")
            f.write(f"   CV Accuracy: {result['cv_score']:.4f}\n")
            f.write(f"   Test Accuracy: {result['test_accuracy']:.4f}\n")
            f.write(f"   Parameters: {result['best_params']}\n\n")
        
        f.write("="*70 + "\n")
        f.write("Note: Models are ranked by Cross-Validation accuracy.\n")
        f.write("="*70 + "\n")
    
    print(f"[OK] Summary saved to: {save_summary_path}")


def main():
    """
    Main execution function for automated model tuning.
    """
    print("\n" + "="*70)
    print("AUTOMATED MODEL TUNING - CROP RECOMMENDATION")
    print("="*70)
    
    start_time = time.time()
    
    # Load and preprocess data
    X_train, X_test, y_train, y_test, feature_names, label_encoder, scaler = \
        load_and_preprocess_data()
    
    # Tune all models
    print("\n" + "="*70)
    print("TUNING MODELS")
    print("="*70)
    
    results = {}
    
    # Random Forest
    results['RandomForestClassifier'] = tune_random_forest(X_train, y_train)
    
    # Decision Tree
    results['DecisionTreeClassifier'] = tune_decision_tree(X_train, y_train)
    
    # SVC
    results['SVC'] = tune_svc(X_train, y_train)
    
    # XGBoost
    if XGBOOST_AVAILABLE:
        results['XGBClassifier'] = tune_xgboost(X_train, y_train)
    
    # Compare models and find best
    best_model_info, all_results = compare_models(results, X_test, y_test)
    
    # Save results
    save_results(best_model_info, all_results, label_encoder, scaler)
    
    elapsed_time = time.time() - start_time
    
    print("\n" + "="*70)
    print("TUNING COMPLETE!")
    print("="*70)
    print(f"\nTotal time: {elapsed_time/60:.2f} minutes ({elapsed_time:.2f} seconds)")
    print(f"\nBest model: {best_model_info['name']}")
    print(f"Accuracy: {best_model_info['cv_score']:.4f}")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()


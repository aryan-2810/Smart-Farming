"""
Hyperparameter Tuning and Feature Importance Analysis
======================================================

This script performs hyperparameter tuning using GridSearchCV for both:
- Crop Recommendation Model (Classification)
- Crop Yield Prediction Model (Regression)

It also visualizes the top 10 most important features for both models.

Author: ML Engineer
Date: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    r2_score, mean_absolute_error, mean_squared_error
)
from pathlib import Path
import joblib
import sys
import warnings
from time import time

warnings.filterwarnings('ignore')

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10


def get_project_paths():
    """
    Get paths to important project directories.
    
    Returns:
    --------
    dict
        Dictionary containing paths to models and results directories
    """
    project_root = Path(__file__).parent.parent
    return {
        'models_dir': project_root / 'models',
        'feature_importance_dir': project_root / 'results' / 'feature_importance'
    }


def tune_classification_model(X_train, y_train, cv=5, n_jobs=-1, verbose=1):
    """
    Perform hyperparameter tuning for RandomForestClassifier using GridSearchCV.
    
    Parameters:
    -----------
    X_train : pd.DataFrame or np.array
        Training features
    y_train : pd.Series or np.array
        Training target labels
    cv : int, default=5
        Number of cross-validation folds
    n_jobs : int, default=-1
        Number of parallel jobs
    verbose : int, default=1
        Verbosity level
    
    Returns:
    --------
    GridSearchCV
        Fitted GridSearchCV object with best model
    """
    print(f"\n{'='*60}")
    print("HYPERPARAMETER TUNING - CROP RECOMMENDATION (CLASSIFICATION)")
    print(f"{'='*60}")
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }
    
    print("Parameter grid:")
    for param, values in param_grid.items():
        print(f"  {param}: {values}")
    
    # Initialize base model
    base_model = RandomForestClassifier(random_state=42, n_jobs=n_jobs)
    
    # Create GridSearchCV
    print(f"\nPerforming GridSearchCV with {cv}-fold cross-validation...")
    print(f"Total parameter combinations: {np.prod([len(v) for v in param_grid.values()])}")
    
    start_time = time()
    
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=cv,
        scoring='accuracy',  # Primary metric for classification
        n_jobs=n_jobs,
        verbose=verbose,
        return_train_score=True
    )
    
    # Fit GridSearchCV
    grid_search.fit(X_train, y_train)
    
    elapsed_time = time() - start_time
    
    print(f"\nGridSearchCV completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    print(f"\nBest parameters found:")
    print("-" * 60)
    for param, value in grid_search.best_params_.items():
        print(f"  {param}: {value}")
    
    print(f"\nBest cross-validation score: {grid_search.best_score_:.4f}")
    
    return grid_search


def tune_regression_model(X_train, y_train, cv=5, n_jobs=-1, verbose=1):
    """
    Perform hyperparameter tuning for RandomForestRegressor using GridSearchCV.
    
    Parameters:
    -----------
    X_train : pd.DataFrame or np.array
        Training features
    y_train : pd.Series or np.array
        Training target values
    cv : int, default=5
        Number of cross-validation folds
    n_jobs : int, default=-1
        Number of parallel jobs
    verbose : int, default=1
        Verbosity level
    
    Returns:
    --------
    GridSearchCV
        Fitted GridSearchCV object with best model
    """
    print(f"\n{'='*60}")
    print("HYPERPARAMETER TUNING - CROP YIELD PREDICTION (REGRESSION)")
    print(f"{'='*60}")
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }
    
    print("Parameter grid:")
    for param, values in param_grid.items():
        print(f"  {param}: {values}")
    
    # Initialize base model
    base_model = RandomForestRegressor(random_state=42, n_jobs=n_jobs)
    
    # Create GridSearchCV
    print(f"\nPerforming GridSearchCV with {cv}-fold cross-validation...")
    print(f"Total parameter combinations: {np.prod([len(v) for v in param_grid.values()])}")
    
    start_time = time()
    
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=cv,
        scoring='r2',  # Primary metric for regression
        n_jobs=n_jobs,
        verbose=verbose,
        return_train_score=True
    )
    
    # Fit GridSearchCV
    grid_search.fit(X_train, y_train)
    
    elapsed_time = time() - start_time
    
    print(f"\nGridSearchCV completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    print(f"\nBest parameters found:")
    print("-" * 60)
    for param, value in grid_search.best_params_.items():
        print(f"  {param}: {value}")
    
    print(f"\nBest cross-validation score (R²): {grid_search.best_score_:.4f}")
    
    return grid_search


def plot_feature_importance(model, feature_names, top_n=10, model_name='model', save_path=None):
    """
    Visualize the top N most important features from a RandomForest model.
    
    Parameters:
    -----------
    model : RandomForestClassifier or RandomForestRegressor
        Trained model with feature_importances_ attribute
    feature_names : list or array
        Names of features
    top_n : int, default=10
        Number of top features to display
    model_name : str, default='model'
        Name of the model (for plot title)
    save_path : str or Path, optional
        Path to save the plot
    """
    print(f"\n{'='*60}")
    print(f"FEATURE IMPORTANCE ANALYSIS - {model_name.upper()}")
    print(f"{'='*60}")
    
    # Get feature importances
    importances = model.feature_importances_
    
    # Create DataFrame for easier handling
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    # Get top N features
    top_features = feature_importance_df.head(top_n)
    
    print(f"\nTop {top_n} Most Important Features:")
    print("-" * 60)
    for idx, row in top_features.iterrows():
        print(f"{row['feature']:<30} {row['importance']:.6f}")
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot 1: Horizontal bar chart (top N)
    ax1 = axes[0]
    colors = sns.color_palette("viridis", len(top_features))
    bars = ax1.barh(range(len(top_features)), top_features['importance'], color=colors)
    ax1.set_yticks(range(len(top_features)))
    ax1.set_yticklabels(top_features['feature'])
    ax1.set_xlabel('Feature Importance', fontsize=12, fontweight='bold')
    ax1.set_title(f'Top {top_n} Feature Importances - {model_name}', 
                  fontsize=14, fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3, axis='x')
    ax1.invert_yaxis()  # Highest importance at top
    
    # Add value labels on bars
    for i, (idx, row) in enumerate(top_features.iterrows()):
        ax1.text(row['importance'], i, f' {row["importance"]:.4f}', 
                va='center', fontsize=9, fontweight='bold')
    
    # Plot 2: Cumulative importance
    ax2 = axes[1]
    sorted_features = feature_importance_df.sort_values('importance', ascending=False)
    cumulative_importance = sorted_features['importance'].cumsum()
    ax2.plot(range(1, len(sorted_features) + 1), cumulative_importance, 
             marker='o', linewidth=2, markersize=6, color='steelblue')
    ax2.axhline(y=0.8, color='r', linestyle='--', linewidth=2, 
                label='80% Cumulative Importance')
    ax2.axhline(y=0.9, color='orange', linestyle='--', linewidth=2, 
                label='90% Cumulative Importance')
    ax2.set_xlabel('Number of Features', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Cumulative Importance', fontsize=12, fontweight='bold')
    ax2.set_title('Cumulative Feature Importance', fontsize=14, fontweight='bold', pad=15)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    
    # Find number of features needed for 80% and 90% importance
    n_80 = len(cumulative_importance[cumulative_importance <= 0.8]) + 1
    n_90 = len(cumulative_importance[cumulative_importance <= 0.9]) + 1
    ax2.axvline(x=n_80, color='r', linestyle=':', alpha=0.7)
    ax2.axvline(x=n_90, color='orange', linestyle=':', alpha=0.7)
    
    print(f"\nFeature importance insights:")
    print(f"  Number of features for 80% importance: {n_80}")
    print(f"  Number of features for 90% importance: {n_90}")
    
    plt.tight_layout()
    
    # Save plot
    if save_path is None:
        paths = get_project_paths()
        save_path = paths['feature_importance_dir'] / f'feature_importance_{model_name}.png'
    
    # Ensure directory exists
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Feature importance plot saved to: {save_path}")
    plt.close()
    
    print(f"{'='*60}\n")
    
    return feature_importance_df


def evaluate_tuned_classification_model(model, X_test, y_test, class_names=None):
    """
    Evaluate the tuned classification model.
    
    Parameters:
    -----------
    model : RandomForestClassifier
        Tuned model
    X_test : pd.DataFrame or np.array
        Test features
    y_test : pd.Series or np.array
        Test target labels
    class_names : list, optional
        Names of classes
    
    Returns:
    --------
    dict
        Dictionary containing evaluation metrics
    """
    print(f"\n{'='*60}")
    print("EVALUATING TUNED CLASSIFICATION MODEL")
    print(f"{'='*60}")
    
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'y_pred': y_pred
    }
    
    print(f"\n{'Metric':<25} {'Score':<15}")
    print("-" * 40)
    print(f"{'Accuracy':<25} {accuracy:.4f}")
    print(f"{'Precision (Weighted)':<25} {precision:.4f}")
    print(f"{'Recall (Weighted)':<25} {recall:.4f}")
    print(f"{'F1-Score (Weighted)':<25} {f1:.4f}")
    print(f"{'='*60}\n")
    
    return metrics


def evaluate_tuned_regression_model(model, X_test, y_test):
    """
    Evaluate the tuned regression model.
    
    Parameters:
    -----------
    model : RandomForestRegressor
        Tuned model
    X_test : pd.DataFrame or np.array
        Test features
    y_test : pd.Series or np.array
        Test target values
    
    Returns:
    --------
    dict
        Dictionary containing evaluation metrics
    """
    print(f"\n{'='*60}")
    print("EVALUATING TUNED REGRESSION MODEL")
    print(f"{'='*60}")
    
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
    
    print(f"\n{'Metric':<25} {'Score':<15}")
    print("-" * 40)
    print(f"{'R² Score':<25} {r2:.4f}")
    print(f"{'MAE':<25} {mae:.4f}")
    print(f"{'MSE':<25} {mse:.4f}")
    print(f"{'RMSE':<25} {rmse:.4f}")
    print(f"{'='*60}\n")
    
    return metrics


def save_tuned_model(model, encoders, scaler, model_name, save_path=None):
    """
    Save the tuned model with preprocessing objects.
    
    Parameters:
    -----------
    model : sklearn model
        Tuned model
    encoders : dict
        Preprocessing encoders
    scaler : StandardScaler
        Feature scaler
    model_name : str
        Name identifier for the model
    save_path : str or Path, optional
        Path to save the model
    """
    paths = get_project_paths()
    if save_path is None:
        save_path = paths['models_dir'] / f'{model_name}.pkl'
    
    # Ensure directory exists
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    model_data = {
        'model': model,
        'encoders': encoders,
        'scaler': scaler,
        'tuned': True
    }
    
    joblib.dump(model_data, save_path)
    print(f"Tuned model saved to: {save_path}")


def tune_and_save_crop_recommendation_model(cv=5, top_n_features=10):
    """
    Complete pipeline: Tune, evaluate, visualize features, and save crop recommendation model.
    
    Parameters:
    -----------
    cv : int, default=5
        Number of CV folds
    top_n_features : int, default=10
        Number of top features to visualize
    
    Returns:
    --------
    dict
        Dictionary containing all results
    """
    print("\n" + "="*60)
    print("COMPLETE TUNING PIPELINE - CROP RECOMMENDATION MODEL")
    print("="*60)
    
    # Load and preprocess data
    from load_data import load_dataset
    from preprocess_data import preprocess_pipeline
    
    df = load_dataset('crop_recommendation.csv', display_info=False)
    preprocessed = preprocess_pipeline(
        df=df,
        target_col='label',
        handle_missing='median',
        remove_duplicates=True,
        encoding_type='label',
        scale_features_flag=True,
        test_size=0.2,
        random_state=42
    )
    
    X_train = preprocessed['X_train']
    X_test = preprocessed['X_test']
    y_train = preprocessed['y_train']
    y_test = preprocessed['y_test']
    encoders = preprocessed['encoders']
    scaler = preprocessed['scaler']
    
    # Get class names
    class_names = encoders['label'].classes_.tolist() if 'label' in encoders else None
    
    # Tune model
    grid_search = tune_classification_model(X_train, y_train, cv=cv)
    best_model = grid_search.best_estimator_
    
    # Evaluate tuned model
    metrics = evaluate_tuned_classification_model(best_model, X_test, y_test, class_names)
    
    # Plot feature importance
    feature_names = X_train.columns.tolist() if hasattr(X_train, 'columns') else [f'Feature_{i}' for i in range(X_train.shape[1])]
    feature_importance_df = plot_feature_importance(
        best_model, feature_names, top_n=top_n_features,
        model_name='crop_recommendation_tuned'
    )
    
    # Save tuned model
    save_tuned_model(best_model, encoders, scaler, 'crop_recommendation_tuned')
    
    return {
        'model': best_model,
        'grid_search': grid_search,
        'metrics': metrics,
        'feature_importance': feature_importance_df,
        'preprocessed': preprocessed
    }


def tune_and_save_yield_prediction_model(cv=5, top_n_features=10):
    """
    Complete pipeline: Tune, evaluate, visualize features, and save yield prediction model.
    
    Parameters:
    -----------
    cv : int, default=5
        Number of CV folds
    top_n_features : int, default=10
        Number of top features to visualize
    
    Returns:
    --------
    dict
        Dictionary containing all results
    """
    print("\n" + "="*60)
    print("COMPLETE TUNING PIPELINE - CROP YIELD PREDICTION MODEL")
    print("="*60)
    
    # Load and preprocess data
    from load_data import load_dataset
    from preprocess_data import preprocess_pipeline
    from yield_prediction_model import prepare_yield_dataset
    
    df = load_dataset('crop_production.csv', display_info=False)
    df = prepare_yield_dataset(df, target_col='yield')
    
    preprocessed = preprocess_pipeline(
        df=df,
        target_col='yield',
        handle_missing='median',
        remove_duplicates=True,
        encoding_type='label',
        scale_features_flag=True,
        test_size=0.2,
        random_state=42
    )
    
    X_train = preprocessed['X_train']
    X_test = preprocessed['X_test']
    y_train = preprocessed['y_train']
    y_test = preprocessed['y_test']
    encoders = preprocessed['encoders']
    scaler = preprocessed['scaler']
    
    # Tune model
    grid_search = tune_regression_model(X_train, y_train, cv=cv)
    best_model = grid_search.best_estimator_
    
    # Evaluate tuned model
    metrics = evaluate_tuned_regression_model(best_model, X_test, y_test)
    
    # Plot feature importance
    feature_names = X_train.columns.tolist() if hasattr(X_train, 'columns') else [f'Feature_{i}' for i in range(X_train.shape[1])]
    feature_importance_df = plot_feature_importance(
        best_model, feature_names, top_n=top_n_features,
        model_name='crop_yield_tuned'
    )
    
    # Save tuned model
    save_tuned_model(best_model, encoders, scaler, 'crop_yield_tuned')
    
    return {
        'model': best_model,
        'grid_search': grid_search,
        'metrics': metrics,
        'feature_importance': feature_importance_df,
        'preprocessed': preprocessed
    }


if __name__ == "__main__":
    """
    Main execution block.
    Tunes both models and saves results.
    """
    print("="*60)
    print("HYPERPARAMETER TUNING - BOTH MODELS")
    print("="*60)
    
    # Tune crop recommendation model
    print("\n" + "="*60)
    print("STEP 1: TUNING CROP RECOMMENDATION MODEL")
    print("="*60)
    recommendation_results = tune_and_save_crop_recommendation_model(cv=5, top_n_features=10)
    
    # Tune yield prediction model
    print("\n" + "="*60)
    print("STEP 2: TUNING CROP YIELD PREDICTION MODEL")
    print("="*60)
    yield_results = tune_and_save_yield_prediction_model(cv=5, top_n_features=10)
    
    # Summary
    print("\n" + "="*60)
    print("HYPERPARAMETER TUNING COMPLETE!")
    print("="*60)
    
    print("\nCrop Recommendation Model - Final Performance:")
    print(f"  Accuracy:  {recommendation_results['metrics']['accuracy']:.4f}")
    print(f"  Precision: {recommendation_results['metrics']['precision']:.4f}")
    print(f"  Recall:    {recommendation_results['metrics']['recall']:.4f}")
    print(f"  F1-Score:  {recommendation_results['metrics']['f1_score']:.4f}")
    
    print("\nCrop Yield Prediction Model - Final Performance:")
    print(f"  R² Score: {yield_results['metrics']['r2_score']:.4f}")
    print(f"  MAE:       {yield_results['metrics']['mae']:.4f}")
    print(f"  MSE:       {yield_results['metrics']['mse']:.4f}")
    print(f"  RMSE:      {yield_results['metrics']['rmse']:.4f}")
    
    print("\nGenerated Files:")
    print("  - models/crop_recommendation_tuned.pkl")
    print("  - models/crop_yield_tuned.pkl")
    print("  - results/feature_importance/feature_importance_crop_recommendation_tuned.png")
    print("  - results/feature_importance/feature_importance_crop_yield_tuned.png")
    print("="*60 + "\n")


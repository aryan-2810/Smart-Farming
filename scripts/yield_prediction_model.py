"""
Crop Yield Prediction Model
============================

This script builds and trains a RandomForestRegressor for crop yield prediction.
It evaluates the model using R² score, MAE, and MSE, and generates
an actual vs predicted yield comparison plot.

Author: ML Engineer
Date: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from pathlib import Path
import joblib
import sys
import warnings

warnings.filterwarnings('ignore')

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
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
        'results_dir': project_root / 'results'
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
            print(f"Yield calculated. Mean yield: {df_prepared[target_col].mean():.2f} tons/hectare")
        else:
            raise ValueError(f"Target column '{target_col}' not found and cannot be calculated.")
    
    # Remove rows with invalid yield values
    df_prepared = df_prepared[df_prepared[target_col].notna()].copy()
    df_prepared = df_prepared[df_prepared[target_col] > 0].copy()
    
    return df_prepared


def load_and_preprocess_data(dataset_name='crop_production.csv', target_col='yield'):
    """
    Load and preprocess the crop yield dataset.
    
    Parameters:
    -----------
    dataset_name : str, default='crop_production.csv'
        Name of the dataset file
    target_col : str, default='yield'
        Name of the target column
    
    Returns:
    --------
    dict
        Dictionary containing preprocessed data and preprocessing objects
    """
    # Import preprocessing modules
    from load_data import load_dataset
    from preprocess_data import preprocess_pipeline
    
    print("="*60)
    print("LOADING AND PREPROCESSING DATA")
    print("="*60)
    
    # Load dataset
    df = load_dataset(dataset_name, display_info=False)
    
    # Prepare dataset (calculate yield if needed)
    df = prepare_yield_dataset(df, target_col=target_col)
    
    # Preprocess data
    preprocessed = preprocess_pipeline(
        df=df,
        target_col=target_col,
        handle_missing='median',
        remove_duplicates=True,
        encoding_type='label',
        scale_features_flag=True,
        test_size=0.2,
        random_state=42
    )
    
    return preprocessed


def train_yield_prediction_model(X_train, y_train, n_estimators=100, random_state=42, **kwargs):
    """
    Train a RandomForestRegressor for crop yield prediction.
    
    Parameters:
    -----------
    X_train : pd.DataFrame or np.array
        Training features
    y_train : pd.Series or np.array
        Training target values
    n_estimators : int, default=100
        Number of trees in the random forest
    random_state : int, default=42
        Random seed for reproducibility
    **kwargs : dict
        Additional parameters for RandomForestRegressor
    
    Returns:
    --------
    RandomForestRegressor
        Trained model
    """
    print(f"\n{'='*60}")
    print("TRAINING RANDOM FOREST REGRESSOR")
    print(f"{'='*60}")
    print(f"Training samples: {len(X_train)}")
    print(f"Features: {X_train.shape[1]}")
    print(f"Target range: [{y_train.min():.2f}, {y_train.max():.2f}]")
    print(f"Target mean: {y_train.mean():.2f}")
    print(f"Number of estimators (trees): {n_estimators}")
    
    # Initialize RandomForestRegressor
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,  # Use all available cores
        verbose=1,
        **kwargs
    )
    
    # Train the model
    print("\nTraining model...")
    model.fit(X_train, y_train)
    print("Training completed!")
    
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained regression model and return metrics.
    
    Parameters:
    -----------
    model : RandomForestRegressor
        Trained model
    X_test : pd.DataFrame or np.array
        Test features
    y_test : pd.Series or np.array
        Test target values
    
    Returns:
    --------
    dict
        Dictionary containing all evaluation metrics
    """
    print(f"\n{'='*60}")
    print("MODEL EVALUATION")
    print(f"{'='*60}")
    
    # Make predictions
    print("Making predictions on test set...")
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    # Calculate additional metrics
    mean_yield = y_test.mean()
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100  # Mean Absolute Percentage Error
    
    # Create metrics dictionary
    metrics = {
        'r2_score': r2,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'mape': mape,
        'y_true': y_test,
        'y_pred': y_pred,
        'mean_yield': mean_yield
    }
    
    # Print metrics
    print(f"\n{'Metric':<25} {'Score':<15} {'Description':<40}")
    print("-" * 80)
    print(f"{'R² Score':<25} {r2:.4f}{'':<10} Higher is better (max 1.0)")
    print(f"{'Mean Absolute Error (MAE)':<25} {mae:.4f}{'':<10} Lower is better")
    print(f"{'Mean Squared Error (MSE)':<25} {mse:.4f}{'':<10} Lower is better")
    print(f"{'Root Mean Squared Error (RMSE)':<25} {rmse:.4f}{'':<10} Lower is better")
    print(f"{'Mean Absolute % Error (MAPE)':<25} {mape:.2f}%{'':<9} Lower is better")
    print(f"{'Mean Yield':<25} {mean_yield:.4f}{'':<10} Average actual yield")
    
    print(f"\n{'='*60}\n")
    
    return metrics


def plot_actual_vs_predicted(y_true, y_pred, save_path=None):
    """
    Create and save an actual vs predicted yield comparison plot.
    
    Parameters:
    -----------
    y_true : array-like
        True yield values
    y_pred : array-like
        Predicted yield values
    save_path : str or Path, optional
        Path to save the plot
    """
    print(f"\n{'='*60}")
    print("GENERATING ACTUAL VS PREDICTED PLOT")
    print(f"{'='*60}")
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Scatter plot with regression line
    ax1 = axes[0]
    scatter = ax1.scatter(y_true, y_pred, alpha=0.6, s=50, c='steelblue', edgecolors='black', linewidth=0.5)
    
    # Perfect prediction line (y = x)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 
             'r--', linewidth=2, label='Perfect Prediction (y = x)')
    
    # Calculate and plot regression line
    z = np.polyfit(y_true, y_pred, 1)
    p = np.poly1d(z)
    ax1.plot(y_true, p(y_true), "g--", linewidth=2, label=f'Best Fit Line (slope={z[0]:.3f})')
    
    ax1.set_xlabel('Actual Yield', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Predicted Yield', fontsize=12, fontweight='bold')
    ax1.set_title('Actual vs Predicted Yield (Scatter Plot)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    # Add R² score to plot
    r2 = r2_score(y_true, y_pred)
    ax1.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax1.transAxes,
             fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 2: Residual plot
    ax2 = axes[1]
    residuals = y_true - y_pred
    ax2.scatter(y_pred, residuals, alpha=0.6, s=50, c='coral', edgecolors='black', linewidth=0.5)
    ax2.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax2.set_xlabel('Predicted Yield', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Residuals (Actual - Predicted)', fontsize=12, fontweight='bold')
    ax2.set_title('Residual Plot', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    if save_path is None:
        paths = get_project_paths()
        save_path = paths['results_dir'] / 'yield_comparison.png'
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Actual vs Predicted plot saved to: {save_path}")
    plt.close()
    
    print(f"{'='*60}\n")


def save_metrics(metrics, save_path=None):
    """
    Save evaluation metrics to a text file.
    
    Parameters:
    -----------
    metrics : dict
        Dictionary containing evaluation metrics
    save_path : str or Path, optional
        Path to save the metrics file
    """
    paths = get_project_paths()
    if save_path is None:
        save_path = paths['results_dir'] / 'metrics' / 'crop_yield_metrics.txt'
    
    # Ensure directory exists
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("CROP YIELD PREDICTION MODEL - EVALUATION METRICS\n")
        f.write("="*60 + "\n\n")
        
        f.write("REGRESSION METRICS:\n")
        f.write("-" * 60 + "\n")
        f.write(f"R² Score (Coefficient of Determination): {metrics['r2_score']:.4f}\n")
        f.write(f"  - Interpretation: {metrics['r2_score']*100:.2f}% of variance in yield is explained by the model\n")
        f.write(f"  - Range: [0, 1], Higher is better\n\n")
        
        f.write(f"Mean Absolute Error (MAE): {metrics['mae']:.4f}\n")
        f.write(f"  - Average absolute difference between actual and predicted yield\n")
        f.write(f"  - Lower is better\n\n")
        
        f.write(f"Mean Squared Error (MSE): {metrics['mse']:.4f}\n")
        f.write(f"  - Average squared difference between actual and predicted yield\n")
        f.write(f"  - Lower is better\n\n")
        
        f.write(f"Root Mean Squared Error (RMSE): {metrics['rmse']:.4f}\n")
        f.write(f"  - Square root of MSE, in same units as target variable\n")
        f.write(f"  - Lower is better\n\n")
        
        f.write(f"Mean Absolute Percentage Error (MAPE): {metrics['mape']:.2f}%\n")
        f.write(f"  - Average percentage error\n")
        f.write(f"  - Lower is better\n\n")
        
        f.write("DATASET STATISTICS:\n")
        f.write("-" * 60 + "\n")
        f.write(f"Mean Yield: {metrics['mean_yield']:.4f} tons/hectare\n")
        f.write(f"Number of test samples: {len(metrics['y_true'])}\n")
    
    print(f"Metrics saved to: {save_path}")


def save_model(model, encoders, scaler, save_path=None):
    """
    Save the trained model and preprocessing objects.
    
    Parameters:
    -----------
    model : RandomForestRegressor
        Trained model
    encoders : dict
        Dictionary of encoders used in preprocessing
    scaler : StandardScaler
        Scaler used for feature normalization
    save_path : str or Path, optional
        Path to save the model
    """
    paths = get_project_paths()
    if save_path is None:
        save_path = paths['models_dir'] / 'crop_yield_predictor.pkl'
    
    # Ensure models directory exists
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save model, encoders, and scaler
    model_data = {
        'model': model,
        'encoders': encoders,
        'scaler': scaler
    }
    
    joblib.dump(model_data, save_path)
    print(f"Model saved to: {save_path}")


def train_and_evaluate_yield_prediction_model(dataset_name='crop_production.csv', 
                                              target_col='yield',
                                              n_estimators=100, 
                                              random_state=42, 
                                              **kwargs):
    """
    Complete pipeline: Load data, train model, evaluate, and save results.
    
    Parameters:
    -----------
    dataset_name : str, default='crop_production.csv'
        Name of the dataset file
    target_col : str, default='yield'
        Name of the target column
    n_estimators : int, default=100
        Number of trees in random forest
    random_state : int, default=42
        Random seed for reproducibility
    **kwargs : dict
        Additional parameters for RandomForestRegressor
    
    Returns:
    --------
    dict
        Dictionary containing model, metrics, and preprocessing objects
    """
    print("\n" + "="*60)
    print("CROP YIELD PREDICTION MODEL - COMPLETE PIPELINE")
    print("="*60)
    
    # Step 1: Load and preprocess data
    preprocessed = load_and_preprocess_data(dataset_name=dataset_name, target_col=target_col)
    
    X_train = preprocessed['X_train']
    X_test = preprocessed['X_test']
    y_train = preprocessed['y_train']
    y_test = preprocessed['y_test']
    encoders = preprocessed['encoders']
    scaler = preprocessed['scaler']
    
    # Step 2: Train model
    model = train_yield_prediction_model(
        X_train, y_train, 
        n_estimators=n_estimators,
        random_state=random_state,
        **kwargs
    )
    
    # Step 3: Evaluate model
    metrics = evaluate_model(model, X_test, y_test)
    
    # Step 4: Generate actual vs predicted plot
    plot_actual_vs_predicted(y_test, metrics['y_pred'])
    
    # Step 5: Save metrics
    save_metrics(metrics)
    
    # Step 6: Save model
    save_model(model, encoders, scaler)
    
    # Summary
    print("\n" + "="*60)
    print("MODEL TRAINING AND EVALUATION COMPLETE!")
    print("="*60)
    print(f"\nFinal Model Performance:")
    print(f"  R² Score: {metrics['r2_score']:.4f}")
    print(f"  MAE:      {metrics['mae']:.4f}")
    print(f"  MSE:      {metrics['mse']:.4f}")
    print(f"  RMSE:     {metrics['rmse']:.4f}")
    print("\n" + "="*60 + "\n")
    
    return {
        'model': model,
        'metrics': metrics,
        'encoders': encoders,
        'scaler': scaler,
        'preprocessed': preprocessed
    }


if __name__ == "__main__":
    """
    Main execution block.
    """
    # Train and evaluate the model
    # Note: If your dataset has a 'yield' column, use target_col='yield'
    # If you need to calculate yield from Production/Area, the function will handle it
    
    results = train_and_evaluate_yield_prediction_model(
        dataset_name='crop_production.csv',
        target_col='yield',  # Will be calculated from Production/Area if not present
        n_estimators=100,
        random_state=42,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1
    )
    
    print("\nModel training completed successfully!")
    print("Check the following directories for outputs:")
    print("  - models/crop_yield_predictor.pkl")
    print("  - results/yield_comparison.png")
    print("  - results/metrics/crop_yield_metrics.txt")


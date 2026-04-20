"""
Model Evaluation and Visualization
===================================

This script combines evaluation results from both baseline and tuned models,
creates comparison visualizations, and generates a comprehensive summary report.

Visualizations:
- Bar chart comparing baseline vs tuned model performance (Accuracy/R²)
- Scatter plot of residual errors for yield prediction
- Feature importance bar charts for both models

Author: ML Engineer
Date: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    r2_score, mean_absolute_error, mean_squared_error
)
from pathlib import Path
import joblib
import sys
import warnings

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    warnings.warn("scipy not available, Q-Q plot will be skipped")

warnings.filterwarnings('ignore')

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 11


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
        'results_dir': project_root / 'results',
        'metrics_dir': project_root / 'results' / 'metrics',
        'feature_importance_dir': project_root / 'results' / 'feature_importance'
    }


def load_model(model_path):
    """
    Load a saved model from disk.
    
    Parameters:
    -----------
    model_path : str or Path
        Path to the model file
    
    Returns:
    --------
    dict
        Dictionary containing model, encoders, scaler
    """
    if not Path(model_path).exists():
        return None
    
    model_data = joblib.load(model_path)
    return model_data


def evaluate_classification_model(model, X_test, y_test):
    """
    Evaluate a classification model and return metrics.
    
    Parameters:
    -----------
    model : sklearn model
        Trained classification model
    X_test : array-like
        Test features
    y_test : array-like
        Test labels
    
    Returns:
    --------
    dict
        Dictionary of evaluation metrics
    """
    y_pred = model.predict(X_test)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
        'y_pred': y_pred
    }
    
    return metrics


def evaluate_regression_model(model, X_test, y_test):
    """
    Evaluate a regression model and return metrics.
    
    Parameters:
    -----------
    model : sklearn model
        Trained regression model
    X_test : array-like
        Test features
    y_test : array-like
        Test target values
    
    Returns:
    --------
    dict
        Dictionary of evaluation metrics
    """
    y_pred = model.predict(X_test)
    
    metrics = {
        'r2_score': r2_score(y_test, y_pred),
        'mae': mean_absolute_error(y_test, y_pred),
        'mse': mean_squared_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'y_pred': y_pred,
        'residuals': y_test - y_pred
    }
    
    return metrics


def plot_baseline_vs_tuned_comparison(baseline_metrics, tuned_metrics, 
                                     model_type='classification', save_path=None):
    """
    Create a bar chart comparing baseline vs tuned model performance.
    
    Parameters:
    -----------
    baseline_metrics : dict
        Metrics from baseline model
    tuned_metrics : dict
        Metrics from tuned model
    model_type : str, default='classification'
        'classification' or 'regression'
    save_path : str or Path, optional
        Path to save the plot
    """
    print(f"\n{'='*60}")
    print(f"CREATING BASELINE VS TUNED COMPARISON - {model_type.upper()}")
    print(f"{'='*60}")
    
    if model_type == 'classification':
        metric_name = 'Accuracy'
        baseline_score = baseline_metrics.get('accuracy', 0)
        tuned_score = tuned_metrics.get('accuracy', 0)
    else:  # regression
        metric_name = 'R² Score'
        baseline_score = baseline_metrics.get('r2_score', 0)
        tuned_score = tuned_metrics.get('r2_score', 0)
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Primary metric comparison
    ax1 = axes[0]
    models = ['Baseline Model', 'Tuned Model']
    scores = [baseline_score, tuned_score]
    colors = ['steelblue', 'coral']
    
    bars = ax1.bar(models, scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Calculate improvement
    improvement = tuned_score - baseline_score
    improvement_pct = (improvement / baseline_score * 100) if baseline_score > 0 else 0
    
    ax1.set_ylabel(metric_name, fontsize=12, fontweight='bold')
    ax1.set_title(f'{metric_name} Comparison: Baseline vs Tuned', 
                  fontsize=14, fontweight='bold', pad=15)
    ax1.set_ylim([0, max(scores) * 1.15])
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add improvement text
    ax1.text(0.5, max(scores) * 1.05, 
             f'Improvement: {improvement:+.4f} ({improvement_pct:+.2f}%)',
             ha='center', fontsize=11, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    # Plot 2: Multiple metrics comparison (if classification)
    if model_type == 'classification':
        ax2 = axes[1]
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score']
        metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        baseline_values = [baseline_metrics.get(m, 0) for m in metrics_to_plot]
        tuned_values = [tuned_metrics.get(m, 0) for m in metrics_to_plot]
        
        x = np.arange(len(metric_labels))
        width = 0.35
        
        bars1 = ax2.bar(x - width/2, baseline_values, width, label='Baseline', 
                       color='steelblue', alpha=0.8)
        bars2 = ax2.bar(x + width/2, tuned_values, width, label='Tuned', 
                       color='coral', alpha=0.8)
        
        ax2.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax2.set_title('All Metrics Comparison', fontsize=14, fontweight='bold', pad=15)
        ax2.set_xticks(x)
        ax2.set_xticklabels(metric_labels)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_ylim([0, 1.05])
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    else:
        # For regression, show MAE, MSE, RMSE comparison
        ax2 = axes[1]
        metrics_to_plot = ['mae', 'mse', 'rmse']
        metric_labels = ['MAE', 'MSE', 'RMSE']
        
        baseline_values = [baseline_metrics.get(m, 0) for m in metrics_to_plot]
        tuned_values = [tuned_metrics.get(m, 0) for m in metrics_to_plot]
        
        x = np.arange(len(metric_labels))
        width = 0.35
        
        bars1 = ax2.bar(x - width/2, baseline_values, width, label='Baseline', 
                       color='steelblue', alpha=0.8)
        bars2 = ax2.bar(x + width/2, tuned_values, width, label='Tuned', 
                       color='coral', alpha=0.8)
        
        ax2.set_ylabel('Error', fontsize=12, fontweight='bold')
        ax2.set_title('Error Metrics Comparison (Lower is Better)', 
                     fontsize=14, fontweight='bold', pad=15)
        ax2.set_xticks(x)
        ax2.set_xticklabels(metric_labels)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    # Save plot
    if save_path is None:
        paths = get_project_paths()
        save_path = paths['results_dir'] / f'baseline_vs_tuned_{model_type}.png'
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to: {save_path}")
    plt.close()
    
    print(f"{'='*60}\n")


def plot_residual_errors(y_test, y_pred, save_path=None):
    """
    Create a scatter plot of residual errors for yield prediction.
    
    Parameters:
    -----------
    y_test : array-like
        True target values
    y_pred : array-like
        Predicted target values
    save_path : str or Path, optional
        Path to save the plot
    """
    print(f"\n{'='*60}")
    print("CREATING RESIDUAL ERRORS PLOT")
    print(f"{'='*60}")
    
    residuals = y_test - y_pred
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Residuals vs Predicted
    ax1 = axes[0, 0]
    scatter = ax1.scatter(y_pred, residuals, alpha=0.6, s=50, c='steelblue', 
                         edgecolors='black', linewidth=0.5)
    ax1.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax1.set_xlabel('Predicted Yield', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Residuals (Actual - Predicted)', fontsize=12, fontweight='bold')
    ax1.set_title('Residuals vs Predicted Values', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Add statistics
    mean_residual = np.mean(residuals)
    std_residual = np.std(residuals)
    ax1.text(0.05, 0.95, f'Mean: {mean_residual:.4f}\nStd: {std_residual:.4f}',
             transform=ax1.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    # Plot 2: Residuals distribution (histogram)
    ax2 = axes[0, 1]
    ax2.hist(residuals, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
    ax2.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax2.axvline(x=mean_residual, color='green', linestyle='--', linewidth=2, 
                label=f'Mean: {mean_residual:.4f}')
    ax2.set_xlabel('Residuals', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax2.set_title('Distribution of Residuals', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Q-Q plot (check for normality)
    ax3 = axes[1, 0]
    if HAS_SCIPY:
        stats.probplot(residuals, dist="norm", plot=ax3)
        ax3.set_title('Q-Q Plot (Normality Check)', fontsize=14, fontweight='bold')
    else:
        ax3.text(0.5, 0.5, 'Q-Q Plot\n(scipy not available)', 
                ha='center', va='center', fontsize=12, transform=ax3.transAxes)
        ax3.set_title('Q-Q Plot (Normality Check)', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Residuals vs Actual
    ax4 = axes[1, 1]
    ax4.scatter(y_test, residuals, alpha=0.6, s=50, c='coral', 
               edgecolors='black', linewidth=0.5)
    ax4.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax4.set_xlabel('Actual Yield', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Residuals', fontsize=12, fontweight='bold')
    ax4.set_title('Residuals vs Actual Values', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    if save_path is None:
        paths = get_project_paths()
        save_path = paths['results_dir'] / 'residual_errors_yield.png'
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Residual errors plot saved to: {save_path}")
    plt.close()
    
    print(f"{'='*60}\n")


def plot_feature_importance_comparison(model, feature_names, model_name='Model', 
                                      top_n=10, save_path=None):
    """
    Create a feature importance bar chart.
    
    Parameters:
    -----------
    model : sklearn model
        Trained model with feature_importances_ attribute
    feature_names : list
        Names of features
    model_name : str, default='Model'
        Name of the model
    top_n : int, default=10
        Number of top features to display
    save_path : str or Path, optional
        Path to save the plot
    """
    print(f"\n{'='*60}")
    print(f"CREATING FEATURE IMPORTANCE CHART - {model_name.upper()}")
    print(f"{'='*60}")
    
    # Get feature importances
    importances = model.feature_importances_
    
    # Create DataFrame
    feature_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    # Get top N features
    top_features = feature_df.head(top_n)
    
    print(f"\nTop {top_n} Most Important Features:")
    print("-" * 60)
    for idx, row in top_features.iterrows():
        print(f"{row['feature']:<30} {row['importance']:.6f}")
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = sns.color_palette("viridis", len(top_features))
    bars = ax.barh(range(len(top_features)), top_features['importance'], color=colors)
    
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'])
    ax.set_xlabel('Feature Importance', fontsize=12, fontweight='bold')
    ax.set_title(f'Top {top_n} Feature Importances - {model_name}', 
                fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()  # Highest importance at top
    
    # Add value labels on bars
    for i, (idx, row) in enumerate(top_features.iterrows()):
        ax.text(row['importance'], i, f' {row["importance"]:.4f}', 
                va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot
    if save_path is None:
        paths = get_project_paths()
        save_path = paths['feature_importance_dir'] / f'feature_importance_{model_name.lower()}.png'
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Feature importance plot saved to: {save_path}")
    plt.close()
    
    print(f"{'='*60}\n")
    
    return feature_df


def generate_summary_metrics(recommendation_baseline=None, recommendation_tuned=None,
                             yield_baseline=None, yield_tuned=None, save_path=None):
    """
    Generate a comprehensive summary metrics file.
    
    Parameters:
    -----------
    recommendation_baseline : dict, optional
        Baseline classification model metrics
    recommendation_tuned : dict, optional
        Tuned classification model metrics
    yield_baseline : dict, optional
        Baseline regression model metrics
    yield_tuned : dict, optional
        Tuned regression model metrics
    save_path : str or Path, optional
        Path to save the summary file
    """
    print(f"\n{'='*60}")
    print("GENERATING SUMMARY METRICS FILE")
    print(f"{'='*60}")
    
    if save_path is None:
        paths = get_project_paths()
        save_path = paths['results_dir'] / 'summary_metrics.txt'
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("MACHINE LEARNING MODEL EVALUATION SUMMARY\n")
        f.write("Smart Farming System - Crop Recommendation & Yield Prediction\n")
        f.write("="*80 + "\n\n")
        
        # Crop Recommendation Model Section
        f.write("="*80 + "\n")
        f.write("CROP RECOMMENDATION MODEL (CLASSIFICATION)\n")
        f.write("="*80 + "\n\n")
        
        if recommendation_baseline or recommendation_tuned:
            f.write("PERFORMANCE METRICS:\n")
            f.write("-"*80 + "\n")
            f.write(f"{'Metric':<25} {'Baseline':<15} {'Tuned':<15} {'Improvement':<15}\n")
            f.write("-"*80 + "\n")
            
            if recommendation_baseline and recommendation_tuned:
                metrics = ['accuracy', 'precision', 'recall', 'f1_score']
                labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
                
                for metric, label in zip(metrics, labels):
                    baseline_val = recommendation_baseline.get(metric, 0)
                    tuned_val = recommendation_tuned.get(metric, 0)
                    improvement = tuned_val - baseline_val
                    improvement_pct = (improvement / baseline_val * 100) if baseline_val > 0 else 0
                    
                    f.write(f"{label:<25} {baseline_val:<15.4f} {tuned_val:<15.4f} "
                           f"{improvement:+.4f} ({improvement_pct:+.2f}%)\n")
            elif recommendation_baseline:
                for metric, label in zip(['accuracy', 'precision', 'recall', 'f1_score'],
                                       ['Accuracy', 'Precision', 'Recall', 'F1-Score']):
                    val = recommendation_baseline.get(metric, 0)
                    f.write(f"{label:<25} {val:<15.4f} {'N/A':<15} {'N/A':<15}\n")
            elif recommendation_tuned:
                for metric, label in zip(['accuracy', 'precision', 'recall', 'f1_score'],
                                       ['Accuracy', 'Precision', 'Recall', 'F1-Score']):
                    val = recommendation_tuned.get(metric, 0)
                    f.write(f"{label:<25} {'N/A':<15} {val:<15.4f} {'N/A':<15}\n")
        else:
            f.write("No evaluation data available for crop recommendation model.\n")
        
        f.write("\n")
        
        # Crop Yield Prediction Model Section
        f.write("="*80 + "\n")
        f.write("CROP YIELD PREDICTION MODEL (REGRESSION)\n")
        f.write("="*80 + "\n\n")
        
        if yield_baseline or yield_tuned:
            f.write("PERFORMANCE METRICS:\n")
            f.write("-"*80 + "\n")
            f.write(f"{'Metric':<25} {'Baseline':<15} {'Tuned':<15} {'Improvement':<15}\n")
            f.write("-"*80 + "\n")
            
            if yield_baseline and yield_tuned:
                metrics = ['r2_score', 'mae', 'mse', 'rmse']
                labels = ['R² Score', 'MAE', 'MSE', 'RMSE']
                
                for metric, label in zip(metrics, labels):
                    baseline_val = yield_baseline.get(metric, 0)
                    tuned_val = yield_tuned.get(metric, 0)
                    
                    if metric == 'r2_score':
                        improvement = tuned_val - baseline_val
                        improvement_pct = (improvement / baseline_val * 100) if baseline_val > 0 else 0
                        f.write(f"{label:<25} {baseline_val:<15.4f} {tuned_val:<15.4f} "
                               f"{improvement:+.4f} ({improvement_pct:+.2f}%)\n")
                    else:
                        improvement = baseline_val - tuned_val  # Lower is better
                        improvement_pct = (improvement / baseline_val * 100) if baseline_val > 0 else 0
                        f.write(f"{label:<25} {baseline_val:<15.4f} {tuned_val:<15.4f} "
                               f"{improvement:+.4f} ({improvement_pct:+.2f}% reduction)\n")
            elif yield_baseline:
                for metric, label in zip(['r2_score', 'mae', 'mse', 'rmse'],
                                       ['R² Score', 'MAE', 'MSE', 'RMSE']):
                    val = yield_baseline.get(metric, 0)
                    f.write(f"{label:<25} {val:<15.4f} {'N/A':<15} {'N/A':<15}\n")
            elif yield_tuned:
                for metric, label in zip(['r2_score', 'mae', 'mse', 'rmse'],
                                       ['R² Score', 'MAE', 'MSE', 'RMSE']):
                    val = yield_tuned.get(metric, 0)
                    f.write(f"{label:<25} {'N/A':<15} {val:<15.4f} {'N/A':<15}\n")
        else:
            f.write("No evaluation data available for crop yield prediction model.\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("SUMMARY\n")
        f.write("="*80 + "\n")
        f.write("This report summarizes the performance of baseline and hyperparameter-tuned\n")
        f.write("machine learning models for crop recommendation and yield prediction.\n\n")
        f.write("For best results, use the tuned models saved in the models/ directory.\n")
        f.write("="*80 + "\n")
    
    print(f"Summary metrics saved to: {save_path}")
    print(f"{'='*60}\n")


def run_complete_evaluation():
    """
    Complete evaluation pipeline: Load models, evaluate, create visualizations, and generate summary.
    """
    print("="*60)
    print("COMPLETE MODEL EVALUATION AND VISUALIZATION")
    print("="*60)
    
    paths = get_project_paths()
    
    # Try to load models
    baseline_rec_path = paths['models_dir'] / 'crop_recommendation.pkl'
    tuned_rec_path = paths['models_dir'] / 'crop_recommendation_tuned.pkl'
    baseline_yield_path = paths['models_dir'] / 'crop_yield_predictor.pkl'
    tuned_yield_path = paths['models_dir'] / 'crop_yield_tuned.pkl'
    
    recommendation_baseline_metrics = None
    recommendation_tuned_metrics = None
    yield_baseline_metrics = None
    yield_tuned_metrics = None
    yield_test_data = None
    
    # Load and preprocess test data for crop recommendation
    print("\n" + "="*60)
    print("LOADING TEST DATA FOR EVALUATION")
    print("="*60)
    
    from load_data import load_dataset
    from preprocess_data import preprocess_pipeline
    from yield_prediction_model import prepare_yield_dataset
    
    # Prepare recommendation test data
    try:
        df_rec = load_dataset('crop_recommendation.csv', display_info=False)
        preprocessed_rec = preprocess_pipeline(
            df=df_rec,
            target_col='label',
            handle_missing='median',
            remove_duplicates=True,
            encoding_type='label',
            scale_features_flag=True,
            test_size=0.2,
            random_state=42
        )
        X_test_rec = preprocessed_rec['X_test']
        y_test_rec = preprocessed_rec['y_test']
        rec_feature_names = X_test_rec.columns.tolist() if hasattr(X_test_rec, 'columns') else None
        print("Crop recommendation test data loaded.")
    except Exception as e:
        print(f"Warning: Could not load crop recommendation data: {e}")
        X_test_rec = None
        y_test_rec = None
        rec_feature_names = None
    
    # Prepare yield prediction test data
    try:
        df_yield = load_dataset('crop_production.csv', display_info=False)
        df_yield = prepare_yield_dataset(df_yield, target_col='yield')
        preprocessed_yield = preprocess_pipeline(
            df=df_yield,
            target_col='yield',
            handle_missing='median',
            remove_duplicates=True,
            encoding_type='label',
            scale_features_flag=True,
            test_size=0.2,
            random_state=42
        )
        X_test_yield = preprocessed_yield['X_test']
        y_test_yield = preprocessed_yield['y_test']
        yield_feature_names = X_test_yield.columns.tolist() if hasattr(X_test_yield, 'columns') else None
        yield_test_data = {'X_test': X_test_yield, 'y_test': y_test_yield}
        print("Crop yield prediction test data loaded.")
    except Exception as e:
        print(f"Warning: Could not load crop yield data: {e}")
        X_test_yield = None
        y_test_yield = None
        yield_feature_names = None
    
    # Evaluate crop recommendation models
    print("\n" + "="*60)
    print("EVALUATING CROP RECOMMENDATION MODELS")
    print("="*60)
    
    # Load and evaluate baseline
    baseline_rec_data = load_model(baseline_rec_path)
    if baseline_rec_data and X_test_rec is not None:
        print("Evaluating baseline crop recommendation model...")
        model = baseline_rec_data['model']
        scaler = baseline_rec_data.get('scaler')
        
        # Apply scaler if needed
        if scaler:
            X_test_scaled = scaler.transform(X_test_rec)
        else:
            X_test_scaled = X_test_rec
        
        recommendation_baseline_metrics = evaluate_classification_model(model, X_test_scaled, y_test_rec)
    
    # Load and evaluate tuned
    tuned_rec_data = load_model(tuned_rec_path)
    if tuned_rec_data and X_test_rec is not None:
        print("Evaluating tuned crop recommendation model...")
        model = tuned_rec_data['model']
        scaler = tuned_rec_data.get('scaler')
        
        # Apply scaler if needed
        if scaler:
            X_test_scaled = scaler.transform(X_test_rec)
        else:
            X_test_scaled = X_test_rec
        
        recommendation_tuned_metrics = evaluate_classification_model(model, X_test_scaled, y_test_rec)
        
        # Plot feature importance for tuned model
        if rec_feature_names:
            plot_feature_importance_comparison(
                model, rec_feature_names, 
                model_name='Crop Recommendation (Tuned)',
                top_n=10
            )
    
    # Evaluate yield prediction models
    print("\n" + "="*60)
    print("EVALUATING CROP YIELD PREDICTION MODELS")
    print("="*60)
    
    # Load and evaluate baseline
    baseline_yield_data = load_model(baseline_yield_path)
    if baseline_yield_data and X_test_yield is not None:
        print("Evaluating baseline yield prediction model...")
        model = baseline_yield_data['model']
        scaler = baseline_yield_data.get('scaler')
        
        # Apply scaler if needed
        if scaler:
            X_test_scaled = scaler.transform(X_test_yield)
        else:
            X_test_scaled = X_test_yield
        
        yield_baseline_metrics = evaluate_regression_model(model, X_test_scaled, y_test_yield)
    
    # Load and evaluate tuned
    tuned_yield_data = load_model(tuned_yield_path)
    if tuned_yield_data and X_test_yield is not None:
        print("Evaluating tuned yield prediction model...")
        model = tuned_yield_data['model']
        scaler = tuned_yield_data.get('scaler')
        
        # Apply scaler if needed
        if scaler:
            X_test_scaled = scaler.transform(X_test_yield)
        else:
            X_test_scaled = X_test_yield
        
        yield_tuned_metrics = evaluate_regression_model(model, X_test_scaled, y_test_yield)
        
        # Plot feature importance for tuned model
        if yield_feature_names:
            plot_feature_importance_comparison(
                model, yield_feature_names,
                model_name='Crop Yield Prediction (Tuned)',
                top_n=10
            )
    
    # Create comparison visualizations
    print("\n" + "="*60)
    print("CREATING COMPARISON VISUALIZATIONS")
    print("="*60)
    
    if recommendation_baseline_metrics and recommendation_tuned_metrics:
        plot_baseline_vs_tuned_comparison(
            recommendation_baseline_metrics, recommendation_tuned_metrics,
            model_type='classification'
        )
    
    if yield_baseline_metrics and yield_tuned_metrics:
        plot_baseline_vs_tuned_comparison(
            yield_baseline_metrics, yield_tuned_metrics,
            model_type='regression'
        )
        
        # Plot residual errors for tuned model
        if 'y_pred' in yield_tuned_metrics:
            plot_residual_errors(y_test_yield, yield_tuned_metrics['y_pred'])
    
    # Generate summary
    generate_summary_metrics(
        recommendation_baseline=recommendation_baseline_metrics,
        recommendation_tuned=recommendation_tuned_metrics,
        yield_baseline=yield_baseline_metrics,
        yield_tuned=yield_tuned_metrics
    )
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE!")
    print("="*60)
    print("\nGenerated files:")
    if recommendation_baseline_metrics and recommendation_tuned_metrics:
        print("  - results/baseline_vs_tuned_classification.png")
    if yield_baseline_metrics and yield_tuned_metrics:
        print("  - results/baseline_vs_tuned_regression.png")
        print("  - results/residual_errors_yield.png")
    print("  - results/summary_metrics.txt")
    if rec_feature_names and tuned_rec_data:
        print("  - results/feature_importance/feature_importance_crop_recommendation_(tuned).png")
    if yield_feature_names and tuned_yield_data:
        print("  - results/feature_importance/feature_importance_crop_yield_prediction_(tuned).png")
    print("="*60 + "\n")


if __name__ == "__main__":
    """
    Main execution block.
    This script is designed to be run after both baseline and tuned models are trained.
    """
    print("\n" + "="*60)
    print("MODEL EVALUATION AND VISUALIZATION")
    print("="*60)
    print("\nNote: This script requires trained models.")
    print("Please ensure you have run:")
    print("  - scripts/crop_recommendation_model.py")
    print("  - scripts/yield_prediction_model.py")
    print("  - scripts/model_tuning.py")
    print("\nRunning complete evaluation pipeline...\n")
    
    # Run the complete evaluation
    run_complete_evaluation()


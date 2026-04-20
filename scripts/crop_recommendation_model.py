"""
Crop Recommendation Model
=========================

This script builds and trains a RandomForestClassifier for crop recommendation.
It evaluates the model using accuracy, precision, recall, F1-score, and generates
a confusion matrix visualization.

Author: ML Engineer
Date: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from pathlib import Path
import joblib
import sys
import warnings

warnings.filterwarnings('ignore')

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 10)
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
        'metrics_dir': project_root / 'results' / 'metrics'
    }


def load_and_preprocess_data():
    """
    Load and preprocess the crop recommendation dataset.
    
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
    df = load_dataset('crop_recommendation.csv', display_info=False)
    
    # Preprocess data
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
    
    return preprocessed


def train_crop_recommendation_model(X_train, y_train, n_estimators=100, random_state=42, **kwargs):
    """
    Train a RandomForestClassifier for crop recommendation.
    
    Parameters:
    -----------
    X_train : pd.DataFrame or np.array
        Training features
    y_train : pd.Series or np.array
        Training target labels
    n_estimators : int, default=100
        Number of trees in the random forest
    random_state : int, default=42
        Random seed for reproducibility
    **kwargs : dict
        Additional parameters for RandomForestClassifier
    
    Returns:
    --------
    RandomForestClassifier
        Trained model
    """
    print(f"\n{'='*60}")
    print("TRAINING RANDOM FOREST CLASSIFIER")
    print(f"{'='*60}")
    print(f"Training samples: {len(X_train)}")
    print(f"Features: {X_train.shape[1]}")
    print(f"Number of classes: {len(np.unique(y_train))}")
    print(f"Number of estimators (trees): {n_estimators}")
    
    # Initialize RandomForestClassifier
    model = RandomForestClassifier(
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


def evaluate_model(model, X_test, y_test, class_names=None):
    """
    Evaluate the trained model and return metrics.
    
    Parameters:
    -----------
    model : RandomForestClassifier
        Trained model
    X_test : pd.DataFrame or np.array
        Test features
    y_test : pd.Series or np.array
        Test target labels
    class_names : list, optional
        Names of classes for display
    
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
    y_pred_proba = model.predict_proba(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    # Per-class metrics
    precision_per_class = precision_score(y_test, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_test, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_test, y_pred, average=None, zero_division=0)
    
    # Create metrics dictionary
    metrics = {
        'accuracy': accuracy,
        'precision_weighted': precision,
        'recall_weighted': recall,
        'f1_weighted': f1,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'f1_per_class': f1_per_class,
        'y_true': y_test,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }
    
    # Print metrics
    print(f"\n{'Metric':<25} {'Score':<15}")
    print("-" * 40)
    print(f"{'Accuracy':<25} {accuracy:.4f}")
    print(f"{'Precision (Weighted)':<25} {precision:.4f}")
    print(f"{'Recall (Weighted)':<25} {recall:.4f}")
    print(f"{'F1-Score (Weighted)':<25} {f1:.4f}")
    
    # Print classification report
    print(f"\n{'='*60}")
    print("CLASSIFICATION REPORT")
    print(f"{'='*60}")
    if class_names is not None:
        print(classification_report(y_test, y_pred, target_names=class_names))
    else:
        print(classification_report(y_test, y_pred))
    
    return metrics


def plot_confusion_matrix(y_true, y_pred, class_names=None, save_path=None):
    """
    Create and save a confusion matrix visualization.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    class_names : list, optional
        Names of classes for display
    save_path : str or Path, optional
        Path to save the plot
    """
    print(f"\n{'='*60}")
    print("GENERATING CONFUSION MATRIX")
    print(f"{'='*60}")
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create figure
    plt.figure(figsize=(14, 12))
    
    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names if class_names else None,
                yticklabels=class_names if class_names else None,
                cbar_kws={"shrink": 0.8},
                linewidths=0.5)
    
    plt.title('Confusion Matrix - Crop Recommendation Model', 
              fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    if save_path is None:
        paths = get_project_paths()
        save_path = paths['metrics_dir'] / 'confusion_matrix_crop_recommendation.png'
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to: {save_path}")
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
    if save_path is None:
        paths = get_project_paths()
        save_path = paths['metrics_dir'] / 'crop_recommendation_metrics.txt'
    
    with open(save_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("CROP RECOMMENDATION MODEL - EVALUATION METRICS\n")
        f.write("="*60 + "\n\n")
        
        f.write("OVERALL METRICS:\n")
        f.write("-" * 60 + "\n")
        f.write(f"Accuracy:           {metrics['accuracy']:.4f}\n")
        f.write(f"Precision (Weighted): {metrics['precision_weighted']:.4f}\n")
        f.write(f"Recall (Weighted):    {metrics['recall_weighted']:.4f}\n")
        f.write(f"F1-Score (Weighted):  {metrics['f1_weighted']:.4f}\n\n")
        
        if 'precision_per_class' in metrics:
            f.write("PER-CLASS METRICS:\n")
            f.write("-" * 60 + "\n")
            f.write(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}\n")
            f.write("-" * 60 + "\n")
            for i, (p, r, f1) in enumerate(zip(
                metrics['precision_per_class'],
                metrics['recall_per_class'],
                metrics['f1_per_class']
            )):
                f.write(f"Class {i:<12} {p:<12.4f} {r:<12.4f} {f1:<12.4f}\n")
    
    print(f"Metrics saved to: {save_path}")


def save_model(model, encoders, scaler, save_path=None):
    """
    Save the trained model and preprocessing objects.
    
    Parameters:
    -----------
    model : RandomForestClassifier
        Trained model
    encoders : dict
        Dictionary of encoders used in preprocessing
    scaler : StandardScaler
        Scaler used for feature normalization
    save_path : str or Path, optional
        Path to save the model
    """
    if save_path is None:
        paths = get_project_paths()
        save_path = paths['models_dir'] / 'crop_recommendation.pkl'
    
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


def train_and_evaluate_crop_recommendation_model(n_estimators=100, random_state=42, **kwargs):
    """
    Complete pipeline: Load data, train model, evaluate, and save results.
    
    Parameters:
    -----------
    n_estimators : int, default=100
        Number of trees in random forest
    random_state : int, default=42
        Random seed for reproducibility
    **kwargs : dict
        Additional parameters for RandomForestClassifier
    
    Returns:
    --------
    dict
        Dictionary containing model, metrics, and preprocessing objects
    """
    print("\n" + "="*60)
    print("CROP RECOMMENDATION MODEL - COMPLETE PIPELINE")
    print("="*60)
    
    # Step 1: Load and preprocess data
    preprocessed = load_and_preprocess_data()
    
    X_train = preprocessed['X_train']
    X_test = preprocessed['X_test']
    y_train = preprocessed['y_train']
    y_test = preprocessed['y_test']
    encoders = preprocessed['encoders']
    scaler = preprocessed['scaler']
    
    # Step 2: Train model
    model = train_crop_recommendation_model(
        X_train, y_train, 
        n_estimators=n_estimators,
        random_state=random_state,
        **kwargs
    )
    
    # Get class names if available (from label encoder)
    class_names = None
    if 'label' in encoders:
        class_names = encoders['label'].classes_.tolist()
        print(f"\nClass names: {class_names}")
    
    # Step 3: Evaluate model
    metrics = evaluate_model(model, X_test, y_test, class_names=class_names)
    
    # Step 4: Generate confusion matrix
    plot_confusion_matrix(y_test, metrics['y_pred'], class_names=class_names)
    
    # Step 5: Save metrics
    save_metrics(metrics)
    
    # Step 6: Save model
    save_model(model, encoders, scaler)
    
    # Summary
    print("\n" + "="*60)
    print("MODEL TRAINING AND EVALUATION COMPLETE!")
    print("="*60)
    print(f"\nFinal Model Performance:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision_weighted']:.4f}")
    print(f"  Recall:    {metrics['recall_weighted']:.4f}")
    print(f"  F1-Score:  {metrics['f1_weighted']:.4f}")
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
    results = train_and_evaluate_crop_recommendation_model(
        n_estimators=100,
        random_state=42,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1
    )
    
    print("\nModel training completed successfully!")
    print("Check the following directories for outputs:")
    print("  - models/crop_recommendation.pkl")
    print("  - results/metrics/confusion_matrix_crop_recommendation.png")
    print("  - results/metrics/crop_recommendation_metrics.txt")


"""
Explainable AI (XAI) Analysis using SHAP
=========================================

This script performs explainable AI analysis on the trained crop recommendation
model using SHAP (SHapley Additive exPlanations) values to understand feature
importance and model predictions.

Author: ML Engineer
Date: 2025
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import warnings
import sys

warnings.filterwarnings('ignore')

# Try to import SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("[WARNING] SHAP library not available. Install with: pip install shap")


def load_model_and_data(model_path='models/best_crop_model.pkl', 
                        data_path='data/crop_recommendation.csv'):
    """
    Load the trained model and dataset.
    
    Parameters:
    -----------
    model_path : str
        Path to the saved model file
    data_path : str
        Path to the dataset file
    
    Returns:
    --------
    tuple
        (model, label_encoder, scaler, X_data, feature_names)
    """
    print("="*70)
    print("LOADING MODEL AND DATA")
    print("="*70)
    
    # Load model
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print(f"\n[INFO] Loading model from: {model_path}")
    model_data = joblib.load(model_path)
    
    model = model_data['model']
    label_encoder = model_data.get('label_encoder')
    scaler = model_data.get('scaler')
    
    print(f"[OK] Model loaded: {type(model).__name__}")
    
    # Load dataset
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    
    print(f"\n[INFO] Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"[OK] Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Prepare feature data
    feature_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    
    # Handle case sensitivity
    available_cols = []
    for col in df.columns:
        if col.lower() in [f.lower() for f in feature_cols]:
            available_cols.append(col)
    
    if len(available_cols) != len(feature_cols):
        # Use all columns except target
        target_col = 'label' if 'label' in df.columns else 'Label' if 'Label' in df.columns else 'crop'
        available_cols = [col for col in df.columns if col != target_col]
    
    X_data = df[available_cols].copy()
    
    # Scale if scaler is available
    if scaler:
        print("\n[INFO] Scaling features using model's scaler...")
        X_data_scaled = scaler.transform(X_data)
        X_data = pd.DataFrame(X_data_scaled, columns=available_cols)
        print("[OK] Features scaled")
    
    print(f"\n[OK] Features prepared: {X_data.shape[0]} samples, {X_data.shape[1]} features")
    print(f"[INFO] Feature names: {list(X_data.columns)}")
    
    return model, label_encoder, scaler, X_data, list(X_data.columns)


def create_shap_explainer(model, X_data, max_samples=100):
    """
    Create SHAP explainer for the model.
    
    Parameters:
    -----------
    model : sklearn model
        Trained model
    X_data : pd.DataFrame
        Feature data
    max_samples : int, default=100
        Maximum number of samples to use for explanation (for speed)
    
    Returns:
    --------
    shap.Explainer
        SHAP explainer object
    """
    print("\n" + "="*70)
    print("CREATING SHAP EXPLAINER")
    print("="*70)
    
    if not SHAP_AVAILABLE:
        raise ImportError("SHAP library is not available. Install with: pip install shap")
    
    # Sample data for faster computation (optional)
    if len(X_data) > max_samples:
        print(f"\n[INFO] Sampling {max_samples} rows from {len(X_data)} total samples for faster computation...")
        X_sample = X_data.sample(n=max_samples, random_state=42)
        print(f"[OK] Using {len(X_sample)} samples for SHAP analysis")
    else:
        X_sample = X_data
        print(f"\n[INFO] Using all {len(X_sample)} samples for SHAP analysis")
    
    print("\n[INFO] Creating SHAP explainer...")
    
    try:
        # Try TreeExplainer for tree-based models
        if isinstance(model, (type(shap.TreeExplainer(model).model)) if hasattr(shap, 'TreeExplainer') else None):
            pass  # Will try below
        
        # Check if model is tree-based (RandomForest, DecisionTree, XGBoost, etc.)
        model_type = type(model).__name__
        tree_models = ['RandomForestClassifier', 'RandomForestRegressor',
                      'DecisionTreeClassifier', 'DecisionTreeRegressor',
                      'XGBClassifier', 'XGBRegressor', 'GradientBoostingClassifier']
        
        if any(model_type.startswith(name.split('Classifier')[0]) for name in tree_models):
            print("[INFO] Detected tree-based model, using TreeExplainer...")
            explainer = shap.TreeExplainer(model)
        else:
            print("[INFO] Using KernelExplainer (slower but works for all models)...")
            # For non-tree models, use KernelExplainer with a subset of data
            # Create a small background dataset
            background = X_sample.iloc[:min(50, len(X_sample))]
            explainer = shap.KernelExplainer(model.predict_proba, background)
        
        print("[OK] SHAP explainer created successfully")
        
        return explainer, X_sample
        
    except Exception as e:
        print(f"[ERROR] Failed to create TreeExplainer: {str(e)}")
        print("[INFO] Falling back to KernelExplainer...")
        
        try:
            # Fallback to KernelExplainer
            background = X_sample.iloc[:min(50, len(X_sample))]
            explainer = shap.KernelExplainer(model.predict_proba, background)
            print("[OK] KernelExplainer created successfully")
            return explainer, X_sample
        except Exception as e2:
            raise RuntimeError(f"Failed to create SHAP explainer: {str(e2)}\n"
                             f"Original error: {str(e)}")


def calculate_shap_values(explainer, X_data):
    """
    Calculate SHAP values for the dataset.
    
    Parameters:
    -----------
    explainer : shap.Explainer
        SHAP explainer object
    X_data : pd.DataFrame
        Feature data
    
    Returns:
    --------
    numpy array
        SHAP values
    """
    print("\n[INFO] Calculating SHAP values...")
    print("[INFO] This may take a while depending on dataset size...")
    
    try:
        # Calculate SHAP values
        if isinstance(explainer, shap.TreeExplainer):
            shap_values = explainer.shap_values(X_data)
            print("[OK] SHAP values calculated using TreeExplainer")
        else:
            # For KernelExplainer, calculate for a sample
            shap_values = explainer.shap_values(X_data.iloc[:100])  # Limit for performance
            print("[OK] SHAP values calculated using KernelExplainer")
        
        # Handle multi-class outputs (SHAP returns list or 3D array for multi-class)
        if isinstance(shap_values, list):
            # Convert list to numpy array: shape will be (n_classes, n_samples, n_features)
            shap_values = np.array(shap_values)
            # Reshape to (n_samples, n_features, n_classes) for easier processing
            shap_values = np.transpose(shap_values, (1, 2, 0))
            print("[INFO] Multi-class model detected: converted list to array")
        
        print(f"[OK] SHAP values shape: {shap_values.shape}")
        
        return shap_values
        
    except Exception as e:
        raise RuntimeError(f"Failed to calculate SHAP values: {str(e)}")


def plot_feature_importance(shap_values, feature_names, save_path='results/feature_importance_bar.png'):
    """
    Create and save a bar plot showing global feature importance.
    
    Parameters:
    -----------
    shap_values : numpy array
        SHAP values
    feature_names : list
        Names of features
    save_path : str
        Path to save the plot
    """
    print("\n" + "="*70)
    print("GENERATING FEATURE IMPORTANCE PLOT")
    print("="*70)
    
    print("\n[INFO] Creating feature importance bar plot...")
    
    # Calculate mean absolute SHAP values for each feature
    print(f"[DEBUG] SHAP values shape: {shap_values.shape}")
    print(f"[DEBUG] Number of features: {len(feature_names)}")
    
    # Handle different SHAP value shapes
    if len(shap_values.shape) == 3:
        # Multi-class: shape is (samples, features, classes)
        # Average across samples (axis 0) and classes (axis 2), keep features (axis 1)
        mean_shap = np.mean(np.abs(shap_values), axis=(0, 2))
        print(f"[DEBUG] Multi-class detected: averaged across samples and classes")
    elif len(shap_values.shape) == 2:
        # Binary or regression: shape is (samples, features)
        # Average across samples (axis 0)
        mean_shap = np.mean(np.abs(shap_values), axis=0)
        print(f"[DEBUG] Binary/regression detected: averaged across samples")
    elif len(shap_values.shape) == 1:
        # Single prediction: shape is (features,)
        mean_shap = np.abs(shap_values)
        print(f"[DEBUG] Single prediction detected")
    else:
        # Fallback: flatten and handle
        mean_shap = np.mean(np.abs(shap_values))
        print(f"[DEBUG] Fallback: using mean of all values")
    
    # Ensure we have the right number of values
    mean_shap = np.array(mean_shap).flatten()
    
    if len(mean_shap) != len(feature_names):
        print(f"[WARNING] Mismatch: {len(mean_shap)} SHAP values vs {len(feature_names)} features")
        # Try to fix: take first len(feature_names) or pad
        if len(mean_shap) > len(feature_names):
            mean_shap = mean_shap[:len(feature_names)]
        else:
            # This shouldn't happen, but handle it
            print(f"[ERROR] Cannot match SHAP values to features")
            return
    
    print(f"[DEBUG] Final mean_shap shape: {mean_shap.shape}")
    
    # Create feature importance DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names[:len(mean_shap)],
        'importance': mean_shap
    }).sort_values('importance', ascending=False)
    
    # Plot using matplotlib
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.barh(range(len(importance_df)), importance_df['importance'], 
                   color='steelblue', alpha=0.8, edgecolor='black')
    
    ax.set_yticks(range(len(importance_df)))
    ax.set_yticklabels(importance_df['feature'])
    ax.set_xlabel('Mean |SHAP value| (Average Impact on Model Output)', 
                  fontsize=12, fontweight='bold')
    ax.set_title('Global Feature Importance (SHAP Values)', 
                fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()
    
    # Add value labels
    for i, (idx, row) in enumerate(importance_df.iterrows()):
        ax.text(row['importance'], i, f' {row["importance"]:.4f}', 
                va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[OK] Feature importance plot saved to: {save_path}")
    
    # Print feature importance summary
    print("\nFeature Importance Ranking:")
    print("-" * 70)
    for i, (idx, row) in enumerate(importance_df.iterrows(), 1):
        print(f"{i}. {row['feature']:<20} {row['importance']:.6f}")


def plot_shap_summary(shap_values, X_data, feature_names, 
                     save_path='results/shap_summary.png'):
    """
    Create and save a SHAP summary plot.
    
    Parameters:
    -----------
    shap_values : numpy array
        SHAP values
    X_data : pd.DataFrame
        Feature data
    feature_names : list
        Names of features
    save_path : str
        Path to save the plot
    """
    print("\n" + "="*70)
    print("GENERATING SHAP SUMMARY PLOT")
    print("="*70)
    
    print("\n[INFO] Creating SHAP summary plot...")
    
    try:
        # Handle multi-output SHAP values
        print(f"[DEBUG] SHAP values shape before processing: {shap_values.shape}")
        
        # Handle 3D array (multi-class): shape is (samples, features, classes)
        if len(shap_values.shape) == 3:
            # For multi-class, average across classes to get one value per sample per feature
            shap_values_to_plot = np.mean(shap_values, axis=2)  # Average across classes
            print(f"[DEBUG] Multi-class: averaged across classes, new shape: {shap_values_to_plot.shape}")
        elif isinstance(shap_values, list):
            # List of arrays (one per class)
            shap_values_to_plot = np.mean(np.array(shap_values), axis=0)
            print(f"[DEBUG] List format: converted and averaged, new shape: {shap_values_to_plot.shape}")
        else:
            shap_values_to_plot = shap_values
            print(f"[DEBUG] Using SHAP values as-is, shape: {shap_values_to_plot.shape}")
        
        # Limit data for plotting if too large
        n_samples = shap_values_to_plot.shape[0]
        max_plot_samples = min(1000, n_samples)
        
        if n_samples > 1000:
            print(f"[INFO] Sampling {max_plot_samples} rows for summary plot...")
            sample_idx = np.random.choice(n_samples, max_plot_samples, replace=False)
            X_plot = X_data.iloc[sample_idx]
            shap_plot = shap_values_to_plot[sample_idx]
        else:
            X_plot = X_data
            shap_plot = shap_values_to_plot
        
        print(f"[DEBUG] Final plot data - X_plot: {X_plot.shape}, shap_plot: {shap_plot.shape}")
        
        # Create SHAP summary plot
        import matplotlib.pyplot as plt
        
        shap.summary_plot(shap_plot, X_plot, feature_names=feature_names, 
                         show=False, plot_size=(12, 8))
        
        # Save plot
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[OK] SHAP summary plot saved to: {save_path}")
        
    except Exception as e:
        print(f"[WARNING] Could not create SHAP summary plot: {str(e)}")
        print("[INFO] Creating alternative summary visualization...")
        
        # Fallback: Create a simpler summary plot
        import matplotlib.pyplot as plt
        
        # Calculate mean SHAP values
        if len(shap_values.shape) == 3:
            # Multi-class: (samples, features, classes) - average across samples and classes
            mean_shap = np.mean(np.abs(shap_values), axis=(0, 2))
        elif len(shap_values.shape) == 2:
            # Binary/regression: (samples, features) - average across samples
            mean_shap = np.mean(np.abs(shap_values), axis=0)
        else:
            # Single dimension
            mean_shap = np.abs(shap_values) if len(shap_values.shape) == 1 else np.mean(np.abs(shap_values))
        
        # Ensure correct length
        mean_shap = np.array(mean_shap).flatten()
        if len(mean_shap) != len(feature_names):
            mean_shap = mean_shap[:len(feature_names)]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create a scatter-like plot showing SHAP values vs feature values
        n_features = len(feature_names)
        colors = plt.cm.viridis(np.linspace(0, 1, n_features))
        
        y_positions = np.arange(n_features)
        for i, (feat_name, color) in enumerate(zip(feature_names, colors)):
            if len(shap_values.shape) == 2:
                shap_vals = shap_values[:, i]
            else:
                shap_vals = shap_values[i] if len(shap_values.shape) == 1 else shap_values[0, i]
            
            ax.scatter(shap_vals[:200], [i]*min(200, len(shap_vals)), 
                     alpha=0.5, c=[color], s=20, label=feat_name)
        
        ax.set_yticks(y_positions)
        ax.set_yticklabels(feature_names)
        ax.set_xlabel('SHAP value (Impact on model output)', fontsize=12, fontweight='bold')
        ax.set_title('SHAP Summary Plot (Alternative)', fontsize=14, fontweight='bold')
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[OK] Alternative summary plot saved to: {save_path}")


def main():
    """
    Main function for XAI analysis using SHAP.
    """
    print("\n" + "="*70)
    print("EXPLAINABLE AI (XAI) ANALYSIS - SHAP")
    print("="*70)
    
    if not SHAP_AVAILABLE:
        print("\n[ERROR] SHAP library is not available!")
        print("Please install it with: pip install shap")
        print("="*70 + "\n")
        return
    
    try:
        # Load model and data
        model, label_encoder, scaler, X_data, feature_names = \
            load_model_and_data()
        
        # Create SHAP explainer
        explainer, X_sample = create_shap_explainer(model, X_data, max_samples=200)
        
        # Calculate SHAP values
        print("\n[INFO] Calculating SHAP values...")
        shap_values = calculate_shap_values(explainer, X_sample)
        
        # Generate plots
        print("\n[INFO] Saving feature importance plots...")
        plot_feature_importance(shap_values, feature_names)
        plot_shap_summary(shap_values, X_sample, feature_names)
        
        print("\n" + "="*70)
        print("XAI ANALYSIS COMPLETE!")
        print("="*70)
        print("\nGenerated files:")
        print("  - results/feature_importance_bar.png")
        print("  - results/shap_summary.png")
        print("\n" + "="*70 + "\n")
        
    except FileNotFoundError as e:
        print(f"\n[ERROR] File not found: {str(e)}")
        print("Please ensure the model and dataset files exist.")
        print("="*70 + "\n")
        
    except ImportError as e:
        print(f"\n[ERROR] Import error: {str(e)}")
        print("="*70 + "\n")
        
    except RuntimeError as e:
        print(f"\n[ERROR] Runtime error: {str(e)}")
        print("\nThis may happen if:")
        print("  - The model type is not compatible with SHAP")
        print("  - There's an issue with the model structure")
        print("  - Try using a tree-based model (RandomForest, XGBoost) for better SHAP support")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {str(e)}")
        print("\nPlease check:")
        print("  - Model file format")
        print("  - Dataset format")
        print("  - SHAP library installation")
        print("="*70 + "\n")


if __name__ == "__main__":
    main()


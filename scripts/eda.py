"""
Exploratory Data Analysis (EDA) Module
======================================

This script performs comprehensive exploratory data analysis:
- Correlation heatmap for numerical features
- Feature distributions (histograms)
- Boxplots for outlier detection
- Basic statistics summary

All plots are saved in results/eda/ directory.

Author: ML Engineer
Date: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def get_results_path(subfolder='eda'):
    """
    Get the path to the results directory.
    
    Parameters:
    -----------
    subfolder : str, default='eda'
        Subfolder within results directory
    
    Returns:
    --------
    Path
        Path object to the results subfolder
    """
    project_root = Path(__file__).parent.parent
    results_dir = project_root / 'results' / subfolder
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def plot_correlation_heatmap(df, save_path=None, figsize=(12, 10)):
    """
    Generate and save a correlation heatmap for numerical features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    save_path : str or Path, optional
        Path to save the plot. If None, saves to results/eda/
    figsize : tuple, default=(12, 10)
        Figure size (width, height)
    """
    print(f"\n{'='*60}")
    print("GENERATING CORRELATION HEATMAP")
    print(f"{'='*60}")
    
    # Select only numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numerical_cols) < 2:
        print("Warning: Need at least 2 numerical columns for correlation heatmap.")
        return
    
    # Calculate correlation matrix
    corr_matrix = df[numerical_cols].corr()
    
    print(f"Computing correlation for {len(numerical_cols)} numerical features...")
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Create heatmap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Mask upper triangle
    sns.heatmap(corr_matrix, 
                annot=True, 
                fmt='.2f', 
                cmap='coolwarm', 
                center=0,
                square=True,
                linewidths=0.5,
                cbar_kws={"shrink": 0.8},
                mask=mask)
    
    plt.title('Correlation Heatmap of Numerical Features', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    
    # Save plot
    if save_path is None:
        save_path = get_results_path('eda') / 'correlation_heatmap.png'
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Correlation heatmap saved to: {save_path}")
    plt.close()
    
    # Print correlation pairs
    print("\nTop correlations (absolute value > 0.5):")
    print("-" * 60)
    corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.5:
                corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))
    
    if corr_pairs:
        for col1, col2, val in sorted(corr_pairs, key=lambda x: abs(x[2]), reverse=True):
            print(f"  {col1} ↔ {col2}: {val:.3f}")
    else:
        print("  No strong correlations found (|r| > 0.5)")
    
    print(f"{'='*60}\n")


def plot_feature_distributions(df, numerical_cols=None, save_path=None, figsize=(15, 10)):
    """
    Generate histograms for numerical feature distributions.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    numerical_cols : list, optional
        List of numerical columns to plot. If None, auto-detects.
    save_path : str or Path, optional
        Path to save the plot. If None, saves to results/eda/
    figsize : tuple, default=(15, 10)
        Figure size (width, height)
    """
    print(f"\n{'='*60}")
    print("GENERATING FEATURE DISTRIBUTION HISTOGRAMS")
    print(f"{'='*60}")
    
    # Select numerical columns
    if numerical_cols is None:
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numerical_cols) == 0:
        print("No numerical columns found for distribution plots.")
        return
    
    print(f"Plotting distributions for {len(numerical_cols)} features...")
    
    # Calculate grid dimensions
    n_cols = 3
    n_rows = (len(numerical_cols) + n_cols - 1) // n_cols
    
    # Create subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if len(numerical_cols) > 1 else [axes] if n_rows == 1 else axes.flatten()
    
    # Plot histograms
    for idx, col in enumerate(numerical_cols):
        ax = axes[idx]
        
        # Plot histogram with KDE overlay
        sns.histplot(df[col], kde=True, ax=ax, bins=30, color='steelblue', alpha=0.7)
        
        # Add statistics
        mean_val = df[col].mean()
        median_val = df[col].median()
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
        ax.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.2f}')
        
        ax.set_title(f'Distribution of {col}', fontweight='bold', fontsize=12)
        ax.set_xlabel(col, fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(len(numerical_cols), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Feature Distribution Histograms', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save plot
    if save_path is None:
        save_path = get_results_path('eda') / 'feature_distributions.png'
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Feature distributions saved to: {save_path}")
    plt.close()
    print(f"{'='*60}\n")


def plot_boxplots_outliers(df, numerical_cols=None, save_path=None, figsize=(15, 10)):
    """
    Generate boxplots for outlier detection in numerical features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    numerical_cols : list, optional
        List of numerical columns to plot. If None, auto-detects.
    save_path : str or Path, optional
        Path to save the plot. If None, saves to results/eda/
    figsize : tuple, default=(15, 10)
        Figure size (width, height)
    """
    print(f"\n{'='*60}")
    print("GENERATING BOXPLOTS FOR OUTLIER DETECTION")
    print(f"{'='*60}")
    
    # Select numerical columns
    if numerical_cols is None:
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numerical_cols) == 0:
        print("No numerical columns found for boxplots.")
        return
    
    print(f"Plotting boxplots for {len(numerical_cols)} features...")
    
    # Calculate grid dimensions
    n_cols = 3
    n_rows = (len(numerical_cols) + n_cols - 1) // n_cols
    
    # Create subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if len(numerical_cols) > 1 else [axes] if n_rows == 1 else axes.flatten()
    
    outlier_info = {}
    
    # Plot boxplots
    for idx, col in enumerate(numerical_cols):
        ax = axes[idx]
        
        # Create boxplot
        bp = ax.boxplot(df[col].dropna(), vert=True, patch_artist=True,
                       boxprops=dict(facecolor='lightblue', alpha=0.7),
                       medianprops=dict(color='red', linewidth=2),
                       whiskerprops=dict(linewidth=1.5),
                       capprops=dict(linewidth=1.5))
        
        ax.set_title(f'Boxplot of {col}', fontweight='bold', fontsize=12)
        ax.set_ylabel(col, fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Detect outliers using IQR method
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
        outlier_count = len(outliers)
        outlier_percentage = (outlier_count / len(df)) * 100
        
        outlier_info[col] = {
            'count': outlier_count,
            'percentage': outlier_percentage,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        }
    
    # Hide unused subplots
    for idx in range(len(numerical_cols), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Boxplots for Outlier Detection', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save plot
    if save_path is None:
        save_path = get_results_path('eda') / 'boxplots_outliers.png'
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Boxplots saved to: {save_path}")
    plt.close()
    
    # Print outlier summary
    print("\nOutlier Detection Summary (IQR method):")
    print("-" * 60)
    print(f"{'Feature':<20} {'Outliers':<12} {'Percentage':<12} {'Lower Bound':<15} {'Upper Bound':<15}")
    print("-" * 60)
    for col, info in outlier_info.items():
        print(f"{col:<20} {info['count']:<12} {info['percentage']:.2f}%{'':<6} {info['lower_bound']:<15.2f} {info['upper_bound']:<15.2f}")
    
    print(f"{'='*60}\n")


def print_basic_statistics(df, save_path=None):
    """
    Print and save basic statistical summary using df.describe().
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    save_path : str or Path, optional
        Path to save the statistics. If None, saves to results/eda/
    """
    print(f"\n{'='*60}")
    print("BASIC STATISTICAL SUMMARY")
    print(f"{'='*60}")
    
    # Get numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numerical_cols) == 0:
        print("No numerical columns found for statistical summary.")
        return
    
    # Generate descriptive statistics
    stats_df = df[numerical_cols].describe()
    
    print("\nDescriptive Statistics:")
    print("=" * 60)
    print(stats_df)
    
    # Additional statistics
    print("\nAdditional Statistics:")
    print("=" * 60)
    additional_stats = pd.DataFrame({
        'Skewness': df[numerical_cols].skew(),
        'Kurtosis': df[numerical_cols].kurtosis(),
        'Variance': df[numerical_cols].var(),
        'Missing Values': df[numerical_cols].isnull().sum(),
        'Missing %': (df[numerical_cols].isnull().sum() / len(df) * 100)
    })
    print(additional_stats)
    
    # Save statistics to file
    if save_path is None:
        save_path = get_results_path('eda') / 'statistical_summary.txt'
    
    with open(save_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("BASIC STATISTICAL SUMMARY\n")
        f.write("="*60 + "\n\n")
        f.write("Descriptive Statistics:\n")
        f.write("="*60 + "\n")
        f.write(stats_df.to_string())
        f.write("\n\n")
        f.write("Additional Statistics:\n")
        f.write("="*60 + "\n")
        f.write(additional_stats.to_string())
        f.write("\n\n")
        f.write("Dataset Information:\n")
        f.write("="*60 + "\n")
        f.write(f"Total rows: {len(df)}\n")
        f.write(f"Total columns: {len(df.columns)}\n")
        f.write(f"Numerical columns: {len(numerical_cols)}\n")
        f.write(f"Categorical columns: {len(df.columns) - len(numerical_cols)}\n")
    
    print(f"\nStatistical summary saved to: {save_path}")
    print(f"{'='*60}\n")


def perform_complete_eda(df, dataset_name='dataset', include_categorical=False):
    """
    Perform complete EDA analysis on the dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    dataset_name : str, default='dataset'
        Name of the dataset (used for saving files)
    include_categorical : bool, default=False
        Whether to include categorical analysis
    
    Returns:
    --------
    dict
        Dictionary containing paths to saved files and summary statistics
    """
    print("\n" + "="*60)
    print(f"COMPLETE EDA ANALYSIS: {dataset_name.upper()}")
    print("="*60)
    
    # Get numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Ensure results directory exists
    results_dir = get_results_path('eda')
    
    # 1. Basic Statistics
    stats_path = results_dir / f'{dataset_name}_statistical_summary.txt'
    print_basic_statistics(df, save_path=stats_path)
    
    # 2. Correlation Heatmap
    if len(numerical_cols) >= 2:
        corr_path = results_dir / f'{dataset_name}_correlation_heatmap.png'
        plot_correlation_heatmap(df, save_path=corr_path)
    
    # 3. Feature Distributions
    if len(numerical_cols) > 0:
        dist_path = results_dir / f'{dataset_name}_feature_distributions.png'
        plot_feature_distributions(df, numerical_cols=numerical_cols, save_path=dist_path)
    
    # 4. Boxplots for Outliers
    if len(numerical_cols) > 0:
        boxplot_path = results_dir / f'{dataset_name}_boxplots_outliers.png'
        plot_boxplots_outliers(df, numerical_cols=numerical_cols, save_path=boxplot_path)
    
    # Summary
    print("\n" + "="*60)
    print("EDA ANALYSIS COMPLETE!")
    print("="*60)
    print(f"\nAll plots and statistics saved to: {results_dir}")
    print(f"\nGenerated files:")
    print(f"  - Statistical summary: {stats_path.name}")
    if len(numerical_cols) >= 2:
        print(f"  - Correlation heatmap: {corr_path.name}")
    if len(numerical_cols) > 0:
        print(f"  - Feature distributions: {dist_path.name}")
        print(f"  - Boxplots: {boxplot_path.name}")
    print("="*60 + "\n")
    
    return {
        'stats_path': stats_path,
        'corr_path': corr_path if len(numerical_cols) >= 2 else None,
        'dist_path': dist_path if len(numerical_cols) > 0 else None,
        'boxplot_path': boxplot_path if len(numerical_cols) > 0 else None
    }


if __name__ == "__main__":
    """
    Example usage of the EDA module.
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
        print("Exploratory Data Analysis - Example Usage")
        print("="*60)
        
        # Load dataset
        df = load_dataset('crop_recommendation.csv', display_info=False)
        
        # Perform complete EDA
        eda_results = perform_complete_eda(df, dataset_name='crop_recommendation')
        
        print("\nEDA analysis completed successfully!")
        
    except ImportError:
        print("Note: load_data module not found. Using standalone EDA functions.")
        print("Please import this module and use the functions directly in your scripts.")


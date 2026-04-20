"""
Data Loading Module
===================

This script loads datasets into Pandas DataFrames and displays basic information:
- Dataset shape (rows, columns)
- Column names
- First 5 rows

Author: ML Engineer
Date: 2025
"""

import pandas as pd
import os
from pathlib import Path


def get_data_path(filename):
    """
    Get the absolute path to a dataset file in the data/ directory.
    
    Parameters:
    -----------
    filename : str
        Name of the CSV file (e.g., 'crop_recommendation.csv')
    
    Returns:
    --------
    str
        Absolute path to the dataset file
    """
    # Get the project root directory (parent of scripts/)
    project_root = Path(__file__).parent.parent
    data_dir = project_root / 'data'
    file_path = data_dir / filename
    
    return file_path


def load_dataset(filename, display_info=True):
    """
    Load a CSV dataset into a Pandas DataFrame.
    
    Parameters:
    -----------
    filename : str
        Name of the CSV file to load
    display_info : bool, default=True
        If True, display dataset shape, columns, and head
    
    Returns:
    --------
    pd.DataFrame
        Loaded dataset as a DataFrame
    """
    file_path = get_data_path(filename)
    
    # Check if file exists
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")
    
    # Load the dataset
    print(f"\n{'='*60}")
    print(f"Loading dataset: {filename}")
    print(f"{'='*60}")
    
    df = pd.read_csv(file_path)
    
    if display_info:
        # Display basic information
        print(f"\nDataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        
        print(f"\nColumn Names ({len(df.columns)} columns):")
        print("-" * 60)
        for i, col in enumerate(df.columns, 1):
            print(f"  {i}. {col}")
        
        print(f"\nFirst 5 Rows:")
        print("-" * 60)
        print(df.head())
        
        print(f"\nData Types:")
        print("-" * 60)
        print(df.dtypes)
        
        print(f"\nBasic Statistics:")
        print("-" * 60)
        print(df.describe())
    
    return df


def load_all_datasets():
    """
    Load all datasets for the Smart Farming project.
    
    Returns:
    --------
    dict
        Dictionary containing all loaded datasets with descriptive keys
    """
    datasets = {}
    
    # Load Crop Recommendation Dataset
    try:
        datasets['crop_recommendation'] = load_dataset('crop_recommendation.csv')
    except FileNotFoundError as e:
        print(f"Warning: {e}")
    
    # Load Crop Production/Yield Dataset
    try:
        datasets['crop_production'] = load_dataset('crop_production.csv')
    except FileNotFoundError as e:
        print(f"Warning: {e}")
    
    return datasets


if __name__ == "__main__":
    """
    Main execution block.
    Loads all datasets and displays their information.
    """
    print("="*60)
    print("Smart Farming System - Data Loading Module")
    print("="*60)
    
    # Load all datasets
    data_dict = load_all_datasets()
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total datasets loaded: {len(data_dict)}")
    for name, df in data_dict.items():
        print(f"  - {name}: {df.shape[0]} rows × {df.shape[1]} columns")
    
    print(f"\n{'='*60}")
    print("Data loading completed successfully!")
    print(f"{'='*60}\n")


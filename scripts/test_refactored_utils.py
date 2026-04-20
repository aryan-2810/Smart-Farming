"""
Test script for refactored visualization and report generation functions.
This demonstrates crop-specific folder organization.
"""

import pandas as pd
from pathlib import Path
import sys

# Add parent directory to path to import utils
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import generate_yield_chart, generate_pdf


def create_sample_data():
    """Create sample DataFrame for testing."""
    import numpy as np
    
    np.random.seed(42)
    n_samples = 50
    
    # Generate sample data
    temperature = np.linspace(20, 35, n_samples)
    rainfall = np.linspace(100, 300, n_samples)
    
    # Create yield based on relationships (for demo purposes)
    yield_data = 2.5 + 0.1 * temperature + 0.005 * rainfall + np.random.normal(0, 0.5, n_samples)
    yield_data = np.maximum(yield_data, 1.0)  # Ensure positive yield
    
    df = pd.DataFrame({
        'temperature': temperature,
        'rainfall': rainfall,
        'yield': yield_data
    })
    
    return df


def main():
    print("="*70)
    print("Testing Refactored Visualization & Report Functions")
    print("="*70)
    
    # Create sample data
    print("\n[1] Creating sample data...")
    df = create_sample_data()
    print(f"[OK] Sample data created: {df.shape[0]} rows")
    
    # Test chart generation for different crops
    crops = ["rice", "wheat", "corn"]
    
    chart_paths = []
    for crop in crops:
        print(f"\n[2] Generating chart for {crop}...")
        chart_path = generate_yield_chart(df, crop_name=crop)
        if chart_path:
            print(f"[OK] Chart saved: {chart_path}")
            chart_paths.append(chart_path)
        else:
            print(f"[ERROR] Failed to generate chart for {crop}")
    
    # Test PDF generation
    print("\n[3] Generating PDF reports...")
    for i, crop in enumerate(crops):
        if i < len(chart_paths):
            weather_data = {
                "city": "Varanasi",
                "temperature": 28.5,
                "rainfall": 210.0,
                "humidity": 78.2
            }
            
            pdf_path = generate_pdf(
                crop_name=crop,
                temperature=weather_data["temperature"],
                rainfall=weather_data["rainfall"],
                humidity=weather_data["humidity"],
                predicted_yield=3.45 + i * 0.5,
                chart_path=chart_paths[i],
                filename_prefix="Crop_Report"
            )
            print(f"[OK] PDF saved for {crop}: {pdf_path}")
    
    print("\n" + "="*70)
    print("Test completed successfully!")
    print("Check outputs/charts/<crop>/ and outputs/reports/<crop>/ folders")
    print("="*70)


if __name__ == "__main__":
    main()


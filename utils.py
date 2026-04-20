"""
Utility functions for the Smart Farming ML project.

This module provides chart generation utilities used across the project.
"""

from pathlib import Path
from datetime import datetime
from typing import Optional

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def generate_yield_chart(
    df: pd.DataFrame,
    output_dir: str = "outputs/charts",
    filename_prefix: str = "yield_chart",
    crop_name: Optional[str] = None
) -> Optional[str]:
    """
    Create a side-by-side line chart for:
    1) Yield vs Temperature
    2) Yield vs Rainfall

    The chart is saved directly under `outputs/charts/` (no per-crop subfolders)
    with a timestamp-based filename (YYYY-MM-DD_HH-MM). If `crop_name` is provided,
    it will be included in the filename: `<prefix>_<crop>_<timestamp>.png`.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing at least columns: 'yield', 'temperature', 'rainfall'.
    output_dir : str, default "outputs/charts"
        Base directory where the chart will be saved.
    filename_prefix : str, default "yield_chart"
        Prefix for the saved filename.
    crop_name : Optional[str]
        Name of the crop used to create subfolder. If None, saves under base dir.

    Returns:
    --------
    Optional[str]
        The path to the saved chart image, or None if required columns are missing.
    """
    required_cols = {"yield", "temperature", "rainfall"}
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        return None

    data = df[["yield", "temperature", "rainfall"]].copy()
    data_temp = data.dropna(subset=["yield", "temperature"]).sort_values("temperature")
    data_rain = data.dropna(subset=["yield", "rainfall"]).sort_values("rainfall")

    base_dir = Path(output_dir)
    charts_dir = base_dir
    charts_dir.mkdir(parents=True, exist_ok=True)

    # Aesthetics
    sns.set_style("whitegrid")
    plt.rcParams.update({
        "figure.figsize": (14, 6),
        "axes.titlesize": 16,
        "axes.titleweight": "bold",
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11
    })

    fig, axes = plt.subplots(1, 2)

    # Natural colors
    color_green = "#2E8B57"
    color_blue = "#1E90FF"

    sns.lineplot(ax=axes[0], data=data_temp, x="temperature", y="yield", color=color_green, linewidth=2)
    axes[0].set_title("Yield vs Temperature")
    axes[0].set_xlabel("Temperature (°C)")
    axes[0].set_ylabel("Yield (tons/hectare)")

    # Annotate peak yield (temperature)
    if not data_temp.empty:
        idx_peak_t = data_temp["yield"].idxmax()
        x_peak_t = data_temp.loc[idx_peak_t, "temperature"]
        y_peak_t = data_temp.loc[idx_peak_t, "yield"]
        axes[0].scatter([x_peak_t], [y_peak_t], color=color_green, s=50, zorder=5)
        axes[0].annotate(
            "Peak yield",
            xy=(x_peak_t, y_peak_t),
            xytext=(x_peak_t, y_peak_t + (data_temp["yield"].max() * 0.05)),
            arrowprops=dict(arrowstyle="->", color=color_green),
            fontsize=10,
            color=color_green
        )

    sns.lineplot(ax=axes[1], data=data_rain, x="rainfall", y="yield", color=color_blue, linewidth=2)
    axes[1].set_title("Yield vs Rainfall")
    axes[1].set_xlabel("Rainfall (mm)")
    axes[1].set_ylabel("Yield (tons/hectare)")

    # Annotate peak yield (rainfall)
    if not data_rain.empty:
        idx_peak_r = data_rain["yield"].idxmax()
        x_peak_r = data_rain.loc[idx_peak_r, "rainfall"]
        y_peak_r = data_rain.loc[idx_peak_r, "yield"]
        axes[1].scatter([x_peak_r], [y_peak_r], color=color_blue, s=50, zorder=5)
        axes[1].annotate(
            "Peak yield",
            xy=(x_peak_r, y_peak_r),
            xytext=(x_peak_r, y_peak_r + (data_rain["yield"].max() * 0.05)),
            arrowprops=dict(arrowstyle="->", color=color_blue),
            fontsize=10,
            color=color_blue
        )

    plt.tight_layout()

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M")
    crop_suffix = f"_{str(crop_name).strip().lower()}" if crop_name else ""
    outfile = charts_dir / f"{filename_prefix}{crop_suffix}_{ts}.png"

    fig.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return str(outfile.resolve())


# ------------------------------------------------------------
# PDF report generation using fpdf
# ------------------------------------------------------------

def generate_pdf(
    crop_name: str,
    temperature: float,
    rainfall: float,
    humidity: float,
    predicted_yield: float,
    chart_path: Optional[str] = None,
    output_dir: str = "outputs/reports",
    filename_prefix: str = "Crop_Report"
) -> str:
    """
    Generate a simple PDF report summarizing key results and save under
    outputs/reports/ (no per-crop subfolders) with timestamp (YYYY-MM-DD_HH-MM).

    Returns absolute file path.
    """
    try:
        from fpdf import FPDF
    except ImportError as e:
        raise ImportError("fpdf is required for PDF generation. Please install with `pip install fpdf`.") from e

    base_dir = Path(output_dir)
    reports_dir = base_dir
    reports_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M")
    crop_suffix = f"_{str(crop_name).strip().lower()}" if crop_name else ""
    pdf_path = reports_dir / f"{filename_prefix}{crop_suffix}_{ts}.pdf"

    pdf = FPDF(orientation='P', unit='mm', format='A4')
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    TITLE_COLOR = (27, 94, 32)
    HEADER_FILL = (235, 255, 235)  # Light green per request
    TEXT_COLOR = (0, 0, 0)

    # Header
    pdf.set_fill_color(*HEADER_FILL)
    pdf.rect(x=10, y=10, w=190, h=14, style='F')
    pdf.set_xy(10, 10)
    pdf.set_text_color(*TITLE_COLOR)
    pdf.set_font('Arial', 'B', 16)
    # Avoid emojis due to FPDF core font encoding limits
    pdf.cell(190, 10, "Smart Farming Report", ln=True, align='C')

    pdf.ln(4)
    pdf.set_text_color(*TEXT_COLOR)
    pdf.set_font('Arial', '', 12)
    pdf.cell(0, 8, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)

    # Recommendation Section (simulated icon label)
    pdf.ln(2)
    pdf.set_text_color(*TITLE_COLOR)
    pdf.set_font('Arial', 'B', 13)
    pdf.cell(0, 8, "[Crop] Recommendation", ln=True)

    pdf.set_text_color(*TEXT_COLOR)
    pdf.set_font('Arial', '', 12)
    # Table-like alignment
    left_w, right_w = 60, 120
    pdf.cell(left_w, 8, "Recommended Crop:", border=0)
    pdf.cell(right_w, 8, f"{crop_name}", ln=True, border=0)
    pdf.cell(left_w, 8, "Predicted Yield:", border=0)
    pdf.cell(right_w, 8, f"{float(predicted_yield):.2f} tons/hectare", ln=True, border=0)

    # Weather Section
    pdf.ln(2)
    pdf.set_text_color(*TITLE_COLOR)
    pdf.set_font('Arial', 'B', 13)
    pdf.cell(0, 8, "[Weather] Summary", ln=True)

    pdf.set_text_color(*TEXT_COLOR)
    pdf.set_font('Arial', '', 12)
    pdf.cell(left_w, 8, "Temperature:", border=0)
    pdf.cell(right_w, 8, f"{float(temperature):.2f} °C", ln=True, border=0)
    pdf.cell(left_w, 8, "Rainfall:", border=0)
    pdf.cell(right_w, 8, f"{float(rainfall):.2f} mm", ln=True, border=0)
    pdf.cell(left_w, 8, "Humidity:", border=0)
    pdf.cell(right_w, 8, f"{float(humidity):.2f} %", ln=True, border=0)

    # Chart Section
    pdf.ln(2)
    pdf.set_text_color(*TITLE_COLOR)
    pdf.set_font('Arial', 'B', 13)
    pdf.cell(0, 8, "[Chart] Yield vs Weather", ln=True)

    pdf.set_text_color(*TEXT_COLOR)
    pdf.set_font('Arial', '', 12)
    if chart_path and Path(chart_path).exists():
        pdf.image(chart_path, x=15, y=None, w=180)
        pdf.ln(4)
    else:
        pdf.cell(0, 8, "Chart not available.", ln=True)

    # Notes Section
    pdf.ln(2)
    pdf.set_text_color(*TITLE_COLOR)
    pdf.set_font('Arial', 'B', 13)
    pdf.cell(0, 8, "[Notes] Remarks", ln=True)

    pdf.set_text_color(*TEXT_COLOR)
    pdf.set_font('Arial', '', 12)
    pdf.multi_cell(0, 6, "This report provides an AI-assisted recommendation based on current inputs. "
                        "Please consider local agronomic practices, soil tests, and expert advice before sowing.")

    # Footer
    pdf.ln(4)
    pdf.set_font('Arial', 'I', 10)
    pdf.cell(0, 8, "Generated by Smart Farming ML System", ln=True, align='C')

    pdf.output(str(pdf_path))
    return str(pdf_path.resolve())



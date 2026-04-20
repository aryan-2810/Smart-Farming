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

    Saves to outputs/charts/yield_chart_[crop]_YYYY-MM-DD_HH-MM.png (no subfolders)
    and returns the absolute path. Uses Seaborn whitegrid style with
    enhanced readability and peak yield annotations.
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

    color_green = "#2E8B57"
    color_blue = "#1E90FF"

    sns.lineplot(ax=axes[0], data=data_temp, x="temperature", y="yield", color=color_green, linewidth=2)
    axes[0].set_title("Yield vs Temperature")
    axes[0].set_xlabel("Temperature (°C)")
    axes[0].set_ylabel("Yield (tons/hectare)")

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


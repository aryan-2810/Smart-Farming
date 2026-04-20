import argparse
from pathlib import Path
import sys

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def main():
	parser = argparse.ArgumentParser(description="Utilities for chart/report generation")
	subparsers = parser.add_subparsers(dest="command", required=True)

	# Chart subcommand
	p_chart = subparsers.add_parser("chart", help="Generate yield chart from CSV or sample data")
	p_chart.add_argument("--csv", help="Path to CSV with columns: yield, temperature, rainfall")
	p_chart.add_argument("--crop", required=True, help="Crop name for subfolder")
	p_chart.add_argument("--output_dir", default="outputs/charts", help="Charts output base directory")

	# PDF subcommand
	p_pdf = subparsers.add_parser("pdf", help="Generate PDF report")
	p_pdf.add_argument("--crop", required=True, help="Predicted crop name")
	p_pdf.add_argument("--yield_tph", required=True, type=float, help="Predicted yield (tons/hectare)")
	p_pdf.add_argument("--city", required=True, help="City name")
	p_pdf.add_argument("--temperature", required=True, type=float, help="Temperature in °C")
	p_pdf.add_argument("--rainfall", required=True, type=float, help="Rainfall in mm")
	p_pdf.add_argument("--humidity", required=True, type=float, help="Humidity in %")
	p_pdf.add_argument("--chart", default="", help="Path to chart image (optional)")
	p_pdf.add_argument("--output_dir", default="outputs/reports", help="Reports output base directory")

	args = parser.parse_args()

	if args.command == "chart":
		try:
			from scripts.visuals.chart_generator import generate_yield_chart
		except Exception as e:
			print(f"Import error: {e}")
			sys.exit(1)

		import pandas as pd
		if args.csv:
			csv_path = Path(args.csv)
			if not csv_path.exists():
				print(f"CSV not found: {csv_path}")
				sys.exit(1)
			df = pd.read_csv(csv_path)
		else:
			# Create sample data if CSV not provided
			import numpy as np
			np.random.seed(42)
			n_samples = 50
			temperature = np.linspace(20, 35, n_samples)
			rainfall = np.linspace(100, 300, n_samples)
			yield_data = 2.5 + 0.1 * temperature + 0.005 * rainfall + np.random.normal(0, 0.5, n_samples)
			yield_data = np.maximum(yield_data, 1.0)
			df = pd.DataFrame({
				"temperature": temperature,
				"rainfall": rainfall,
				"yield": yield_data,
			})

		chart_path = generate_yield_chart(df, output_dir=args.output_dir, crop_name=args.crop)
		print(chart_path if chart_path else "Chart generation failed")
		return

	if args.command == "pdf":
		try:
			from scripts.reports.report_generator import generate_pdf
		except Exception as e:
			print(f"Import error: {e}")
			import traceback
			traceback.print_exc()
			sys.exit(1)

		weather = {
			"city": args.city,
			"temperature": args.temperature,
			"rainfall": args.rainfall,
			"humidity": args.humidity,
		}
		chart_path = args.chart if args.chart else None
		pdf_path = generate_pdf(
			predicted_crop=args.crop,
			predicted_yield=args.yield_tph,
			weather_data=weather,
			chart_path=chart_path,
			output_dir=args.output_dir,
		)
		print(pdf_path)
		return


if __name__ == "__main__":
	main()

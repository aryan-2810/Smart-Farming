import argparse
from pathlib import Path
import sys
import pandas as pd

try:
	from utils import generate_yield_chart
except Exception as e:
	print(f"Error importing utils.generate_yield_chart: {e}")
	sys.exit(1)


def main():
	parser = argparse.ArgumentParser(description="Generate Yield vs Weather charts from a CSV.")
	parser.add_argument("csv", help="Path to CSV file containing at least 'yield', 'temperature', 'rainfall' columns")
	parser.add_argument("--output_dir", default="outputs/charts", help="Directory to save chart")
	parser.add_argument("--prefix", default="yield_vs_weather", help="Filename prefix")
	args = parser.parse_args()

	csv_path = Path(args.csv)
	if not csv_path.exists():
		print(f"CSV not found: {csv_path}")
		sys.exit(1)

	df = pd.read_csv(csv_path)
	chart_path = generate_yield_chart(df, output_dir=args.output_dir, filename_prefix=args.prefix)
	if chart_path is None:
		print("Required columns missing. Ensure CSV has: 'yield', 'temperature', 'rainfall'.")
		sys.exit(2)
	print(f"Chart saved to: {chart_path}")


if __name__ == "__main__":
	main()

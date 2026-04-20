import argparse
from pathlib import Path
import sys

try:
	# Import sibling module directly when executed from project root
	from visuals_and_report import generate_pdf
except Exception as e:
	print(f"Error importing generate_pdf: {e}")
	sys.exit(1)


def main():
	parser = argparse.ArgumentParser(description="Generate Smart Farming PDF report")
	parser.add_argument("--crop", required=True, help="Predicted crop name")
	parser.add_argument("--yield_tph", required=True, type=float, help="Predicted yield (tons/hectare)")
	parser.add_argument("--city", required=True, help="City name")
	parser.add_argument("--temperature", required=True, type=float, help="Temperature in °C")
	parser.add_argument("--rainfall", required=True, type=float, help="Rainfall in mm")
	parser.add_argument("--humidity", required=True, type=float, help="Humidity in %")
	parser.add_argument("--chart", default="", help="Path to chart image (optional)")
	parser.add_argument("--output_dir", default="outputs/reports", help="Reports output directory")
	args = parser.parse_args()

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


if __name__ == "__main__":
	main()

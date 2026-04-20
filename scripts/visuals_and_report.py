"""
Backwards-compatibility wrappers after splitting into separate modules.
Prefer importing from:
  - scripts.visuals.chart_generator import generate_yield_chart
  - scripts.reports.report_generator import generate_pdf
"""

from scripts.visuals.chart_generator import generate_yield_chart  # noqa: F401
from scripts.reports.report_generator import generate_pdf  # noqa: F401

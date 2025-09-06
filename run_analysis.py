"""
run_analysis.py (template)

HOW TO USE AFTER YOU UPLOAD YOUR DATA:

1) Place your CSV or JSON under /mnt/data, e.g. /mnt/data/trades.csv
2) Edit INPUT_PATH below.
3) (Optional) Provide a column_map if your broker uses different names.
4) Run this script. It will produce:
   - /mnt/data/analysis/equity_curve.png
   - /mnt/data/analysis/drawdown.png
   - /mnt/data/analysis/summary.json
   - /mnt/data/analysis/summary.md
"""

from pathlib import Path
import argparse
import pandas as pd
from core.trade_analysis import load_trades, compute_metrics, aggregate, plot_equity_curve, plot_drawdown, write_summary


# Default input (can be overridden with --input)
DEFAULT_INPUT = Path('mnt') / 'data' / 'trades.csv'

# Example column map (uncomment & edit)
# column_map = {
#     "Trade Date": "date",
#     "Trade Time": "time",
#     "Realized P/L": "pnl",
#     "Symbol": "symbol",
# }
# Provide a safe default mapping for common broker CSVs in mnt/data
column_map = {
   "Date": "date",
   "Amount": "pnl",
}


def find_fallback_input(data_dir: Path) -> Path | None:
   # prefer CSVs, then JSONs
   for ext in ("*.csv", "*.json"):
      files = sorted(data_dir.glob(ext))
      if files:
         return files[0]
   return None


def main():
   parser = argparse.ArgumentParser(description='Run trade analysis')
   parser.add_argument('--input', '-i', help='Path to trades CSV/JSON', default=None)
   parser.add_argument('--column-map', help='JSON string mapping original column names to expected names, e.g. "{\"Date\":\"date\",\"Amount\":\"pnl\"}"', default=None)
   args = parser.parse_args()

   if args.input:
      input_path = Path(args.input)
   else:
      input_path = DEFAULT_INPUT

   if not input_path.exists():
      fallback = find_fallback_input(Path('mnt') / 'data')
      if fallback:
         print(f"Input file {input_path} not found, falling back to {fallback}")
         input_path = fallback
      else:
         raise FileNotFoundError(f"File not found: {input_path}")

   # Parse column map if provided on the CLI
   import json as _json
   if args.column_map:
      try:
         cli_map = _json.loads(args.column_map)
         if isinstance(cli_map, dict):
            # overlay CLI-provided map on top of default
            merged = dict(column_map) if column_map else {}
            merged.update(cli_map)
            use_column_map = merged
         else:
            use_column_map = column_map
      except Exception as e:
         print('Failed to parse --column-map JSON:', e)
         use_column_map = column_map
   else:
      use_column_map = column_map

   print(f"Using input: {input_path}")

   df = load_trades(input_path, column_map=use_column_map)

   # Diagnostic information to help debug compute_metrics failures
   print('DEBUG: df.shape=', getattr(df, 'shape', None))
   print('DEBUG: df.columns=', list(getattr(df, 'columns', [])))
   try:
      print('DEBUG: df.head()=\n', df.head().to_string())
   except Exception as e:
      print('DEBUG: failed to print head:', e)
   if 'pnl' in df.columns:
      try:
         print('DEBUG: pnl stats -> count:', df['pnl'].count(), 'sum:', df['pnl'].sum())
      except Exception as e:
         print('DEBUG: pnl column present but computing stats failed:', e)

   metrics = compute_metrics(df)
   aggs = aggregate(df)

   out_dir = Path('mnt') / 'data' / 'analysis'
   out_dir.mkdir(parents=True, exist_ok=True)

   ec_path = plot_equity_curve(df, out_dir / 'equity_curve.png')
   dd_path = plot_drawdown(df, out_dir / 'drawdown.png')
   json_path, md_path = write_summary(df, metrics, out_dir)

   print('Saved:')
   print(ec_path)
   print(dd_path)
   print(json_path)
   print(md_path)

   # Preview top lines
   print(df.head().to_string())
   print('Metrics:', metrics)


if __name__ == '__main__':
   main()

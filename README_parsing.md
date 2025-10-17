Parsing Plan and First-Pass Implementation

What this does
- Reads locally available datasets from a data/ folder (or a custom path you pass to the script).
- Unifies country names (e.g., US = USA = United States) and drops region/aggregate groupings.
- Handles mixed formats (csv/xls/xlsx/dta). Special-case military strength from sheet "Constant (2023) US$" (row 6 year header, column 0 countries).
- Builds a Country–Year panel with one column per metric. Missing metric values are set to -1 (exclusion sentinel).
- For each metric and year, applies min–max normalization among valid values (ignores -1): yearly min -> 0, yearly max -> 1, linear in between; if no variation, assigns 0.5. GlobalDebt is inverted (lower debt -> higher score).
- Computes relative standing at each (Country, Year) by averaging the available normalized metrics only (i.e., skipping negative/missing values).
- Ranks Top 5 countries by area under the composite curve since 1900 and plots them.

How to run
- Put your data files into data/ (or pass a custom folder).
- Run: `python scripts/build_empire_standings.py` (or `python scripts/build_empire_standings.py <path-to-data>`)
- Outputs are written to results/:
  - countries_list.csv — list of countries discovered after standardization
  - parsed_schema_summary.csv — metric coverage summary
  - empire_composite.csv — composite and per-metric normalized values
  - country_area_ranking.csv — area (sum of composite) since 1900, descending
  - empire_standings_top5.png — line chart of Top 5 curves (>= 1900)
  - parsing_warnings.csv — notes about files that were skipped/problematic

Notes and assumptions
- ReserveCurrency.csv is excluded (world/currency aggregates, not country-level).
- ReserveCurrency.csv is now included via a derived metric ReservePower:
  - Parse COFER currency shares and compute per-year normalized shares relative to the leading currency (leader=1.0; others=share/leader).
  - Map currencies to issuing countries (USD→United States, GBP→United Kingdom, JPY→Japan, CHF→Switzerland, CAD→Canada, AUD→Australia, CNY→China).
  - Distribute EUR to core Eurozone members (Germany, France, Italy, Spain, Netherlands) using GDP weights for that year (fallback: equal split).
- Military strength: parsed from sheet "Constant (2023) US$"; if the row/column layout changes, the script will need an updated hint.
- Data with no year variation in a given metric-year maps to 0.5 for all countries for that metric-year.
- The composite averages only available normalized metrics (non-negative raw), per your specification that -1 is the explicit missing/exclude mark.
- If additional metrics need direction inversion (lower-is-better), we can extend the script to invert those before normalization.

# World Order Metrics — Formulas and Pipeline

## Data Loading -> located in data/
- GMD.csv: [rGDP_USD, USDFX, cGovDebtGDP, exports_USD, imports_USD]
- education.csv: education (all columns)
- military.csv: [CINC]
- polity.csv: [xconst, parcomp]
- CHAT.csv: [all columns]

Note: Country names are canonicalized with common ISO3/name aliases (e.g., US/USA/United States → USA; CHN → CHINA).

## Normalization
- norm(series):
  - Per year across countries, compute: (series − min) / (max − min)
  - If max == min or only one value exists: return 0.5 for non-null entries
- "share of" = value / sum of df

## Data supplementation - exempt CHAT.csv
- Skip cells that have < 4 countries to normalize against.
  - keep count of countries that have seen data
  - Once we find the 4th country, use linear interpolation normalize(start here)
  - continue interpolating for missing data points


## Metrics
- Education = norm(education)
- Military = norm(share of CINC)
  - Share computed per year: CINC_country / sum(CINC)_year
- EconomicIndex = norm(share of rGDP_USD)
  - rGDP_USD_country / sum(rGDP_USD)_year, then robust norm
- TradeShare = average of norm(share of exports_USD) and norm(share of imports_USD)
  - share computed per year, each component min–max normalized per year, then averaged
- ReserveCurrency = 1 − norm(USDfx)
- FinancialCenter = 1 − norm(cgovdebt_GDP)
- Innovation = average of all per-year normalized CHAT columns available for the country-year
- Competitiveness = avg(norm(xconst) and norm(parcomp)) # need both for valid metric


Metrics.csv will include selected countries with columns [metrics after filling]
selected countries = [CHINA, USA, FRANCE, GERMANY, UNITED KINGDOM, RUSSIA]

- Note: forward fill last available metric point.
## Composite Index (weighted)
- WorldOrderIndex[c,y] =
  0.15*Education + 0.15*Competitiveness + 0.15*Technology +
  0.15*EconomicOutput + 0.10*TradeShare + 0.10*Military +
  0.10*FinancialCenter + 0.10*ReserveCurrency

Note: if any component is missing for a country-year, weights of available components are renormalized to sum to 1 for that row.

## Plotting
- Composite graph shows weighted WorldOrderIndex (smoothed), with key countries emphasized and WW1/WW2 shaded
  - 4 metric threshold to start line
- Raw metrics per country are plotted on a dynamic grid with per-series smoothing.
- All plots start at year ≥1800 and end at ≤2024.

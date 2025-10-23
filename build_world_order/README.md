# World Order Metrics — Formulas and Pipeline

## Data Loading -> data/

### ID
- build country_id.csv: [name, abv, code, COW]
  - country_names.csv: [name, alpha-3 = abv ,country-code = code]
  - cow2iso.csv: [cow_id = COW, iso_id = code]
- Note: when reading id's, convert to int and trim leading zeros

### Measures
- general schema: [column = reference to country_id : value]
- GMD.csv: [ISO3 = ABV: rGDP_USD, USDfx, cGovDebtGDP, exports_USD, imports_USD, M0, finv_GDP, CA_USD]
- Education.csv: (all collumns) [CCODE = CODE: year]
- military.csv: [CCODE = COW: CINC]
- polity.csv: [CCODE = COW: xconst, parcomp]
- CHAT.csv: [iso3 = ABV: all columns]

## Clean data
- Build `data/clean_data.csv` with the above data
- For each datapoint listed above excluding chat.csv
  - add as a column in clean_data.csv
    - [country_name, ccode, iso3, 
      rGDP_USD, USDfx, cGovDebtGDP,
      exports_USD, imports_USD, M0, 
      finv_GDP, CA_USD, education, CINC,
      xconst, parcomp]
  - each row is a country-year
    - each country only needs a row for years it has atleast 1 data point
  - pass and linear interpolate between first and last point
- Interpolate chat.csv file
- Use `clean_data.csv` and `CHAT.csv` for the remaining steps.
- no need to clean every run

## Metrics
- Metrics.csv will include all countries & their calculate metrics (for training)
  - [EDU, MIL, ECON,TRAD, RESV, FIN, INV, CMPT, INDEX]
- Do not include country-years that have no data (for sparsity)
- Metrics that require multiple sources must have all available to be non-null
- Forward-fill last available metric to 2024, max ten years.
- Linear interpolate metrics.csv (some may have gaps due to multivariable calculations).

### Metric Normalization
- norm(val, series, year):
  - compute against all countries for a given year
  - if len(series(year)) < 4: return null, need atleast 4 to normalize against
  - return (val − series_min(year)) / (series_max(year) − series_min(year))

### Metric Calculation
  - Education = norm(education)
  - Military = norm(CINC)
  - EconomicIndex = norm(rGDP_USD)
  - TradeShare = avg((exports_USD),  norm(imports_USD))
  - ReserveCurrency = norm(CA_USD)
  - FinancialCenter = 0.5*norm(M0)+ 0.3*norm(finv_GDP) + 0.2*( 1 − norm(cgovdebt_GDP))
  - Innovation = avg([norm(col) for col in chat.csv])
  - Competitiveness = avg(norm(xconst), norm(parcomp))
#### Composite Index
- WorldOrderIndex[c,y] =
  0.15*Education + 0.15*Competitiveness + 0.15*Technology +
  0.15*EconomicOutput + 0.10*TradeShare + 0.10*Military +
  0.10*FinancialCenter + 0.10*ReserveCurrency
- If a component is missing, weights of available components are renormalized to sum to 1

## Plotting - results/
- selected countries = [CHINA, USA, FRANCE, GERMANY, UNITED KINGDOM, RUSSIA]
- All plots are smoothed and are dated [1800, 2024]
- Composite graph shows "Relative Standing of Great Empires"
  - At least 4 component metrics must be present to start the line
  - WW1/WW2 shaded
  - USA / CHINA / UK are highlighted, the rest are dotted and faint
  - (y: min(composite)..max(composite))
- Raw metrics for the top 25 countries with most data are plotted on a 4x2 grid (y: -0.1..1.1)
- In another top25/ folder, they are mapped on 25 individual plots. 


## Run
- Build cleaned data (one-time): `python -m build_world_order.clean_data`
- Run pipeline: `python -m build_world_order.run_pipeline `


World Order Metrics — Formulas and Pipeline

Normalization
- norm(series):
  - Per year across countries, compute: (series − min) / (max − min)
  - If max == min or only one value exists: return 0.5 for non-null entries

Data handling
- Forward-fill inputs per country over time (year-1, year-2, …) for core datasets (Education, Military, GMD).
- CHAT and polity are ingested with year cleaning; CHAT features are forward-filled per country; polity is not forward-filled.
- Country names are canonicalized with common ISO3/name aliases (e.g., US/USA/United States → USA; CHN → CHINA).

Metrics
- Education = norm(education)

- Military = norm(share of CINC)
  - Share computed per year: CINC_country / sum(CINC)_year

- EconomicOutput (EconomicIndex) = norm(share of rGDP_USD)
  - rGDP_USD_country / sum(rGDP_USD)_year, then robust norm

- TradeShare = average of norm(share of exports_USD) and norm(share of imports_USD)
  - share computed per year, each component min–max normalized per year, then averaged

- ReserveCurrency = 1 − norm(USDfx)

- FinancialCenter = 1 − norm(cgovdebt_GDP)

- Innovation = average of all per-year normalized CHAT columns available for the country-year

- Competitiveness = average of all normalized, non-null metrics available for the country-year
  - Uses: Education, Military, EconomicIndex, TradeShare, ReserveCurrency, FinancialCenter, Innovation

Composite Index (weighted)
- WorldOrderIndex[c,y] =
  0.15*Education + 0.15*Competitiveness + 0.15*Technology +
  0.15*EconomicOutput + 0.10*TradeShare + 0.10*Military +
  0.10*FinancialCenter + 0.10*ReserveCurrency

- Notes:
  - Technology is an alias for Innovation; EconomicOutput is an alias for EconomicIndex.
  - If any component is missing for a country-year, weights of available components are renormalized to sum to 1 for that row.

Plotting
- Composite graph shows weighted WorldOrderIndex (smoothed), with key countries emphasized and WW1/WW2 shaded.
- Raw metrics per country are plotted on a dynamic grid with per-series smoothing.
- All plots start at year ≥1800 and end at ≤2024.

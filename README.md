# Deep Leaning to predict the future

This project utilizes public datasets and machine learning libraries to project socio-economic trends ten years in the future. The estimated accuracy is n%.


### Research question: Are Empires and Companies one in the same?

I split my data in two: empires and companies. From here, I trained three models:

1) World Order Forecast (WOF)
2) Market Share Forecast (MSF)
3) MSF Diluted from WOF

To develop an accuracy metric, I left one country and one industry out, and averaged a walk forward loss function. I must admit this metric is biased because I cherrypick a stable industry/country, and have already trained the model on wordly trends from that time period (it already understands historical shifts and events).
## Background

I was first introduced to this idea reading Ray Dalio's compelling piece, [Principles for dealing with the changing world order](https://www.economicprinciples.org/DalioChangingWorldOrderCharts.pdf). 

His team assembled data that dates back nearly 1000 years from hundred of cross referenced sources. My first instinct was to reproduce his graphs, but much of his data was privatized and internal. Therefore, the primary limitation with my project-as with any deep learning pursuit-is data. I could only pull from a few publically available sources within the scope of my resources.

## Data

### Public Data for Empires (Countries)
| Metric | Source | Range | Description |
|--------|--------|-------|-------------|
| Global Debt | World Bank WDI | 2000-2023 | Government debt as % of GDP |
| Military Strength | SIPRI | 2000-2023 | Military expenditure as % of GDP |
| GDP & Trade | World Bank WDI | 2000-2023 | GDP per capita and trade as % of GDP |
| Reserve Currency | IMF COFER | 2000-2023 | Currency share in global foreign exchange reserves |
| Education | World Bank WDI | 2000-2023 | Average years of schooling (population 25+) |
| Innovation | WIPO | 2000-2023 | Patent applications per million population |
| Competitiveness | WEF | 2000-2023 | Global Competitiveness Index (0-100 scale) |
| Financial Centers | GFCI | 2000-2023 | Financial center development index |

### Public Data for Companies (Industries)
| Metric | Source | Range | Description |
|--------|--------|-------|-------------|
| Market Capitalization | Yahoo Finance/Bloomberg | 2000-2023 | Total market value by industry sector |
| R&D Spending | OECD STAN | 2000-2023 | Research & development expenditure by industry |
| Revenue Growth | SEC Filings/Annual Reports | 2000-2023 | Year-over-year revenue growth by sector |
| Employment Share | Bureau of Labor Statistics | 2000-2023 | Employment as % of total workforce |
| Patent Filings | USPTO/WIPO | 2000-2023 | Patent applications by industry classification |
| Global Trade Share | UN Comtrade | 2000-2023 | Export/import volumes by industry |
| Energy Consumption | IEA | 2000-2023 | Energy usage by industrial sector |
| Productivity Index | OECD | 2000-2023 | Labor productivity by industry |

## Data Transparency

A critical limitation of this analysis is the **data privatization problem**. Many countries, particularly those with authoritarian governance structures have FALSE OR RESTRICTED data; [CHN, RUS, IRN, PRK, VEN, MMR, SAU, ARE, QAT, BRN]

**Initial Focus: Democratic Countries with Data Transparency**

To ensure model reliability, the initial implementation should focus on countries with a relatively high freedom score; [ USA, GBR, DEU, FRA, CAN, AUS, NLD, CHE, DNK, SWE, JPN, KOR, ITA, ESP, BEL, AUT, NOR, FIN, NZL, IRL]

## Validation Strategy & Exclusions

For model validation using the leave-one-out approach, we need to avoid **biased candidates** that are either too stable (like Switzerland/Utilities) or too volatile. Instead, we should select entities with **average volatility and predictability** for unbiased validation.

**Analysis Approach**: Run `python analyze_loco_candidates.py` to identify optimal candidates based on:
- **Volatility Metrics**: Coefficient of variation, year-over-year changes
- **Predictability Scores**: Trend consistency, autocorrelation, range stability  
- **Target Range**: Predictability score between 0.3-0.7 (avoiding extremes)

**Previous Recommendations (TOO BIASED)**:
- ~~Switzerland (CHE)~~ - Too stable, highly predictable
- ~~Utilities Sector~~ - Too stable, highly predictable

**New Approach**: Select countries and industries that are:
- **Moderately volatile** (not too stable, not too erratic)
- **Average predictability** (some trend but not perfectly predictable)
- **Representative** of typical economic behavior
- **Sufficient data** (15+ years of observations)

## Implementation Status

### ✅ Completed Features

- **Dual Data Framework**: Separate pipelines for Empire (country) and Company (industry) data
- **Democratic Data Focus**: Initial implementation prioritizes transparent, democratic countries
- **Comprehensive Data Schema**: 8 country factors + 4 industry factors with proper validation
- **Synthetic Data Generation**: Development-ready synthetic data for all sources
- **LOCO Analysis Tool**: Automated identification of optimal validation candidates
- **Data Quality Framework**: Missing data handling, masking, and quality checks

### 🚧 Next Steps

1. **Real Data Integration**: Replace synthetic data with actual API calls to data sources
2. **Model Training**: Implement the three forecasting models (WOF, MSF, MSF Diluted)
3. **Uncertainty Quantification**: Add confidence intervals for predictions
4. **Authoritarian Data Handling**: Develop techniques for data-scarce/unreliable environments

### 🎯 Usage

```bash
# Analyze LOCO candidates to find optimal validation entities
python analyze_loco_candidates.py

# Run the dual analysis demo
python demo_dual_analysis.py

# Generated files:
# - results/country_volatility_analysis.csv
# - results/industry_volatility_analysis.csv
# - results/country_loco_candidates.csv
# - results/industry_loco_candidates.csv
# - results/loco_candidate_analysis.png
# - results/empire_standing_scores.csv
# - results/company_dominance_scores.csv
```

🌐 **[Live Demo & Documentation](https://mackthompson16.github.io/World-Order-Forecast)**


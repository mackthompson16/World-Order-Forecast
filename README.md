# Deep Learning to predict the future

This project utilizes public datasets and machine learning libraries to project socio-economic trends ten years in the future. The estimated accuracy is n%.


### Research question: Are Empires and Companies one in the same?

I split my data in two: empires and companies. From here, I trained three models:

1) World Order Forecast (WOF)
2) Market Share Forecast (MSF)
3) MSF Diluted from WOF

To develop an accuracy metric, I leave one country/industry out of training, and average a walk forward loss function. Because I could cherrypick the left out candidate with the most predictable history (to over estimate my model's accuracy), I also determine a candidate with an **average predictability** score for unbiased validation. Though, it can still be said that because global events are captured by all training data, the accuracy metric will still be overestimated.
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

## Corruption & Geography Constants

### Corruption Constant: Data Trustworthiness

The corruption constant models the **reliability and trustworthiness** of economic data based on institutional corruption levels. This addresses the critical issue that countries with high corruption often manipulate or restrict access to economic statistics.

#### Calculation Method

**Corruption Levels** (based on Transparency International CPI):
- **Very Low** (0.95): Denmark, Finland, New Zealand, Norway, Sweden, Switzerland, Singapore, Netherlands
- **Low** (0.85): Australia, Canada, Germany, UK, Austria, Belgium, Ireland, Japan, Estonia, Iceland  
- **Moderate** (0.70): USA, France, Italy, Spain, South Korea, Portugal, Poland, Czech Republic
- **High** (0.50): India, Brazil, China, Mexico, Turkey, Russia, Indonesia, Thailand, Saudi Arabia
- **Very High** (0.25): Venezuela, Myanmar, North Korea, Iran, Afghanistan, Syria, Yemen

**Effect on Predictions**:
```
Final Score = Base Composite Score × Corruption Trust Score
```

**Confidence Impact**:
- **High corruption** → Lower confidence in predictions due to unreliable underlying data
- **Low corruption** → Higher confidence due to transparent, verifiable data
- **Data manipulation detection** → Model can identify countries with suspiciously smooth or unrealistic patterns

#### Real-World Example

| Country | Corruption Level | Trust Score | Impact on Score |
|---------|------------------|-------------|-----------------|
| Denmark | Very Low | 0.95 | Minimal penalty (-5%) |
| USA | Moderate | 0.70 | Moderate penalty (-30%) |
| China | High | 0.50 | Significant penalty (-50%) |
| Venezuela | Very High | 0.25 | Severe penalty (-75%) |

### Geography Constant: Optimistic Curves

The geography constant creates **optimistic curves** for countries with natural geographic advantages, reflecting how location, resources, and terrain influence long-term economic potential.

#### Calculation Method

**Geography Advantages** (multiplicative effects):
- **Island Advantage** (1.15x): Natural defense, trade advantages (UK, Japan, Australia, New Zealand)
- **Coastal Access** (1.10x): Maritime trade, resource access (USA, France, Italy, Spain)
- **Strategic Location** (1.05x): Geographic strategic importance (Germany, Turkey, Egypt)
- **Resource Rich** (1.08x): Natural resource abundance (USA, Russia, Saudi Arabia, Brazil)
- **Landlocked** (0.90x): Limited trade routes, dependency (Switzerland, Austria, Czech Republic)
- **Arctic Challenges** (0.85x): Harsh climate, infrastructure costs (Norway, Sweden, Finland, Canada)
- **Desert Limitations** (0.88x): Water scarcity, agricultural challenges (Saudi Arabia, UAE, Egypt)
- **Mountain Barriers** (0.92x): Transportation difficulties (Switzerland, Austria, Afghanistan)

**Combined Geography Score**:
```
Geography Multiplier = ∏(Individual Geography Effects)
Final Multiplier = min(1.5, max(0.5, Geography Multiplier))
```

#### Real-World Examples

| Country | Geography Advantages | Calculation | Final Multiplier |
|---------|---------------------|-------------|------------------|
| USA | Coastal + Resource + Strategic | 1.10 × 1.08 × 1.05 | 1.25x |
| UK | Island + Coastal + Strategic | 1.15 × 1.10 × 1.05 | 1.33x |
| Switzerland | Mountain + Strategic | 0.92 × 1.05 | 0.97x |
| Saudi Arabia | Desert + Resource | 0.88 × 1.08 | 0.95x |

### Combined Effects on Predictions

#### Final Score Calculation

```
Final Standing Score = Base Composite Score × Corruption Trust Score × Geography Multiplier
```

#### Prediction Confidence Levels

**High Confidence** (Corruption < 0.7, Geography > 1.0):
- Transparent data + geographic advantages
- Examples: Denmark, Norway, Singapore

**Medium Confidence** (Corruption 0.5-0.7, Geography 0.9-1.1):
- Moderate data reliability + neutral geography  
- Examples: USA, France, Italy

**Low Confidence** (Corruption > 0.7 OR Geography < 0.9):
- Unreliable data OR geographic disadvantages
- Examples: China, Russia, Venezuela

#### Model Training Implications

**Democratic Countries First**: Train initial models on high-confidence countries (low corruption, good geography)
**Gradual Expansion**: Gradually incorporate lower-confidence countries with uncertainty quantification
**Bias Correction**: Use corruption/geography constants to adjust for systematic data quality differences

#### Validation Strategy Enhancement

**LOCO Selection**: Consider corruption and geography when selecting leave-one-out candidates
- Avoid very stable countries (Switzerland) - too predictable
- Avoid very unstable countries (Venezuela) - too unreliable  
- Select countries with **moderate corruption** (0.5-0.7) and **average geography** (0.9-1.1)

**Confidence Intervals**: Wider prediction intervals for high-corruption countries
**Sensitivity Analysis**: Test model robustness across different corruption/geography scenarios

## Implementation Status

### ✅ Completed Features

- **Dual Data Framework**: Separate pipelines for Empire (country) and Company (industry) data
- **Democratic Data Focus**: Initial implementation prioritizes transparent, democratic countries
- **Comprehensive Data Schema**: 10 country factors + 4 industry factors with proper validation
- **Corruption Constants**: Data trustworthiness modeling based on corruption levels
- **Geography Constants**: Optimistic curves based on geographic advantages/disadvantages
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
# Analyze corruption and geography constant effects
python demo_corruption_geography.py

# Analyze LOCO candidates to find optimal validation entities
python analyze_loco_candidates.py

# Run the dual analysis demo
python demo_dual_analysis.py

# Generated files:
# - results/corruption_geography_analysis.csv
# - results/corruption_geography_effects.png
# - results/country_volatility_analysis.csv
# - results/industry_volatility_analysis.csv
# - results/country_loco_candidates.csv
# - results/industry_loco_candidates.csv
# - results/loco_candidate_analysis.png
# - results/empire_standing_scores.csv
# - results/company_dominance_scores.csv
```

🌐 **[Live Demo & Documentation](https://mackthompson16.github.io/World-Order-Forecast)**


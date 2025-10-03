# Deep Learning to predict the future
## Project under construction....
This project utilizes public datasets and machine learning libraries to project socio-economic trends ten years in the future. The estimated accuracy is n% (TBD).


### Research question: Are Empires and Companies one in the same?

I split my data in two: empires and companies. From here, I trained three models:

1) World Order Forecast (WOF)
2) Market Share Forecast (MSF)
3) MSF Diluted from WOF

To develop an accuracy metric, I leave one country/industry out of training, and average a walk forward loss function. Because I could cherrypick the left out candidate with the most predictable history (to over estimate my model's accuracy), I also determine a candidate with an **average predictability** score for unbiased validation. Though, it can still be said that because global events are captured by all training data, the accuracy metric will still be overestimated.

I also include a corruption score for data purity and a geography score for optimism (more details below).

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

**Effect on Learning & Gradients**:
```
Learning Rate = Base Learning Rate × Corruption Trust Score
```

**Gradient Trust Impact**:
- **High corruption** → Lower learning rate (be more conservative with updates)
- **Low corruption** → Higher learning rate (trust gradients more, learn faster)
- **Data manipulation detection** → Model adjusts learning rate based on data reliability

#### Real-World Example

| Country | Corruption Level | Trust Score | Learning Rate Impact |
|---------|------------------|-------------|---------------------|
| Denmark | Very Low | 0.95 | 95% of base learning rate |
| USA | Moderate | 0.70 | 70% of base learning rate |
| China | High | 0.50 | 50% of base learning rate |
| Venezuela | Very High | 0.25 | 25% of base learning rate |

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

**Effect on Growth Trends**:
```
Adjusted Trend = Base Trend × Geography Multiplier
```

**Derivative Impact**:
- **Higher geography multiplier** → More optimistic growth trends (higher derivatives)
- **Lower geography multiplier** → More pessimistic growth trends (lower derivatives)
- **Long-term effects** → Geography affects the rate of change over time, not current standing

#### Real-World Examples

| Country | Geography Advantages | Calculation | Growth Multiplier | Trend Impact |
|---------|---------------------|-------------|------------------|--------------|
| USA | Coastal + Resource + Strategic | 1.10 × 1.08 × 1.05 | 1.25x | 25% faster growth |
| UK | Island + Coastal + Strategic | 1.15 × 1.10 × 1.05 | 1.33x | 33% faster growth |
| Switzerland | Mountain + Strategic | 0.92 × 1.05 | 0.97x | 3% slower growth |
| Saudi Arabia | Desert + Resource | 0.88 × 1.08 | 0.95x | 5% slower growth |

### Combined Effects on Model Training & Predictions

#### Model Training Approach

```
Learning Rate = Base Learning Rate × Corruption Trust Score
Growth Trend = Base Trend × Geography Multiplier
```

#### Training Strategy by Data Quality

**High Data Confidence** (Corruption > 0.8):
- **Learning Rate**: 80-95% of base rate (trust gradients)
- **Training**: Aggressive learning from these countries
- **Examples**: Denmark, Norway, Singapore

**Medium Data Confidence** (Corruption 0.5-0.8):
- **Learning Rate**: 50-80% of base rate (moderate trust)
- **Training**: Balanced learning with some caution
- **Examples**: USA, France, Italy

**Low Data Confidence** (Corruption < 0.5):
- **Learning Rate**: 10-50% of base rate (very conservative)
- **Training**: Minimal learning, mostly for pattern recognition
- **Examples**: China, Russia, Venezuela

#### Prediction Strategy by Geography

**High Geography Advantage** (Multiplier > 1.1):
- **Trend Adjustment**: More optimistic growth trajectories
- **Forecasting**: Higher likelihood of continued thriving
- **Examples**: USA, UK, Japan

**Neutral Geography** (Multiplier 0.9-1.1):
- **Trend Adjustment**: Standard growth trajectories
- **Forecasting**: Baseline expectations
- **Examples**: Germany, France, Italy

**Low Geography Advantage** (Multiplier < 0.9):
- **Trend Adjustment**: More pessimistic growth trajectories
- **Forecasting**: Lower likelihood of continued thriving
- **Examples**: Switzerland, Austria, landlocked countries

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
# Analyze corrected corruption and geography effects (learning rates & trends)
python demo_learning_rate_trends.py

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


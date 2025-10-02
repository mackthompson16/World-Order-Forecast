# Deep Leaning to predict the future

This project utilizes public datasets and machine learning libraries to train a model on socio-economic trends in hopes of predicting the future (10 year projection).


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

## Validation Strategy & Exclusions

### Recommended Leave-Out Strategy

For model validation using the leave-one-out approach, I recommend:

**Country to Leave Out**: **Switzerland (CHE)**
- Reasons: Small, stable economy with consistent democratic governance
- Highly predictable socio-economic patterns
- Strong data transparency and availability
- Neutral political stance reduces geopolitical volatility
- Well-established financial sector provides stable baseline

**Industry to Leave Out**: **Utilities Sector**
- Reasons: Highly regulated and stable industry
- Predictable growth patterns tied to population and economic growth
- Less susceptible to technological disruption compared to other sectors
- Consistent demand regardless of economic cycles
- Strong government oversight ensures data reliability

### Data Quality & Democratic Transparency

#### The Data Privatization Challenge

A critical limitation of this analysis is the **data privatization problem**. Many countries, particularly those with authoritarian governance structures, either:

1. **Restrict data access** - Limiting public availability of economic and social indicators
2. **Manipulate reported statistics** - Publishing misleading or fabricated data to project strength
3. **Lack institutional capacity** - Having insufficient infrastructure to collect reliable data
4. **Classify strategic information** - Treating economic data as state secrets

#### Democratic Data Advantage

**Initial Focus: Democratic Countries with Data Transparency**

To ensure model reliability, the initial implementation should focus on countries with:

- **Strong democratic institutions** (Freedom House score > 70)
- **Transparent statistical agencies** (World Bank Statistical Capacity score > 80)  
- **Independent central banks** with public data disclosure
- **Free press** to verify and challenge official statistics
- **International audit compliance** (IMF Article IV consultations)

**Recommended Initial Country Set:**
- **Tier 1**: USA, GBR, DEU, FRA, CAN, AUS, NLD, CHE, DNK, SWE
- **Tier 2**: JPN, KOR, ITA, ESP, BEL, AUT, NOR, FIN, NZL, IRL

**Countries to Exclude Initially:**
- **Data reliability concerns**: CHN, RUS, IRN, PRK, VEN, MMR
- **Limited transparency**: SAU, ARE, QAT, BRN
- **Institutional instability**: Countries with recent regime changes or ongoing conflicts

#### Future Expansion Strategy

1. **Phase 1**: Train models on high-quality democratic data
2. **Phase 2**: Develop techniques to detect and adjust for data manipulation
3. **Phase 3**: Gradually incorporate authoritarian countries with uncertainty quantification
4. **Phase 4**: Build specialized models for data-scarce environments

## Implementation Status

### ✅ Completed Features

- **Dual Data Framework**: Separate pipelines for Empire (country) and Company (industry) data
- **Democratic Data Focus**: Initial implementation prioritizes transparent, democratic countries
- **Comprehensive Data Schema**: 8 country factors + 4 industry factors with proper validation
- **Synthetic Data Generation**: Development-ready synthetic data for all sources
- **Validation Strategy**: Leave-one-out approach with Switzerland (country) and Utilities (industry)
- **Data Quality Framework**: Missing data handling, masking, and quality checks

### 🚧 Next Steps

1. **Real Data Integration**: Replace synthetic data with actual API calls to data sources
2. **Model Training**: Implement the three forecasting models (WOF, MSF, MSF Diluted)
3. **Uncertainty Quantification**: Add confidence intervals for predictions
4. **Authoritarian Data Handling**: Develop techniques for data-scarce/unreliable environments

### 🎯 Usage

```bash
# Run the dual analysis demo
python demo_dual_analysis.py

# This will generate:
# - results/empire_standing_scores.csv
# - results/company_dominance_scores.csv
```

🌐 **[Live Demo & Documentation](https://mackthompson16.github.io/World-Order-Forecast)**


# Deep Learning to predict the future
## Project under construction....
This repository utilizes public datasets and machine learning libraries to project socio-economic trends ten years in the future. The estimated accuracy is TBD.


### Research question: Are Empires and Companies one in the same?

I split my data in two: empires and companies. They are measured in the 8 dimensions below (Debt, Strength, Education...).
### Constants (for empires)
 - Corruption score: poor data purity will restrict the learning rate and pentalize projection.
 - Geography score:  weighs optimism based on positioning and natural resources.

### (I may add constants for companies as well but have not thought it out yet)

From here, I train three models:

1) World Order Forecast (WOF)
2) Market Share Forecast (MSF)
3) MSF Diluted from WOF

### Validation
I leave one country/industry out of training, and average a walk forward loss function. My initial idea was to cherrypick a candidate with a predictable history (to over estimate my model's accuracy), but a better approach is to determine the candidate with the most **average predictability** score. Though, it can still be said that because global events are captured by all training data, the accuracy metric will still be overestimated. This brings up the larger philsophical pursuit; ***it is impossible to judge the accuracy of predictive models, and man will never wield enough information to confidently predict the future.***

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

For model validation using the leave-one-out approach, we need to avoid **biased candidates** that are either too stable (like Switzerland/Utilities) or too volatile. Instead, we should select entities with **average volatility and predictability** for unbiased validation.

Select countries and industries that are:
- **Moderately volatile** (not too stable, not too erratic)
- **Average predictability** (some trend but not perfectly predictable)
- **Representative** of typical economic behavior
- **Sufficient data** (15+ years of observations)

## Constants

### Corruption Constant: Data Trustworthiness

The corruption constant models the **reliability and trustworthiness** of economic data based on institutional corruption levels. This addresses the critical issue that countries with high corruption often manipulate or restrict access to economic statistics.

**Gradient Trust Impact**:
- **High corruption** → Lower learning rate (be more conservative with updates)
- **Low corruption** → Higher learning rate (trust gradients more, learn faster)
- **Data manipulation detection** → Model adjusts learning rate based on data reliability

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
Adjusted Trend = Base Trend × Geography Multiplier
```

🌐 **[Live Demo & Documentation](https://mackthompson16.github.io/World-Order-Forecast)**


# Deep Learning to predict the future
## Project under construction....
This repository explores whether nations and corporations follow similar rise-and-fall dynamics, and whether machine learning can project these trajectories into the future. By combining public socio-economic datasets with deep learning architectures, the project attempts to generate ten-year forecasts of both empires (countries) and companies (industries).

At the core is a simple but provocative research question:
### Are Empires and Companies one in the same?

To test this, I divide the data into two domains:

- Empires (Countries): measured across eight structural dimensions such as debt, military strength, education, innovation, and reserve currency status.
- Companies (Industries): measured across parallel dimensions such as market capitalization, R&D spending, revenue growth, employment share, and global market share.

Additional constants are layered into the models to account for corruption (data trustworthiness) and geography (structural advantages or constraints like natural resources, trade access, and climate). These are treated not as predictors of year-to-year variance, but as underlying priors that shape long-term trajectories.

From this foundation, I train three experimental models:

1) World Order Forecast (WOF): projecting the relative standing of nations.
2) Market Share Forecast (MSF): projecting industry and corporate dominance.
3) MSF Diluted from WOF: combining the two perspectives to test whether national and corporate cycles reinforce or diverge from one another.

### Validation Philosophy

To measure accuracy, the models use a leave-one-out cross-validation strategy: excluding one country or industry from training, then testing predictions against its historical trajectory. Instead of cherry-picking predictable cases (which would inflate accuracy), the focus is on average predictability—candidates with moderate volatility and representative dynamics.

Still, all forecasts remain provisional. Global events are interdependent, and any model trained on past data necessarily inherits both its scope and its blind spots. This highlights a larger truth:
***predictive models can never know the future, but they can help uncover the structures and cycles that shape it.***

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


# Data Sources and Schema

This document describes the data sources, expected formats, and schema for the country standing forecast project.

## Data Sources

### 1. World Bank WDI (World Development Indicators)

**Source**: [World Bank Open Data](https://data.worldbank.org/)

**Indicators**:
- **Education**: `SE.SEC.DURS.AG` - Average years of schooling (population 25+)
- **Trade Share**: `NE.TRD.GNFS.ZS` - Trade as % of GDP
- **Debt**: `GC.DOD.TOTL.GD.ZS` - Government debt as % of GDP

**Expected CSV Format**:
```csv
country,year,education,trade_share,debt
USA,2000,12.5,25.3,55.2
USA,2001,12.6,24.8,56.1
...
```

### 2. WIPO (World Intellectual Property Organization)

**Source**: [WIPO Patent Statistics](https://www.wipo.int/ipstats/en/)

**Indicator**: Patent applications per million population

**Expected CSV Format**:
```csv
country,year,innovation
USA,2000,125.3
USA,2001,128.7
...
```

### 3. WEF (World Economic Forum)

**Source**: [Global Competitiveness Report](https://www.weforum.org/reports/)

**Indicator**: Global Competitiveness Index (0-100 scale)

**Expected CSV Format**:
```csv
country,year,competitiveness
USA,2000,78.5
USA,2001,79.2
...
```

### 4. SIPRI (Stockholm International Peace Research Institute)

**Source**: [SIPRI Military Expenditure Database](https://www.sipri.org/databases/milex)

**Indicator**: Military expenditure as % of GDP

**Expected CSV Format**:
```csv
country,year,military
USA,2000,3.2
USA,2001,3.4
...
```

### 5. IMF COFER (Currency Composition of Foreign Exchange Reserves)

**Source**: [IMF COFER Database](https://www.imf.org/en/Data/Statistics/COFER)

**Indicator**: Currency share in global foreign exchange reserves

**Expected CSV Format**:
```csv
country,year,reserve_currency_proxy
USA,2000,65.2
USA,2001,64.8
...
```

### 6. GFCI (Global Financial Centres Index)

**Source**: [GFCI Reports](https://www.longfinance.net/programmes/financial-centres/)

**Indicator**: Financial center development index (0-100 scale)

**Expected CSV Format**:
```csv
country,year,financial_center_proxy
USA,2000,85.3
USA,2001,86.1
...
```

## Data Schema

### Country Data Model

```python
class CountryData(BaseModel):
    country: str                    # Country code (ISO 3-letter)
    year: int                       # Year (2000-2023)
    
    # Factor values
    education: Optional[float] = None
    innovation: Optional[float] = None
    competitiveness: Optional[float] = None
    military: Optional[float] = None
    trade_share: Optional[float] = None
    reserve_currency_proxy: Optional[float] = None
    financial_center_proxy: Optional[float] = None
    debt: Optional[float] = None
    
    # Optional exogenous variables
    gdp_per_capita: Optional[float] = None
    population: Optional[float] = None
    
    # Data quality flags
    mask_education: bool = False
    mask_innovation: bool = False
    mask_competitiveness: bool = False
    mask_military: bool = False
    mask_trade_share: bool = False
    mask_reserve_currency_proxy: bool = False
    mask_financial_center_proxy: bool = False
    mask_debt: bool = False
```

### Factor Definitions

| Factor | Unit | Min | Max | Higher is Better | Transform |
|--------|------|-----|-----|------------------|-----------|
| education | years | 0 | 20 | Yes | level |
| innovation | patents/million | 0.1 | 10000 | Yes | log |
| competitiveness | index | 0 | 100 | Yes | level |
| military | % of GDP | 0 | 20 | Yes | level |
| trade_share | % of GDP | 0 | 200 | Yes | level |
| reserve_currency_proxy | % of global reserves | 0 | 100 | Yes | level |
| financial_center_proxy | index | 0 | 100 | Yes | level |
| debt | % of GDP | 0 | 300 | No (inverted) | level |

### Data Quality Requirements

#### Minimum Coverage
- **Countries**: At least 5 countries per year
- **Years**: At least 15 years per country
- **Missing Data**: Maximum 30% missing per factor

#### Data Validation
- **Range Checks**: Values within expected min/max bounds
- **Consistency**: No negative values for positive-only factors
- **Temporal Consistency**: Reasonable year-over-year changes

#### Missing Data Handling
- **Imputation**: KNN imputation with k=5 neighbors
- **Masking**: Track missing data with boolean flags
- **No Leakage**: Fit imputers on training data only

## Expected Data Format

### Panel Data Format

The final panel data should be in the following format:

```csv
country,year,education,innovation,competitiveness,military,trade_share,reserve_currency_proxy,financial_center_proxy,debt,mask_education,mask_innovation,mask_competitiveness,mask_military,mask_trade_share,mask_reserve_currency_proxy,mask_financial_center_proxy,mask_debt
USA,2000,12.5,125.3,78.5,3.2,25.3,65.2,85.3,55.2,False,False,False,False,False,False,False,False
USA,2001,12.6,128.7,79.2,3.4,24.8,64.8,86.1,56.1,False,False,False,False,False,False,False,False
...
CHN,2000,8.2,15.7,45.3,2.1,45.6,0.1,25.4,35.8,False,False,False,False,False,False,False,False
...
```

### Country Codes

Use ISO 3-letter country codes:

| Country | Code | Country | Code |
|---------|------|---------|------|
| United States | USA | China | CHN |
| Germany | DEU | Japan | JPN |
| United Kingdom | GBR | France | FRA |
| India | IND | Brazil | BRA |
| Canada | CAN | Australia | AUS |
| Russia | RUS | South Korea | KOR |
| Italy | ITA | Spain | ESP |
| Mexico | MEX | Indonesia | IDN |

## Data Loading Implementation

### Current Status

The data loaders in `src/data_ingest/` currently provide synthetic data generation for development and testing. To use real data:

1. **Complete TODO sections** in each loader file
2. **Implement API calls** or CSV parsing for each data source
3. **Add data validation** and error handling
4. **Update factor mappings** if needed

### Example Implementation

```python
def load_wdi_data(data_dir: Path, countries: List[str], years: List[int]) -> pd.DataFrame:
    """
    Load World Bank WDI data.
    
    TODO: Implement actual WDI API integration
    """
    # Option 1: Use World Bank API
    # import wbgapi as wb
    # data = wb.data.DataFrame(['SE.SEC.DURS.AG', 'NE.TRD.GNFS.ZS'], 
    #                          countries, years, skipBlanks=True)
    
    # Option 2: Load from CSV files
    # df = pd.read_csv(data_dir / "wdi_data.csv")
    # df = df[df['country'].isin(countries) & df['year'].isin(years)]
    
    # For now, return synthetic data
    return generate_synthetic_wdi_data(countries, years)
```

## Data Updates

### Frequency
- **Annual Updates**: Most indicators are updated annually
- **Quarterly Updates**: Some indicators (e.g., GDP) may be available quarterly
- **Lag**: Data typically available with 1-2 year lag

### Version Control
- **Data Versioning**: Track data versions and update dates
- **Backup**: Maintain historical data snapshots
- **Change Log**: Document data source changes

## Troubleshooting

### Common Issues

1. **Missing Data**: Check data source availability and coverage
2. **Format Changes**: Verify CSV column names and data types
3. **API Limits**: Implement rate limiting for API calls
4. **Data Quality**: Validate ranges and consistency

### Validation Scripts

```python
# Check data quality
def validate_panel_data(df: pd.DataFrame) -> Dict[str, Any]:
    """Validate panel data quality."""
    issues = []
    
    # Check required columns
    required_cols = ['country', 'year'] + FACTOR_NAMES
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        issues.append(f"Missing columns: {missing_cols}")
    
    # Check data ranges
    for factor in FACTOR_NAMES:
        if factor in df.columns:
            factor_def = FACTORS[factor]
            out_of_range = df[factor].notna() & (
                (df[factor] < factor_def.min_value) | 
                (df[factor] > factor_def.max_value)
            )
            if out_of_range.any():
                issues.append(f"{factor}: {out_of_range.sum()} values out of range")
    
    return {"issues": issues, "valid": len(issues) == 0}
```

## Contact

For questions about data sources or schema, please refer to the main README.md or create an issue in the repository.

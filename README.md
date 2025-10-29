# Deep Learning to predict the future

## Status: Empire Forecast model is (almost) 60% accuracy
 - Will move on to markets once I get to 70%
# Current forecast
![Empire Composite Standing - Projected 30 years](src/results/projection_2054.png)
This repository explores whether nations and corporations follow similar rise-and-fall dynamics, and whether machine learning can project these trajectories into the future. By combining public socio-economic datasets with deep learning architectures, the project attempts to generate ten-year forecasts of both empires (countries) and companies (industries).

## Background

I was first introduced to this idea reading Ray Dalio's compelling piece, [Principles for dealing with the changing world order](https://www.economicprinciples.org/DalioChangingWorldOrderCharts.pdf). 

His team assembled data that dates back nearly 1000 years from hundred of cross referenced sources. My first instinct was to reproduce his graphs, but much of his data was privatized and internal. Therefore, the primary limitation with my project-as with any deep learning pursuit-is data. I could only pull from a few publically available sources within the scope of my resources.

# Data coverage
![Metric Distrubution grid](src/results/metrics_grid.png)
![Metric Distrubution grid](src/results/geography_index.png)
- Further explanation of assembley in the folder's readme
- Need to get more financial / Currency data (mostly privatized)
## Training Data

- Located in src/results/metrics.csv
    -   [country_name,ISO3,year,EDU,MIL,ECON,TRAD,RESV,FIN,INV,CMPT,INDEX,geography_index]
- Each Country ranges between 1800 - 2024
- Atleast 4 available metrics
- Each includes a geography index, which scores optimism;
        - Expected Future Growth=f(Xt​)+β⋅geography 
# Training

- If a country has < 100 years of valid data, exclude.
    - exclude the validation candidate; denmark
- rolling windows of length=50 ( gaussian)
-  forecast horizon=30.
- Use the last known geography_index in the input window
    - Add β·geography as an additive bias to the output head
- Temporal ConvNet (1D Conv over time) on the K channels.
- Output a 30-step forecast for each target metric
- Build a boolean mask (K,50) for inputs; use masked loss.
- Only generate loss on targets actually present in the 30y horizon.

## Validation Strategy
![Empire Composite Standing - Projected 30 years](src/results/trajectory_dnk_index_spaghetti.png)
- Current candidate to exclude: denmark
- walk forward, regression based evaluation
- Percent of deltas with correct sign at each year gives the final accuracy

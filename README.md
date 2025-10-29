# Can I use Deep Learning to model political trends?


## Results
![Empire Composite Standing - Projected 30 years](src/results/projection_2054.png)
## Conclusion

While it is conceptually possible to apply deep learning to forecast world orders, the scope of data required to capture the cyclical nature of global politics exceeds the resources available for this independent study. Meaningful temporal windows would need to span approximately 250 years per empire, yet most countries only provide about 150 years of reliable data. The model’s 60% accuracy suggests that some predictive signal exists, but overall performance represents only a modest 10% improvement over random chance and aligns closely with results produced by a 50-year windowed linear regression approach. More traditional frameworks, such as statistical arbitrage, may therefore prove more suitable for this kind of analysis.

Nevertheless, as I continue my research, I hypothesize that the greater availability of data within the American stock market—and the shorter life cycles of individual companies—will allow for more precise modeling of economic trajectories on a monthly scale. This insight motivates the next phase of my work: developing a data-driven trading tool capable of forecasting corporate performance trends with higher temporal accuracy.

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

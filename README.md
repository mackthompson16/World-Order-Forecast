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

Public Data for Empires 
Metric | Source | Range |
Global Debt | IMF | 
Military Strength | SIPRI|
GDP | IMF |
Reserve Currency | IMF |
Education | WIPO |

🌐 **[Live Demo & Documentation](https://mackthompson16.github.io/World-Order-Forecast)**


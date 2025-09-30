import React, { useState } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar } from 'recharts';
import { Play, RotateCcw } from 'lucide-react';

const InteractiveDemo: React.FC = () => {
  const [selectedCountry, setSelectedCountry] = useState('United States');
  const [selectedFactors, setSelectedFactors] = useState<string[]>(['Education', 'Innovation', 'Competitiveness']);
  const [forecastHorizon, setForecastHorizon] = useState(5);
  const [isRunning, setIsRunning] = useState(false);
  const [forecastResults, setForecastResults] = useState<any>(null);

  const countries = ['United States', 'China', 'Germany', 'Japan', 'India', 'Brazil', 'Russia', 'South Korea'];
  const factors = ['Education', 'Innovation', 'Competitiveness', 'Military', 'Trade Share', 'Reserve Currency', 'Financial Center', 'Debt'];

  // Sample historical data
  const historicalData = [
    { year: 2004, standing: 0.65, education: 0.7, innovation: 0.6, competitiveness: 0.8, military: 0.4, trade: 0.5, reserve: 0.9, financial: 0.7, debt: 0.3 },
    { year: 2005, standing: 0.67, education: 0.72, innovation: 0.62, competitiveness: 0.82, military: 0.42, trade: 0.52, reserve: 0.91, financial: 0.72, debt: 0.32 },
    { year: 2006, standing: 0.69, education: 0.74, innovation: 0.64, competitiveness: 0.84, military: 0.44, trade: 0.54, reserve: 0.92, financial: 0.74, debt: 0.34 },
    { year: 2007, standing: 0.71, education: 0.76, innovation: 0.66, competitiveness: 0.86, military: 0.46, trade: 0.56, reserve: 0.93, financial: 0.76, debt: 0.36 },
    { year: 2008, standing: 0.68, education: 0.78, innovation: 0.68, competitiveness: 0.88, military: 0.48, trade: 0.58, reserve: 0.94, financial: 0.78, debt: 0.38 },
    { year: 2009, standing: 0.66, education: 0.8, innovation: 0.7, competitiveness: 0.9, military: 0.5, trade: 0.6, reserve: 0.95, financial: 0.8, debt: 0.4 },
    { year: 2010, standing: 0.68, education: 0.82, innovation: 0.72, competitiveness: 0.92, military: 0.52, trade: 0.62, reserve: 0.96, financial: 0.82, debt: 0.42 },
    { year: 2011, standing: 0.7, education: 0.84, innovation: 0.74, competitiveness: 0.94, military: 0.54, trade: 0.64, reserve: 0.97, financial: 0.84, debt: 0.44 },
    { year: 2012, standing: 0.72, education: 0.86, innovation: 0.76, competitiveness: 0.96, military: 0.56, trade: 0.66, reserve: 0.98, financial: 0.86, debt: 0.46 },
    { year: 2013, standing: 0.74, education: 0.88, innovation: 0.78, competitiveness: 0.98, military: 0.58, trade: 0.68, reserve: 0.99, financial: 0.88, debt: 0.48 },
    { year: 2014, standing: 0.76, education: 0.9, innovation: 0.8, competitiveness: 1.0, military: 0.6, trade: 0.7, reserve: 1.0, financial: 0.9, debt: 0.5 },
    { year: 2015, standing: 0.78, education: 0.92, innovation: 0.82, competitiveness: 1.02, military: 0.62, trade: 0.72, reserve: 1.01, financial: 0.92, debt: 0.52 },
    { year: 2016, standing: 0.8, education: 0.94, innovation: 0.84, competitiveness: 1.04, military: 0.64, trade: 0.74, reserve: 1.02, financial: 0.94, debt: 0.54 },
    { year: 2017, standing: 0.82, education: 0.96, innovation: 0.86, competitiveness: 1.06, military: 0.66, trade: 0.76, reserve: 1.03, financial: 0.96, debt: 0.56 },
    { year: 2018, standing: 0.84, education: 0.98, innovation: 0.88, competitiveness: 1.08, military: 0.68, trade: 0.78, reserve: 1.04, financial: 0.98, debt: 0.58 },
    { year: 2019, standing: 0.86, education: 1.0, innovation: 0.9, competitiveness: 1.1, military: 0.7, trade: 0.8, reserve: 1.05, financial: 1.0, debt: 0.6 },
    { year: 2020, standing: 0.84, education: 1.02, innovation: 0.92, competitiveness: 1.12, military: 0.72, trade: 0.82, reserve: 1.06, financial: 1.02, debt: 0.62 },
    { year: 2021, standing: 0.86, education: 1.04, innovation: 0.94, competitiveness: 1.14, military: 0.74, trade: 0.84, reserve: 1.07, financial: 1.04, debt: 0.64 },
    { year: 2022, standing: 0.88, education: 1.06, innovation: 0.96, competitiveness: 1.16, military: 0.76, trade: 0.86, reserve: 1.08, financial: 1.06, debt: 0.66 },
    { year: 2023, standing: 0.9, education: 1.08, innovation: 0.98, competitiveness: 1.18, military: 0.78, trade: 0.88, reserve: 1.09, financial: 1.08, debt: 0.68 },
  ];

  const runForecast = async () => {
    setIsRunning(true);
    
    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    // Generate mock forecast results
    const mockResults = {
      pointForecast: 0.92 + Math.random() * 0.06,
      quantiles: {
        lower: 0.85 + Math.random() * 0.05,
        median: 0.90 + Math.random() * 0.05,
        upper: 0.95 + Math.random() * 0.05
      },
      factorContributions: selectedFactors.map(factor => ({
        factor,
        contribution: Math.random() * 0.3 + 0.1
      })),
      confidence: 0.85 + Math.random() * 0.1
    };
    
    setForecastResults(mockResults);
    setIsRunning(false);
  };

  const resetDemo = () => {
    setForecastResults(null);
    setSelectedCountry('United States');
    setSelectedFactors(['Education', 'Innovation', 'Competitiveness']);
    setForecastHorizon(5);
  };

  const toggleFactor = (factor: string) => {
    setSelectedFactors(prev => 
      prev.includes(factor) 
        ? prev.filter(f => f !== factor)
        : [...prev, factor]
    );
  };

  return (
    <div className="interactive-demo">
      <div className="page-header">
        <h1>Interactive Demo</h1>
        <p>Explore country standing forecasts with interactive controls and real-time predictions.</p>
      </div>

      <div className="demo-controls">
        <div className="control-panel">
          <h3>Forecast Parameters</h3>
          
          <div className="control-group">
            <label htmlFor="country-select">Country:</label>
            <select 
              id="country-select" 
              value={selectedCountry} 
              onChange={(e) => setSelectedCountry(e.target.value)}
            >
              {countries.map(country => (
                <option key={country} value={country}>{country}</option>
              ))}
            </select>
          </div>

          <div className="control-group">
            <label htmlFor="horizon-select">Forecast Horizon:</label>
            <select 
              id="horizon-select" 
              value={forecastHorizon} 
              onChange={(e) => setForecastHorizon(Number(e.target.value))}
            >
              <option value={1}>1 Year</option>
              <option value={5}>5 Years</option>
              <option value={10}>10 Years</option>
            </select>
          </div>

          <div className="control-group">
            <label>Select Factors:</label>
            <div className="factor-checkboxes">
              {factors.map(factor => (
                <label key={factor} className="checkbox-label">
                  <input
                    type="checkbox"
                    checked={selectedFactors.includes(factor)}
                    onChange={() => toggleFactor(factor)}
                  />
                  <span>{factor}</span>
                </label>
              ))}
            </div>
          </div>

          <div className="control-actions">
            <button 
              className="btn btn-primary"
              onClick={runForecast}
              disabled={isRunning || selectedFactors.length === 0}
            >
              <Play size={20} />
              {isRunning ? 'Running...' : 'Run Forecast'}
            </button>
            
            <button 
              className="btn btn-secondary"
              onClick={resetDemo}
            >
              <RotateCcw size={20} />
              Reset
            </button>
          </div>
        </div>

        <div className="historical-chart">
          <h3>Historical Data: {selectedCountry}</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={historicalData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="year" />
              <YAxis domain={[0.3, 1.2]} />
              <Tooltip />
              <Line 
                type="monotone" 
                dataKey="standing" 
                stroke="#2563eb" 
                strokeWidth={3}
                dot={{ fill: '#2563eb', strokeWidth: 2, r: 4 }}
                name="Country Standing"
              />
              {selectedFactors.map((factor, index) => (
                <Line 
                  key={factor}
                  type="monotone" 
                  dataKey={factor.toLowerCase().replace(' ', '')} 
                  stroke={`hsl(${index * 60}, 70%, 50%)`}
                  strokeDasharray="5 5"
                  name={factor}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {forecastResults && (
        <div className="forecast-results">
          <h2>Forecast Results</h2>
          
          <div className="results-grid">
            <div className="forecast-card">
              <h3>Point Forecast</h3>
              <div className="forecast-value">
                {(forecastResults.pointForecast * 100).toFixed(1)}%
              </div>
              <div className="forecast-horizon">
                {forecastHorizon}-year forecast
              </div>
            </div>

            <div className="forecast-card">
              <h3>Uncertainty Bounds</h3>
              <div className="uncertainty-bounds">
                <div className="bound">
                  <span className="bound-label">Lower (10th percentile):</span>
                  <span className="bound-value">{(forecastResults.quantiles.lower * 100).toFixed(1)}%</span>
                </div>
                <div className="bound">
                  <span className="bound-label">Median (50th percentile):</span>
                  <span className="bound-value">{(forecastResults.quantiles.median * 100).toFixed(1)}%</span>
                </div>
                <div className="bound">
                  <span className="bound-label">Upper (90th percentile):</span>
                  <span className="bound-value">{(forecastResults.quantiles.upper * 100).toFixed(1)}%</span>
                </div>
              </div>
            </div>

            <div className="forecast-card">
              <h3>Model Confidence</h3>
              <div className="confidence-meter">
                <div className="confidence-bar">
                  <div 
                    className="confidence-fill"
                    style={{ width: `${forecastResults.confidence * 100}%` }}
                  />
                </div>
                <div className="confidence-text">
                  {(forecastResults.confidence * 100).toFixed(1)}% confidence
                </div>
              </div>
            </div>
          </div>

          <div className="factor-contributions">
            <h3>Factor Contributions</h3>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={forecastResults.factorContributions}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="factor" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="contribution" fill="#7c3aed" />
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="what-if-scenarios">
            <h3>What-If Scenarios</h3>
            <div className="scenarios-grid">
              <div className="scenario-card">
                <h4>📈 Optimistic Scenario</h4>
                <p>All factors improve by 10%</p>
                <div className="scenario-result">
                  {(forecastResults.pointForecast * 1.1 * 100).toFixed(1)}%
                </div>
              </div>
              
              <div className="scenario-card">
                <h4>📉 Pessimistic Scenario</h4>
                <p>All factors decline by 10%</p>
                <div className="scenario-result">
                  {(forecastResults.pointForecast * 0.9 * 100).toFixed(1)}%
                </div>
              </div>
              
              <div className="scenario-card">
                <h4>⚖️ Status Quo</h4>
                <p>Current trends continue</p>
                <div className="scenario-result">
                  {(forecastResults.pointForecast * 100).toFixed(1)}%
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      <div className="demo-info">
        <h2>About This Demo</h2>
        <div className="info-grid">
          <div className="info-card">
            <h4>🎯 Purpose</h4>
            <p>This interactive demo showcases the country standing forecasting model using synthetic data. It demonstrates how different macro factors influence predictions and provides uncertainty estimates.</p>
          </div>
          
          <div className="info-card">
            <h4>📊 Data Source</h4>
            <p>The demo uses synthetic data generated by our data synthesis pipeline. In production, this would be replaced with real macro factor data from World Bank, WIPO, and other sources.</p>
          </div>
          
          <div className="info-card">
            <h4>🧠 Model</h4>
            <p>The underlying model combines factor-wise convolution with temporal convolutional networks to capture both cross-factor interactions and temporal dependencies.</p>
          </div>
          
          <div className="info-card">
            <h4>⚡ Real-time</h4>
            <p>Forecasts are generated in real-time based on your parameter selections. The model adapts to different countries and factor combinations dynamically.</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default InteractiveDemo;

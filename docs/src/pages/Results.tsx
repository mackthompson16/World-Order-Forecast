import React, { useState } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar } from 'recharts';
import { TrendingUp, TrendingDown, Target, AlertCircle } from 'lucide-react';

const Results: React.FC = () => {
  const [selectedCountry, setSelectedCountry] = useState('United States');
  const [selectedMetric, setSelectedMetric] = useState('MASE');

  // Sample data for demonstration
  const forecastData = [
    { year: 2020, actual: 0.75, predicted_1y: 0.72, predicted_5y: 0.68, predicted_10y: 0.65 },
    { year: 2021, actual: 0.78, predicted_1y: 0.75, predicted_5y: 0.71, predicted_10y: 0.68 },
    { year: 2022, actual: 0.82, predicted_1y: 0.79, predicted_5y: 0.75, predicted_10y: 0.72 },
    { year: 2023, actual: 0.85, predicted_1y: 0.83, predicted_5y: 0.79, predicted_10y: 0.76 },
    { year: 2024, actual: null, predicted_1y: 0.87, predicted_5y: 0.83, predicted_10y: 0.80 },
    { year: 2025, actual: null, predicted_1y: 0.89, predicted_5y: 0.85, predicted_10y: 0.82 },
    { year: 2026, actual: null, predicted_1y: null, predicted_5y: 0.87, predicted_10y: 0.84 },
    { year: 2027, actual: null, predicted_1y: null, predicted_5y: 0.89, predicted_10y: 0.86 },
    { year: 2028, actual: null, predicted_1y: null, predicted_5y: 0.91, predicted_10y: 0.88 },
    { year: 2029, actual: null, predicted_1y: null, predicted_5y: 0.93, predicted_10y: 0.90 },
    { year: 2030, actual: null, predicted_1y: null, predicted_5y: null, predicted_10y: 0.92 },
  ];

  const locoResults = [
    { country: 'United States', mase_1y: 0.85, mase_5y: 0.92, mase_10y: 1.15, spearman: 0.78 },
    { country: 'China', mase_1y: 0.78, mase_5y: 0.89, mase_10y: 1.08, spearman: 0.82 },
    { country: 'Germany', mase_1y: 0.82, mase_5y: 0.95, mase_10y: 1.12, spearman: 0.75 },
    { country: 'Japan', mase_1y: 0.88, mase_5y: 0.98, mase_10y: 1.20, spearman: 0.73 },
    { country: 'India', mase_1y: 0.75, mase_5y: 0.87, mase_10y: 1.05, spearman: 0.80 },
    { country: 'Brazil', mase_1y: 0.90, mase_5y: 1.02, mase_10y: 1.25, spearman: 0.68 },
    { country: 'Russia', mase_1y: 0.95, mase_5y: 1.08, mase_10y: 1.30, spearman: 0.65 },
    { country: 'South Korea', mase_1y: 0.80, mase_5y: 0.90, mase_10y: 1.10, spearman: 0.77 },
  ];

  const factorImportance = [
    { factor: 'Education', importance: 0.18, trend: 'up' },
    { factor: 'Innovation', importance: 0.16, trend: 'up' },
    { factor: 'Competitiveness', importance: 0.15, trend: 'up' },
    { factor: 'Military', importance: 0.14, trend: 'down' },
    { factor: 'Trade Share', importance: 0.13, trend: 'up' },
    { factor: 'Reserve Currency', importance: 0.12, trend: 'stable' },
    { factor: 'Financial Center', importance: 0.08, trend: 'up' },
    { factor: 'Debt', importance: 0.04, trend: 'down' },
  ];

  const uncertaintyData = [
    { horizon: '1 Year', lower_bound: 0.05, upper_bound: 0.12, median: 0.08 },
    { horizon: '5 Years', lower_bound: 0.08, upper_bound: 0.18, median: 0.13 },
    { horizon: '10 Years', lower_bound: 0.12, upper_bound: 0.25, median: 0.18 },
  ];

  const countries = ['United States', 'China', 'Germany', 'Japan', 'India', 'Brazil', 'Russia', 'South Korea'];
  const metrics = ['MASE', 'RMSE', 'MAE', 'Spearman'];

  return (
    <div className="results">
      <div className="page-header">
        <h1>Model Performance & Results</h1>
        <p>Comprehensive evaluation of our country standing forecasting model across different countries and time horizons.</p>
      </div>

      <div className="controls">
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
          <label htmlFor="metric-select">Metric:</label>
          <select 
            id="metric-select" 
            value={selectedMetric} 
            onChange={(e) => setSelectedMetric(e.target.value)}
          >
            {metrics.map(metric => (
              <option key={metric} value={metric}>{metric}</option>
            ))}
          </select>
        </div>
      </div>

      <div className="results-grid">
        <div className="chart-container">
          <h3>Forecast vs Actual: {selectedCountry}</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={forecastData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="year" />
              <YAxis domain={[0.6, 1.0]} />
              <Tooltip />
              <Line 
                type="monotone" 
                dataKey="actual" 
                stroke="#2563eb" 
                strokeWidth={3}
                dot={{ fill: '#2563eb', strokeWidth: 2, r: 4 }}
                name="Actual"
              />
              <Line 
                type="monotone" 
                dataKey="predicted_1y" 
                stroke="#dc2626" 
                strokeDasharray="5 5"
                name="1-Year Forecast"
              />
              <Line 
                type="monotone" 
                dataKey="predicted_5y" 
                stroke="#059669" 
                strokeDasharray="5 5"
                name="5-Year Forecast"
              />
              <Line 
                type="monotone" 
                dataKey="predicted_10y" 
                stroke="#7c3aed" 
                strokeDasharray="5 5"
                name="10-Year Forecast"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="chart-container">
          <h3>LOCO Performance by Country</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={locoResults}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="country" angle={-45} textAnchor="end" height={80} />
              <YAxis />
              <Tooltip />
              <Bar dataKey="mase_1y" fill="#2563eb" name="1-Year MASE" />
              <Bar dataKey="mase_5y" fill="#059669" name="5-Year MASE" />
              <Bar dataKey="mase_10y" fill="#dc2626" name="10-Year MASE" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="chart-container">
          <h3>Factor Importance Analysis</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={factorImportance} layout="horizontal">
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" domain={[0, 0.2]} />
              <YAxis dataKey="factor" type="category" width={120} />
              <Tooltip />
              <Bar dataKey="importance" fill="#7c3aed" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="chart-container">
          <h3>Uncertainty by Forecast Horizon</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={uncertaintyData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="horizon" />
              <YAxis />
              <Tooltip />
              <Bar dataKey="lower_bound" stackId="a" fill="#fbbf24" name="Lower Bound" />
              <Bar dataKey="median" stackId="a" fill="#3b82f6" name="Median" />
              <Bar dataKey="upper_bound" stackId="a" fill="#f59e0b" name="Upper Bound" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="performance-summary">
        <h2>Performance Summary</h2>
        <div className="summary-grid">
          <div className="summary-card">
            <div className="summary-icon">
              <Target />
            </div>
            <div className="summary-content">
              <h4>Overall MASE</h4>
              <div className="summary-value">0.85</div>
              <div className="summary-trend">
                <TrendingDown size={16} />
                <span>12% improvement vs baseline</span>
              </div>
            </div>
          </div>

          <div className="summary-card">
            <div className="summary-icon">
              <TrendingUp />
            </div>
            <div className="summary-content">
              <h4>Spearman Correlation</h4>
              <div className="summary-value">0.75</div>
              <div className="summary-trend">
                <TrendingUp size={16} />
                <span>Strong rank correlation</span>
              </div>
            </div>
          </div>

          <div className="summary-card">
            <div className="summary-icon">
              <AlertCircle />
            </div>
            <div className="summary-content">
              <h4>Interval Coverage</h4>
              <div className="summary-value">89%</div>
              <div className="summary-trend">
                <TrendingUp size={16} />
                <span>Close to 90% target</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="key-insights">
        <h2>Key Insights</h2>
        <div className="insights-grid">
          <div className="insight-card">
            <h4>🎯 Short-term Accuracy</h4>
            <p>1-year forecasts show excellent accuracy (MASE &lt; 0.9) across most countries, with developed countries performing slightly better.</p>
          </div>
          
          <div className="insight-card">
            <h4>📈 Factor Importance</h4>
            <p>Education and Innovation emerge as the most predictive factors, while Debt shows the lowest importance for long-term forecasting.</p>
          </div>
          
          <div className="insight-card">
            <h4>🌍 Country Generalization</h4>
            <p>Model generalizes well across different country types, though emerging markets show slightly higher uncertainty in long-term forecasts.</p>
          </div>
          
          <div className="insight-card">
            <h4>⏰ Horizon Effects</h4>
            <p>Uncertainty increases predictably with forecast horizon, with 10-year forecasts showing 2-3x higher uncertainty than 1-year forecasts.</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Results;

import React from 'react';
import { Link } from 'react-router-dom';
import { ArrowRight, Brain, BarChart3, PlayCircle, Globe, Shield } from 'lucide-react';

const Home: React.FC = () => {
  return (
    <div className="home">
      <div className="hero">
        <h1>Country Standing Forecast</h1>
        <p className="hero-subtitle">
          A machine learning approach to forecasting a country's "standing" using 8 macro factors 
          inspired by Ray Dalio's framework for measuring national strength.
        </p>
        
        <div className="hero-actions">
          <Link to="/demo" className="btn btn-primary">
            <PlayCircle size={20} />
            Interactive Demo
          </Link>
          <Link to="/architecture" className="btn btn-secondary">
            <Brain size={20} />
            Explore Architecture
          </Link>
        </div>
      </div>

      <div className="features-grid">
        <div className="feature-card">
          <div className="feature-icon">
            <Brain />
          </div>
          <h3>Neural Network Architecture</h3>
          <p>Factor convolution + Temporal TCN with attention mechanism for capturing complex relationships</p>
        </div>

        <div className="feature-card">
          <div className="feature-icon">
            <BarChart3 />
          </div>
          <h3>Multi-Horizon Forecasting</h3>
          <p>Predict country standing at 1, 5, and 10-year horizons with uncertainty quantification</p>
        </div>

        <div className="feature-card">
          <div className="feature-icon">
            <Shield />
          </div>
          <h3>Robust Evaluation</h3>
          <p>Leave-One-Country-Out cross-validation ensures generalization across different countries</p>
        </div>

        <div className="feature-card">
          <div className="feature-icon">
            <Globe />
          </div>
          <h3>8 Macro Factors</h3>
          <p>Education, Innovation, Competitiveness, Military, Trade, Reserve Currency, Financial Center, Debt</p>
        </div>
      </div>

      <div className="quick-start">
        <h2>Quick Start</h2>
        <div className="steps">
          <div className="step">
            <div className="step-number">1</div>
            <div className="step-content">
              <h4>Generate Synthetic Data</h4>
              <code>python make_synth_data.py</code>
            </div>
          </div>
          
          <div className="step">
            <div className="step-number">2</div>
            <div className="step-content">
              <h4>Launch Interactive UI</h4>
              <code>streamlit run src/ui_demo.py</code>
            </div>
          </div>
          
          <div className="step">
            <div className="step-number">3</div>
            <div className="step-content">
              <h4>Run Jupyter Tutorial</h4>
              <code>jupyter notebook notebooks/00_quickstart.ipynb</code>
            </div>
          </div>
        </div>
      </div>

      <div className="architecture-overview">
        <h2>Architecture Overview</h2>
        <div className="architecture-diagram">
          <div className="arch-component">
            <div className="arch-icon">📊</div>
            <h4>8 Macro Factors</h4>
            <p>Input data</p>
          </div>
          
          <ArrowRight className="arch-arrow" />
          
          <div className="arch-component">
            <div className="arch-icon">🧠</div>
            <h4>Factor Convolution</h4>
            <p>Capture interactions</p>
          </div>
          
          <ArrowRight className="arch-arrow" />
          
          <div className="arch-component">
            <div className="arch-icon">⏰</div>
            <h4>Temporal TCN</h4>
            <p>Time dependencies</p>
          </div>
          
          <ArrowRight className="arch-arrow" />
          
          <div className="arch-component">
            <div className="arch-icon">🎯</div>
            <h4>Multi-Head Output</h4>
            <p>1, 5, 10-year forecasts</p>
          </div>
        </div>
      </div>

      <div className="cta">
        <h2>Ready to Explore?</h2>
        <p>Dive deeper into the architecture, explore results, and try the interactive demo.</p>
        <div className="cta-actions">
          <Link to="/architecture" className="btn btn-primary">
            <Brain size={20} />
            Architecture Deep Dive
          </Link>
          <Link to="/results" className="btn btn-secondary">
            <BarChart3 size={20} />
            View Results
          </Link>
        </div>
      </div>
    </div>
  );
};

export default Home;

import React, { useState } from 'react';
import { Copy, Check, Play, Download, ExternalLink } from 'lucide-react';

const Implementation: React.FC = () => {
  const [copiedCode, setCopiedCode] = useState<string | null>(null);

  const copyToClipboard = (code: string, id: string) => {
    navigator.clipboard.writeText(code);
    setCopiedCode(id);
    setTimeout(() => setCopiedCode(null), 2000);
  };

  const codeSnippets = {
    'basic-training': `from src.training import rolling_origin_training
from src.data_ingest.merge_panel import load_all_data
from pathlib import Path

# Load data
panel_data = load_all_data(Path("data"))

# Train model
results = rolling_origin_training(
    panel_data=panel_data,
    config=config,
    train_years=list(range(2000, 2015)),
    val_years=list(range(2015, 2020)),
    test_years=list(range(2020, 2024))
)

print(f"Training completed. Final MASE: {results['final_mase']:.3f}")`,

    'model-creation': `from src.models.forecast_net import create_model
from src.features import FactorScaler
import torch

# Create model
model = create_model(config["model"], device="cpu")

# Load scaler
scaler = FactorScaler()
scaler.load_state("models/scaler.pkl")

# Generate forecast
input_tensor = torch.randn(1, 20, 8)  # [batch, years, factors]
predictions = model(input_tensor)

print(f"Forecasts: {predictions['point_forecasts']}")
print(f"Uncertainty: {predictions['quantiles']}")`,

    'data-preprocessing': `from src.features import FactorScaler, create_windows
from src.data_schema import CountryData
import pandas as pd

# Load raw data
df = pd.read_csv("data/synthetic_country_data.csv")
country_data = [CountryData(**row) for _, row in df.iterrows()]

# Create scaler and fit
scaler = FactorScaler()
scaler.fit(country_data)

# Transform data
scaled_data = scaler.transform(country_data)

# Create windows
windows = create_windows(
    scaled_data, 
    window_length=20, 
    horizons=[1, 5, 10]
)`,

    'evaluation': `from src.eval_loco import run_loco_evaluation
from src.metrics import calculate_all_metrics

# Run LOCO evaluation
loco_results = run_loco_evaluation(
    panel_data=panel_data,
    config=config,
    output_dir="results/loco"
)

# Calculate comprehensive metrics
metrics = calculate_all_metrics(
    predictions=loco_results['predictions'],
    targets=loco_results['targets'],
    quantiles=loco_results['quantiles']
)

print(f"LOCO MASE: {metrics['mase']:.3f}")
print(f"Spearman Correlation: {metrics['spearman']:.3f}")`,

    'streamlit-ui': `import streamlit as st
from src.ui_demo import create_app

# Launch Streamlit app
if __name__ == "__main__":
    st.set_page_config(
        page_title="Country Standing Forecast",
        page_icon="🌍",
        layout="wide"
    )
    
    create_app()`
  };

  const implementationSteps = [
    {
      title: "1. Environment Setup",
      description: "Set up the Python environment and install dependencies",
      code: `# Clone repository
git clone https://github.com/your-username/country-standing-forecast.git
cd country-standing-forecast

# Install dependencies
pip install -r requirements.txt

# Generate synthetic data
python make_synth_data.py`,
      type: "bash"
    },
    {
      title: "2. Data Loading & Preprocessing",
      description: "Load and preprocess the macro factor data",
      code: codeSnippets['data-preprocessing'],
      type: "python"
    },
    {
      title: "3. Model Training",
      description: "Train the neural network using rolling-origin walk-forward validation",
      code: codeSnippets['basic-training'],
      type: "python"
    },
    {
      title: "4. Model Evaluation",
      description: "Evaluate model performance using LOCO cross-validation",
      code: codeSnippets['evaluation'],
      type: "python"
    },
    {
      title: "5. Generate Forecasts",
      description: "Use trained model to generate country standing forecasts",
      code: codeSnippets['model-creation'],
      type: "python"
    },
    {
      title: "6. Interactive UI",
      description: "Launch the Streamlit web application for exploration",
      code: codeSnippets['streamlit-ui'],
      type: "python"
    }
  ];

  const projectStructure = [
    { path: "src/", description: "Core Python modules", files: ["data_schema.py", "features.py", "training.py", "metrics.py"] },
    { path: "src/models/", description: "Neural network implementations", files: ["factor_conv.py", "temporal_tcn.py", "forecast_net.py"] },
    { path: "src/data_ingest/", description: "Data loading modules", files: ["load_wdi.py", "load_wipo.py", "merge_panel.py"] },
    { path: "configs/", description: "Configuration files", files: ["base.yaml"] },
    { path: "notebooks/", description: "Jupyter tutorials", files: ["00_quickstart.ipynb"] },
    { path: "tests/", description: "Unit tests", files: ["test_core.py"] },
    { path: "docs/", description: "React documentation app", files: ["src/", "public/", "package.json"] }
  ];

  return (
    <div className="implementation">
      <div className="page-header">
        <h1>Implementation Guide</h1>
        <p>Step-by-step guide to implementing and using the country standing forecasting system.</p>
      </div>

      <div className="quick-start">
        <h2>Quick Start</h2>
        <div className="quick-start-grid">
          <div className="quick-start-card">
            <Play className="card-icon" />
            <h3>Run Locally</h3>
            <p>Get started with synthetic data in minutes</p>
            <code>python make_synth_data.py</code>
          </div>
          
          <div className="quick-start-card">
            <Download className="card-icon" />
            <h3>Install Dependencies</h3>
            <p>All required packages listed in requirements.txt</p>
            <code>pip install -r requirements.txt</code>
          </div>
          
          <div className="quick-start-card">
            <ExternalLink className="card-icon" />
            <h3>Launch UI</h3>
            <p>Interactive Streamlit application</p>
            <code>streamlit run src/ui_demo.py</code>
          </div>
        </div>
      </div>

      <div className="implementation-steps">
        <h2>Implementation Steps</h2>
        {implementationSteps.map((step, index) => (
          <div key={index} className="step-card">
            <div className="step-header">
              <div className="step-number">{index + 1}</div>
              <div className="step-title">
                <h3>{step.title}</h3>
                <p>{step.description}</p>
              </div>
            </div>
            
            <div className="code-block">
              <div className="code-header">
                <span className="code-language">{step.type}</span>
                <button 
                  className="copy-button"
                  onClick={() => copyToClipboard(step.code, `step-${index}`)}
                >
                  {copiedCode === `step-${index}` ? <Check size={16} /> : <Copy size={16} />}
                </button>
              </div>
              <pre><code>{step.code}</code></pre>
            </div>
          </div>
        ))}
      </div>

      <div className="project-structure">
        <h2>Project Structure</h2>
        <div className="structure-tree">
          {projectStructure.map((item, index) => (
            <div key={index} className="structure-item">
              <div className="structure-path">
                <strong>{item.path}</strong>
                <span className="structure-description">{item.description}</span>
              </div>
              <div className="structure-files">
                {item.files.map((file, fileIndex) => (
                  <span key={fileIndex} className="structure-file">{file}</span>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="configuration">
        <h2>Configuration</h2>
        <div className="config-section">
          <h3>Model Hyperparameters</h3>
          <div className="config-grid">
            <div className="config-item">
              <strong>Window Length:</strong> 20 years
            </div>
            <div className="config-item">
              <strong>Forecast Horizons:</strong> [1, 5, 10] years
            </div>
            <div className="config-item">
              <strong>Factor Convolution:</strong> kernel_size=3, out_channels=16
            </div>
            <div className="config-item">
              <strong>TCN Dilations:</strong> [1, 2, 4, 8]
            </div>
            <div className="config-item">
              <strong>Learning Rate:</strong> 0.001
            </div>
            <div className="config-item">
              <strong>Batch Size:</strong> 32
            </div>
          </div>
        </div>

        <div className="config-section">
          <h3>Data Settings</h3>
          <div className="config-grid">
            <div className="config-item">
              <strong>Scaling Method:</strong> Robust scaling (median, IQR)
            </div>
            <div className="config-item">
              <strong>Missing Data:</strong> KNN imputation
            </div>
            <div className="config-item">
              <strong>Validation:</strong> Rolling-origin walk-forward
            </div>
            <div className="config-item">
              <strong>Cross-validation:</strong> Leave-One-Country-Out
            </div>
          </div>
        </div>
      </div>

      <div className="troubleshooting">
        <h2>Troubleshooting</h2>
        <div className="troubleshooting-grid">
          <div className="troubleshooting-item">
            <h4>Import Errors</h4>
            <p><strong>Problem:</strong> ModuleNotFoundError when importing src modules</p>
            <p><strong>Solution:</strong> Ensure you're running from the project root directory</p>
          </div>
          
          <div className="troubleshooting-item">
            <h4>CUDA Issues</h4>
            <p><strong>Problem:</strong> CUDA out of memory or device not found</p>
            <p><strong>Solution:</strong> Set device="cpu" in config or reduce batch size</p>
          </div>
          
          <div className="troubleshooting-item">
            <h4>Data Loading</h4>
            <p><strong>Problem:</strong> Missing data files or incorrect format</p>
            <p><strong>Solution:</strong> Run make_synth_data.py to generate synthetic data</p>
          </div>
          
          <div className="troubleshooting-item">
            <h4>Memory Issues</h4>
            <p><strong>Problem:</strong> Out of memory during training</p>
            <p><strong>Solution:</strong> Reduce batch size or window length in config</p>
          </div>
        </div>
      </div>

      <div className="next-steps">
        <h2>Next Steps</h2>
        <div className="next-steps-grid">
          <div className="next-step-card">
            <h4>🔗 Integrate Real Data</h4>
            <p>Replace synthetic data with real macro factor data from World Bank, WIPO, and other sources.</p>
          </div>
          
          <div className="next-step-card">
            <h4>🎯 Model Improvements</h4>
            <p>Implement the architectural improvements discussed in the Architecture section.</p>
          </div>
          
          <div className="next-step-card">
            <h4>📊 Production Deployment</h4>
            <p>Deploy the model as a web service for real-time country standing forecasts.</p>
          </div>
          
          <div className="next-step-card">
            <h4>🔬 Research Extensions</h4>
            <p>Explore causal inference, external shock modeling, and country-specific adaptations.</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Implementation;

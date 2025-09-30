import React, { useState } from 'react';
import { ChevronDown, ChevronRight, Brain, Clock, Target, Shield, AlertTriangle } from 'lucide-react';

const Architecture: React.FC = () => {
  const [expandedSections, setExpandedSections] = useState<Set<string>>(new Set(['factor-conv']));

  const toggleSection = (section: string) => {
    const newExpanded = new Set(expandedSections);
    if (newExpanded.has(section)) {
      newExpanded.delete(section);
    } else {
      newExpanded.add(section);
    }
    setExpandedSections(newExpanded);
  };

  const architectureComponents = [
    {
      id: 'factor-conv',
      title: 'Factor-Wise Convolution',
      icon: Brain,
      problem: 'The 8 macro factors don\'t exist in isolation. Education affects innovation capacity, military spending influences trade relationships.',
      solution: 'Factor-wise 1D convolution across the factor dimension at each time step.',
      benefits: [
        'Local Interactions: Kernel size 2-3 captures immediate factor relationships',
        'Translation Invariance: Same relationships apply regardless of factor ordering',
        'Parameter Efficiency: Shared weights across time steps reduce overfitting',
        'Interpretability: Convolution patterns reveal which factor combinations matter most'
      ],
      code: `class FactorConv1D(nn.Module):
    def __init__(self, n_factors=8, out_channels=16, kernel_size=3):
        self.conv = nn.Conv1d(1, out_channels, kernel_size, padding="same")
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        # x: [batch_size, seq_len, n_factors]
        x_reshaped = x.view(batch_size * seq_len, 1, n_factors)
        conv_out = self.conv(x_reshaped)
        return self.activation(self.dropout(conv_out))`
    },
    {
      id: 'temporal-tcn',
      title: 'Temporal Convolutional Network',
      icon: Clock,
      problem: 'Country standing evolves over decades. A policy change in education today affects competitiveness 10 years later.',
      solution: 'Dilated temporal convolutions with dilations [1, 2, 4, 8] to capture multi-scale temporal patterns.',
      benefits: [
        'Parallel Processing: All time steps processed simultaneously (faster training)',
        'Long Memory: Dilated convolutions capture dependencies across 20+ years',
        'Stable Gradients: No vanishing gradient problem',
        'Causal Structure: Only past information influences predictions'
      ],
      code: `class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, 
                              dilation=dilation, padding=dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3,
                              dilation=dilation, padding=dilation)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.1)`
    },
    {
      id: 'attention',
      title: 'Attention Mechanism',
      icon: Target,
      problem: 'Not all historical periods are equally relevant. Crisis years (2008, 2020) may be more predictive than stable periods.',
      solution: 'Temporal attention over the TCN output to focus on critical periods.',
      benefits: [
        'Adaptive Weighting: Model learns which time periods are most predictive',
        'Crisis Sensitivity: Can focus on economic shocks, wars, or policy changes',
        'Interpretability: Attention weights reveal which historical periods matter most',
        'Flexibility: Different countries may have different critical periods'
      ],
      code: `class TemporalAttention(nn.Module):
    def __init__(self, in_channels, attention_dim=64):
        self.query = nn.Linear(in_channels, attention_dim)
        self.key = nn.Linear(in_channels, attention_dim)
        self.value = nn.Linear(in_channels, in_channels)
    
    def forward(self, x):
        Q, K, V = self.query(x), self.key(x), self.value(x)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / sqrt(attention_dim)
        attention_weights = F.softmax(scores, dim=-1)
        return torch.matmul(attention_weights, V)`
    },
    {
      id: 'quantile-regression',
      title: 'Quantile Regression',
      icon: Shield,
      problem: 'Country futures are inherently uncertain. Point predictions are insufficient for policy decisions.',
      solution: 'Quantile regression at [0.1, 0.5, 0.9] quantiles for uncertainty quantification.',
      benefits: [
        'Policy Planning: Decision-makers need confidence intervals, not just best guesses',
        'Risk Assessment: Lower quantiles show worst-case scenarios',
        'Robust Predictions: Median (0.5) is more robust than mean for skewed distributions',
        'Regulatory Compliance: Many applications require uncertainty estimates'
      ],
      code: `class QuantileHead(nn.Module):
    def __init__(self, in_channels, quantiles=[0.1, 0.5, 0.9]):
        self.quantile_heads = nn.ModuleList([
            nn.Linear(in_channels, 1) for _ in quantiles
        ])
    
    def forward(self, x):
        return torch.cat([head(x) for head in self.quantile_heads], dim=1)`
    }
  ];

  const limitations = [
    {
      title: 'Static Factor Set',
      description: 'Fixed 8 factors may miss emerging indicators (e.g., digital infrastructure, climate resilience)',
      impact: 'Medium',
      solution: 'Dynamic factor selection with attention-based weighting'
    },
    {
      title: 'Linear Interactions',
      description: 'Convolution captures local interactions but may miss complex non-linear relationships',
      impact: 'High',
      solution: 'Multi-scale factor convolution with attention mechanisms'
    },
    {
      title: 'No External Shocks',
      description: 'Model doesn\'t explicitly account for global events (pandemics, wars, technological disruptions)',
      impact: 'High',
      solution: 'Multi-modal architecture with global factor integration'
    },
    {
      title: 'Homogeneous Treatment',
      description: 'All countries treated equally despite different development stages',
      impact: 'Medium',
      solution: 'Hierarchical country-specific modeling'
    }
  ];

  return (
    <div className="architecture">
      <div className="page-header">
        <h1>Architecture Deep Dive</h1>
        <p>Understanding the design decisions behind our neural network architecture for country standing prediction.</p>
      </div>

      <div className="architecture-overview">
        <h2>Why This Architecture?</h2>
        <p>
          The choice of neural network architecture is driven by the unique challenges of predicting country futures using macro factors. 
          Each component addresses specific problems in time series forecasting and factor interaction modeling.
        </p>
      </div>

      <div className="components-section">
        <h2>Architecture Components</h2>
        {architectureComponents.map((component) => {
          const Icon = component.icon;
          const isExpanded = expandedSections.has(component.id);
          
          return (
            <div key={component.id} className="component-card">
              <div 
                className="component-header"
                onClick={() => toggleSection(component.id)}
              >
                <div className="component-title">
                  <Icon size={24} />
                  <h3>{component.title}</h3>
                </div>
                {isExpanded ? <ChevronDown size={20} /> : <ChevronRight size={20} />}
              </div>
              
              {isExpanded && (
                <div className="component-content">
                  <div className="problem-solution">
                    <div className="problem">
                      <h4>Problem</h4>
                      <p>{component.problem}</p>
                    </div>
                    <div className="solution">
                      <h4>Solution</h4>
                      <p>{component.solution}</p>
                    </div>
                  </div>
                  
                  <div className="benefits">
                    <h4>Benefits</h4>
                    <ul>
                      {component.benefits.map((benefit, index) => (
                        <li key={index}>{benefit}</li>
                      ))}
                    </ul>
                  </div>
                  
                  <div className="code-example">
                    <h4>Implementation</h4>
                    <pre><code>{component.code}</code></pre>
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>

      <div className="limitations-section">
        <h2>Current Limitations & Improvements</h2>
        <div className="limitations-grid">
          {limitations.map((limitation, index) => (
            <div key={index} className="limitation-card">
              <div className="limitation-header">
                <AlertTriangle size={20} />
                <h4>{limitation.title}</h4>
                <span className={`impact-badge impact-${limitation.impact.toLowerCase()}`}>
                  {limitation.impact} Impact
                </span>
              </div>
              <p>{limitation.description}</p>
              <div className="solution">
                <strong>Solution:</strong> {limitation.solution}
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="data-choices">
        <h2>Data Architecture Choices</h2>
        <div className="choice-cards">
          <div className="choice-card">
            <h4>Robust Scaling over Standard Scaling</h4>
            <p><strong>Why:</strong> Macro factors have outliers (e.g., military spending spikes during wars)</p>
            <p><strong>Benefit:</strong> More robust to extreme values, better for skewed distributions</p>
          </div>
          
          <div className="choice-card">
            <h4>KNN Imputation over Mean/Median</h4>
            <p><strong>Why:</strong> Missing data patterns may be informative</p>
            <p><strong>Benefit:</strong> Preserves data structure, more sophisticated than simple imputation</p>
          </div>
          
          <div className="choice-card">
            <h4>20-Year Windows</h4>
            <p><strong>Why:</strong> Balances sufficient history with computational efficiency</p>
            <p><strong>Benefit:</strong> Captures multiple business cycles while keeping model manageable</p>
          </div>
        </div>
      </div>

      <div className="alternatives">
        <h2>Why Not Alternative Architectures?</h2>
        <div className="alternatives-grid">
          <div className="alternative-card">
            <h4>Transformers</h4>
            <div className="pros-cons">
              <div className="pros">
                <strong>Pros:</strong> Excellent at capturing long-range dependencies
              </div>
              <div className="cons">
                <strong>Cons:</strong> Require much more data, computationally expensive, less interpretable
              </div>
            </div>
          </div>
          
          <div className="alternative-card">
            <h4>Graph Neural Networks</h4>
            <div className="pros-cons">
              <div className="pros">
                <strong>Pros:</strong> Could model country relationships and dependencies
              </div>
              <div className="cons">
                <strong>Cons:</strong> Requires country relationship data, adds complexity
              </div>
            </div>
          </div>
          
          <div className="alternative-card">
            <h4>Traditional Econometric Models</h4>
            <div className="pros-cons">
              <div className="pros">
                <strong>Pros:</strong> Highly interpretable, well-established
              </div>
              <div className="cons">
                <strong>Cons:</strong> Assume linear relationships, struggle with non-stationarity
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Architecture;

# DRL Coastal EWS - Streamlit Dashboard

## Overview

Interactive dashboard for training, evaluating, and deploying Deep Reinforcement Learning agents for coastal flood emergency warning systems.

## Features

### 1. 🏠 Home
- Quick overview of recent runs and available models
- Quick access buttons to all features
- System status and information

### 2. ⚙️ Configuration Hub
Interactive parameter configuration with 5 tabs:

**Training Parameters**
- Learning rate, batch size, episodes
- PPO hyperparameters (clip epsilon, GAE lambda, gamma)
- Loss coefficients (value, entropy, gradient clipping)

**Reward Function Designer**
- Lives saved weight (prioritize safety)
- Infrastructure damage penalty
- Economic cost weight
- Early warning bonus
- False alarm penalty
- Quick response bonus
- Multi-objective weight balancing (safety, economy, efficiency)

**Action Space**
- Configure available actions (evacuate, sandbags, flood gates, alerts)
- Set action costs
- Enable/disable action masking

**Environment**
- Grid size and population density
- Hazard parameters (storm intensity, spawn rate)
- Evacuation time settings

**Save & Export**
- Save configurations as YAML files
- Download configurations
- Preview full config

### 3. 🚀 Training Control Center
Real-time training with live metrics:

**Controls**
- Start/Pause/Resume/Stop training
- Auto-refresh for live updates
- Episode progress tracking

**Live Metrics Visualization**
- Episode rewards with moving average
- Lives saved/lost tracking
- Cumulative damage monitoring
- Policy loss, value loss, entropy
- False alarms and resource usage

**Export**
- Save checkpoints manually
- Download metrics as CSV
- Training statistics summary

### 4. 📊 Evaluation
Test trained models on synthetic scenarios:

**Features**
- Load any saved checkpoint
- Run N evaluation episodes
- Performance metrics dashboard
- Action frequency analysis
- Episode-by-episode comparison
- Distribution analysis
- Best/worst episode identification

**Export**
- Download results as JSON or CSV
- Save evaluation reports

### 5. 🎯 Deployment & Monitoring
Deploy models for live inference:

**Live Control**
- Step-by-step inference
- Auto-run episodes
- Reset environment
- View current state

**Real-time Monitoring**
- Live reward tracking
- Action distribution
- Lives at risk monitoring
- At-risk zones display with evacuation/protection status

**Performance Analytics**
- Total reward and steps
- Decision-making metrics
- Action effectiveness analysis
- Response time tracking

## Installation

```bash
cd /home/msano/Projects/DRL_Synthetic

# Install dependencies
pip3 install -r requirements.txt --break-system-packages
```

## Quick Start

### Option 1: Using the startup script
```bash
./run_dashboard.sh
```

### Option 2: Direct launch
```bash
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`

## Usage Workflow

### 1. Configure Your Experiment
1. Go to **Configuration Hub**
2. Load a base config or create custom settings
3. Adjust training parameters
4. Design reward function weights
5. Configure action space
6. Save your configuration

### 2. Train Your Agent
1. Navigate to **Training Control Center**
2. Select your saved configuration
3. Click "Start Training"
4. Monitor live metrics in real-time
5. Save checkpoints periodically

### 3. Evaluate Performance
1. Go to **Evaluation**
2. Load a trained checkpoint
3. Run evaluation episodes
4. Analyze performance metrics
5. Compare action strategies
6. Export results

### 4. Deploy for Inference
1. Navigate to **Deployment**
2. Select trained model
3. Deploy model
4. Run step-by-step or auto-run episodes
5. Monitor live performance
6. Analyze decision-making effectiveness

## Configuration Files

All configurations are stored in YAML format:

- `config/base.yaml` - Default base configuration
- `config/experiments/quick_test.yaml` - Fast testing config
- `config/experiments/reward_tuning.yaml` - Reward optimization
- `config/experiments/hyperparameter_sweep.yaml` - Parameter exploration

## Synthetic Data

The dashboard works entirely with **synthetic data** generated on-the-fly by the FOZ environment:

- Flood Operational Zones (FOZ) with realistic parameters
- Storm surge scenarios with varying intensity
- Population distribution and vulnerability
- Infrastructure and evacuation modeling

No external data files required - everything is generated procedurally.

## Training Runs

All training runs are saved in `runs/` directory with:
- Configuration snapshot
- Metrics history (JSON)
- Model checkpoints
- Logs and metadata

## Tips

### Performance Optimization
- Use smaller grid sizes (10-20) for quick testing
- Reduce episodes for initial experiments
- Enable GPU if available
- Use batch training for faster convergence

### Reward Function Tuning
- Start with high lives_saved_weight (10-20)
- Keep false_alarm_penalty moderate (-2 to -5)
- Balance multi-objective weights (safety > economy ≈ efficiency)
- Use early_warning_bonus to encourage proactive behavior

### Action Space Design
- Always keep "do_nothing" action (action 0)
- Set realistic costs for each action
- Enable action masking to prevent invalid actions
- Test action effectiveness in evaluation

### Training Monitoring
- Watch for reward convergence
- Lives lost should decrease over time
- Policy loss should stabilize
- High entropy early, decreasing later

## Troubleshooting

**Dashboard won't start**
```bash
# Check streamlit installation
python3 -c "import streamlit; print(streamlit.__version__)"

# Reinstall if needed
pip3 install streamlit --break-system-packages --upgrade
```

**Import errors**
```bash
# Ensure you're in project root
cd /home/msano/Projects/DRL_Synthetic

# Check Python path
python3 -c "import sys; print(sys.path)"
```

**Training crashes**
- Reduce batch_size if out of memory
- Lower max_steps_per_episode
- Check config for invalid values
- Review logs in runs/ directory

**No checkpoints found**
- Train a model first in Training Control
- Check runs/ directory exists
- Verify checkpoints saved in runs/*/checkpoints/

## Architecture

```
DRL_Synthetic/
├── app.py                          # Main dashboard entry
├── dashboard/
│   ├── pages/
│   │   ├── 1_Configuration.py      # Config editor
│   │   ├── 2_Training.py           # Training control
│   │   ├── 3_Evaluation.py         # Model evaluation
│   │   └── 4_Deployment.py         # Live inference
│   └── utils/
│       └── trainer.py              # Training wrapper
├── models/
│   └── ppo_agent.py                # PPO implementation
├── environments/
│   └── foz_environment.py          # FOZ environment
├── utils/
│   ├── config_manager.py           # Config handling
│   └── parameter_optimizer.py      # Hyperparameter search
└── config/                         # Configuration files
```

## Support

For issues or questions:
- Check this README
- Review configuration examples
- Examine training logs in runs/
- Test with quick_test.yaml configuration first

## Next Steps

1. **Run Quick Test**: Start with `config/experiments/quick_test.yaml`
2. **Tune Rewards**: Experiment with reward weights in Configuration Hub
3. **Hyperparameter Search**: Use parameter_optimizer.py for systematic search
4. **Compare Models**: Train multiple configurations and compare in Evaluation
5. **Deploy Best Model**: Use top-performing model in Deployment for inference

---

**Built for:** DRL Coastal Emergency Warning System
**Purpose:** Interactive training and deployment of flood response agents
**Data:** Synthetic FOZ scenarios (Venice, Gold Coast style environments)
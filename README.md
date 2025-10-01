# DRL Coastal Emergency Warning System

Deep Reinforcement Learning for Flood Operational Zone (FOZ) management in coastal areas.

## Overview

This project implements a DRL-based emergency warning and response system for coastal flooding events. It transitions from synthetic grid-based environments to real-world Flood Operational Zones (FOZ) for locations like Venice, Italy and Gold Coast, Australia.

## Project Structure

```
DRL_Synthetic/
├── config/                 # Location-specific configurations
│   ├── venice.yaml        # Venice FOZ configuration
│   └── goldcoast.yaml     # Gold Coast FOZ configuration
├── environments/          # Environment implementations
│   ├── base_environment.py
│   └── foz_environment.py # FOZ environment for real locations
├── models/                # Neural network models
│   └── attention.py       # Spatial attention mechanisms
├── utils/                 # Utility modules
│   ├── curriculum.py      # Curriculum learning scheduler
│   └── state_processor.py # State preprocessing
├── data/                  # Location data (populated at runtime)
│   ├── venice/
│   └── goldcoast/
└── notebooks/            # Jupyter notebooks for experiments

```

## Key Features

- **Flood Operational Zones (FOZ)**: Hydraulically coherent areas for real-world flood management
- **Multi-head Attention**: Spatial awareness of hazard clusters
- **Action Masking**: Ensures only feasible actions are taken
- **Curriculum Learning**: Progressive difficulty for stable training
- **Real-World Actions**: Location-specific interventions (MOSE barriers, evacuations, etc.)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/drl-coastal-ews.git
cd drl-coastal-ews

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Local Development

```python
from environments import FOZEnvironment
from utils import StateProcessor, CurriculumScheduler

# Create environment for Venice
env = FOZEnvironment('config/venice.yaml')

# Initialize state processor
processor = StateProcessor()

# Reset environment
state = env.reset()

# Take actions
action = 0  # Do nothing
next_state, reward, done, info = env.step(action)
```

### Training on Google Colab

1. Push your changes to GitHub:
```bash
git add .
git commit -m "Update FOZ implementation"
git push origin main
```

2. In Google Colab:
```python
# Clone repository
!git clone https://github.com/yourusername/drl-coastal-ews.git
%cd drl-coastal-ews

# Install requirements
!pip install -r requirements.txt

# Import and train
from environments import FOZEnvironment
from train import train_ppo_agent

# Train agent
agent = train_ppo_agent('config/venice.yaml', episodes=10000)
```

## Configuration

Each location has its own YAML configuration file defining:
- Zone definitions and connectivity
- Available actions and costs
- Population distribution
- Infrastructure vulnerability
- Training hyperparameters

Example from `config/venice.yaml`:
```yaml
location:
  name: "Venice"
  country: "Italy"

zones:
  count: 30
  definition_file: "data/venice/foz_definitions.csv"

actions:
  available:
    - id: 1
      name: "activate_mose_partial"
      cost: 50
      zones_affected: ["inlet_zones"]
```

## Development Workflow

1. **Code Locally**: Develop and test modules using `test_local.py`
2. **Push to GitHub**: Version control all changes
3. **Train on Colab**: Use free GPU for intensive training
4. **Evaluate Results**: Analyze performance metrics

## Testing

```bash
# Test project structure
python3 test_structure.py

# Test functionality (requires dependencies)
python3 test_local.py
```

## Original Colab Implementation

The original monolithic Colab notebook (`drl_ews_20082025_v2.py`) contains:
- Complete PPO implementation
- Vectorized environments
- Comprehensive diagnostics
- Research paper visualizations

## Citation

If you use this code in your research, please cite:
```
@software{drl_coastal_ews,
  title={Deep Reinforcement Learning for Coastal Emergency Warning Systems},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/drl-coastal-ews}
}
```

## License

MIT License - See LICENSE file for details

## Contact

For questions or collaboration: your.email@example.com
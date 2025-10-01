# CLAUDE.md - Project Context for AI Assistants

> **Last Updated:** 2025-10-01
> **Project:** DRL Coastal Emergency Warning System
> **Status:** Active Development - Dashboard Operational
> **Owner:** Marcello Sano

---

## 🎯 Project Overview

### What This Project Does

This is a **Deep Reinforcement Learning (DRL) system** for coastal flood emergency management. It trains AI agents to make optimal decisions during coastal flooding events to:
- Minimize loss of life
- Reduce infrastructure damage
- Optimize resource allocation (evacuations, barriers, alerts)
- Learn proactive warning strategies

### Current Capabilities

✅ **Operational:**
- Interactive Streamlit dashboard for configuration, training, evaluation, deployment
- PPO (Proximal Policy Optimization) agent implementation
- FOZ (Flood Operational Zone) environment with synthetic data generation
- Hierarchical YAML configuration system
- Real-time training monitoring with live metrics
- Model evaluation and comparison tools
- Inference deployment interface

🚧 **In Progress:**
- Google Colab integration for GPU training
- Curriculum learning implementation
- Multi-agent support

### Key Technologies

- **Framework:** PyTorch (neural networks), Custom PPO implementation
- **Dashboard:** Streamlit (interactive UI), Plotly (visualizations)
- **Environment:** Custom FOZ environment (synthetic data generation)
- **Configuration:** YAML-based hierarchical configs
- **Deployment:** Local (WSL2), planned Google Colab integration

---

## 📁 Project Structure

```
DRL_Synthetic/
├── app.py                              # Main Streamlit dashboard entry
├── dashboard/
│   ├── pages/
│   │   ├── 1_Configuration.py          # Parameter editor, reward designer
│   │   ├── 2_Training.py               # Training control, live metrics
│   │   ├── 3_Evaluation.py             # Model testing, comparison
│   │   └── 4_Deployment.py             # Live inference, monitoring
│   └── utils/
│       └── trainer.py                  # TrainingSession, TrainingManager
├── models/
│   ├── ppo_agent.py                    # PPO implementation (actor-critic)
│   └── attention.py                    # Attention mechanisms
├── environments/
│   ├── base_environment.py             # Base environment interface
│   └── foz_environment.py              # FOZ environment (synthetic data)
├── utils/
│   ├── config_manager.py               # YAML config handling, inheritance
│   ├── parameter_optimizer.py          # Hyperparameter search
│   ├── curriculum.py                   # Curriculum learning
│   └── state_processor.py              # State preprocessing
├── config/
│   ├── base.yaml                       # Base configuration template
│   ├── venice.yaml                     # Venice-specific config
│   ├── goldcoast.yaml                  # Gold Coast-specific config
│   └── experiments/
│       ├── quick_test.yaml             # Fast testing (100 episodes)
│       ├── reward_tuning.yaml          # Reward optimization focus
│       └── hyperparameter_sweep.yaml   # Parameter exploration
├── runs/                               # Training outputs (gitignored)
│   └── [run_name]/
│       ├── config.yaml                 # Config snapshot
│       ├── metrics.json                # Episode metrics
│       └── checkpoints/*.pt            # Model weights
├── requirements.txt                    # Python dependencies
├── DASHBOARD_README.md                 # Dashboard user guide
├── DASHBOARD_SETUP_COMPLETE.md         # Setup documentation
└── CLAUDE.md                           # This file

Gitignored:
- runs/ (training outputs, checkpoints)
- *.pt, *.pth (model weights)
- __pycache__/ (Python cache)
- logs/, results/, diagnostics/
```

---

## 🏗️ Architecture

### Component Relationships

```
┌─────────────────────────────────────────────────────────┐
│ STREAMLIT DASHBOARD (Main Thread)                      │
│ - User interface                                        │
│ - Configuration editor                                  │
│ - Metrics visualization                                 │
│ - Model evaluation                                      │
└─────────────────────────────────────────────────────────┘
                        ↕ (shared state)
┌─────────────────────────────────────────────────────────┐
│ TRAINING SESSION (Background Thread)                   │
│                                                         │
│  ┌─────────────────┐    ┌──────────────────┐          │
│  │ PPO Agent       │ ←→ │ FOZ Environment  │          │
│  │ - Policy Network│    │ - Synthetic Data │          │
│  │ - Critic Network│    │ - Flood Sim      │          │
│  │ - Memory Buffer │    │ - Reward Calc    │          │
│  └─────────────────┘    └──────────────────┘          │
│           ↓                       ↓                     │
│  ┌─────────────────────────────────────────┐          │
│  │ Training Loop                            │          │
│  │ 1. Reset environment (new scenario)     │          │
│  │ 2. Agent selects actions                │          │
│  │ 3. Environment simulates                │          │
│  │ 4. Store transitions                    │          │
│  │ 5. PPO update (backprop)                │          │
│  │ 6. Save metrics                         │          │
│  └─────────────────────────────────────────┘          │
│                                                         │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ PERSISTENT STORAGE                                      │
│ - runs/[name]/checkpoints/*.pt (model weights)         │
│ - runs/[name]/metrics.json (training history)          │
│ - runs/[name]/config.yaml (configuration snapshot)     │
└─────────────────────────────────────────────────────────┘
```

### Data Flow: Training Episode

```
1. env.reset()
   ↓ Generates synthetic scenario:
   - 30 FOZ zones (population, elevation, vulnerability)
   - Random storm (intensity, arrival time, affected zones)
   - Reset resources to 100

2. Agent observes state (164 features):
   - Time, resources, hazard info
   - Top 20 zones by risk (8 features each)

3. Agent selects action:
   policy_network(state) → action_logits
   sample from distribution → action
   Actions: 0=nothing, 1=evacuate, 2=sandbags, 3=gates, 4=alert

4. env.step(action):
   - Apply action effects (evacuate zones, deploy protections)
   - Simulate water propagation
   - Calculate casualties in non-evacuated zones
   - Compute damage to infrastructure
   - Calculate reward

5. Reward computation:
   reward = (
       -100 × lives_lost +
       -0.01 × damage +
       +50 if zero casualties else 0 +
       early_warning_bonus +
       -action_cost
   )

6. Store transition in memory buffer

7. After episode: PPO update
   - Compute advantages (GAE)
   - For 4 epochs:
     - Policy loss (clipped objective)
     - Value loss (critic)
     - Entropy bonus
     - Backpropagation
     - Gradient descent

8. Update dashboard metrics
```

---

## 🚀 Development Setup

### Prerequisites

- Python 3.12+
- WSL2 (Windows Subsystem for Linux) or Linux
- 16GB RAM recommended
- Optional: NVIDIA GPU with CUDA for local training

### Installation

```bash
cd /home/msano/Projects/DRL_Synthetic

# Install dependencies
pip3 install -r requirements.txt --break-system-packages

# Verify setup
python3 verify_dashboard.py

# Launch dashboard
./run_dashboard.sh
# Or:
python3 -m streamlit run app.py
```

### Accessing Dashboard

**Local (WSL2):** http://localhost:8501
**Network:** http://172.28.44.249:8501

Dashboard pages:
- 🏠 Home - Overview and quick actions
- ⚙️ Configuration - Parameter editor, reward designer
- 🚀 Training - Launch training, live metrics
- 📊 Evaluation - Test models, compare performance
- 🎯 Deployment - Live inference, monitoring

---

## ⚙️ Configuration System

### YAML Hierarchy

```yaml
# config/base.yaml (template)
experiment:
  name: "coastal_drl_base"
  seed: 42

training:
  learning_rate: 0.0003
  episodes: 1000
  batch_size: 64

reward:
  lives_saved_weight: 10.0
  early_warning_bonus: 5.0
  false_alarm_penalty: -2.0

# config/experiments/quick_test.yaml (inherits from base)
extends: "base.yaml"

experiment:
  name: "quick_test"

training:
  episodes: 100  # Override
  batch_size: 32
```

### Runtime Overrides

```python
from utils.config_manager import ConfigManager

manager = ConfigManager()
config = manager.load_config(
    "config/base.yaml",
    overrides={
        'training.learning_rate': 0.001,
        'reward.lives_saved_weight': 15.0
    }
)
```

---

## 🎓 Training Workflows

### Local Training (CPU/GPU)

**Use for:**
- Quick testing (10-100 episodes)
- Reward function tuning
- Configuration validation
- Short experiments

**Hardware:** Intel i7-1265U (12 cores), 16GB RAM, Integrated GPU
**Speed:** ~2-3 hours for 1000 episodes (CPU)

**Workflow:**
```bash
# 1. Configure in dashboard
Dashboard → Configuration → Adjust parameters → Save

# 2. Start training
Dashboard → Training → Select config → Start Training

# 3. Monitor live
Auto-refresh updates every 2 seconds
Watch: episode rewards, lives lost, policy loss

# 4. Evaluate
Dashboard → Evaluation → Load checkpoint → Run tests
```

### Google Colab Training (GPU) - RECOMMENDED

**Use for:**
- Production training (1000+ episodes)
- Hyperparameter sweeps
- Multiple parallel experiments
- Faster iteration

**Hardware:** Free Tesla T4 GPU (12GB VRAM)
**Speed:** ~15-30 minutes for 1000 episodes (GPU)
**Cost:** Free (limited) or $10/mo (Colab Pro)

**Workflow:**
```bash
# 1. Configure locally
Dashboard → Configuration → Save config

# 2. Push to GitHub
git add config/experiments/my_experiment.yaml
git commit -m "New experiment"
git push

# 3. Train on Colab
Open Colab notebook → Run All
(Auto: clone, install, train, save)

# 4. Download results
files.download('checkpoints/final.pt')

# 5. Evaluate locally
Dashboard → Evaluation → Load checkpoint
```

### Training Location Decision

```
Use Local When:
├─ Testing configurations (< 100 episodes)
├─ Debugging code
├─ Reward function experiments
└─ Evaluation and inference

Use Colab When:
├─ Production training (1000+ episodes)
├─ Hyperparameter sweeps
├─ Need GPU acceleration
└─ Multiple experiments in parallel
```

---

## 📊 Key Metrics

### During Training (Live)

- **Episode Reward:** Total reward per episode (should increase)
- **Lives Lost:** Casualties per episode (should decrease to ~0)
- **Policy Loss:** How much policy is changing (should stabilize)
- **Value Loss:** Critic prediction error (should decrease)
- **Entropy:** Exploration vs exploitation (high early, decrease later)

### Success Indicators

✅ **Good Training:**
- Episode rewards trending upward
- Lives lost decreasing to near-zero
- Policy loss stabilizing (not spiking)
- Moving average shows clear improvement
- Agent takes proactive actions early

❌ **Problem Signs:**
- Rewards stay flat or decrease
- Lives lost stays high
- Policy loss explodes (LR too high)
- Agent only takes "do nothing"

---

## 🔧 Common Tasks

### 1. Create New Experiment Configuration

```bash
# Via dashboard
Dashboard → Configuration
→ Load base.yaml
→ Adjust parameters
→ Save as "my_experiment.yaml"

# Or manually
cp config/base.yaml config/experiments/my_experiment.yaml
# Edit my_experiment.yaml
```

### 2. Train a Model

```bash
# Via dashboard
Dashboard → Training
→ Select config
→ Click "Start Training"
→ Enable "Auto Refresh"

# Via Python
from dashboard.utils.trainer import TrainingSession

session = TrainingSession("config/experiments/my_experiment.yaml")
session.train()
```

### 3. Evaluate Trained Model

```bash
Dashboard → Evaluation
→ Select checkpoint (runs/*/checkpoints/*.pt)
→ Click "Load Model"
→ Click "Run Evaluation" (10 episodes)
→ View metrics, action analysis, distributions
```

### 4. Compare Multiple Models

```bash
Dashboard → Evaluation
→ Load checkpoint 1 → Run evaluation → Export CSV
→ Load checkpoint 2 → Run evaluation → Export CSV
→ Compare CSVs in spreadsheet or notebook
```

### 5. Deploy for Live Inference

```bash
Dashboard → Deployment
→ Select trained checkpoint
→ Click "Deploy Model"
→ Use "Step" or "Run Episode"
→ Monitor real-time performance
```

### 6. Tune Reward Function

```bash
Dashboard → Configuration → Reward Function tab

Adjust sliders:
- Lives Saved Weight: 10-20 (higher = prioritize safety)
- Early Warning Bonus: 5-10 (encourage proactive action)
- False Alarm Penalty: -2 to -5 (discourage unnecessary warnings)
- Multi-objective weights: safety/economy/efficiency (sum to 1.0)

Save → Train → Evaluate → Compare
```

---

## 🎯 Technical Decisions

### Why Synthetic Data?

**Decision:** Generate flood scenarios on-the-fly rather than use historical data

**Rationale:**
- ✅ Infinite training scenarios (no overfitting to limited dataset)
- ✅ No data collection/cleaning required
- ✅ Easy to test different hazard parameters
- ✅ Controlled experiments (reproducible scenarios)
- ✅ Privacy (no real population data needed)

**Trade-offs:**
- ⚠️ Synthetic may not capture all real-world complexity
- ⚠️ Requires domain knowledge to configure realistic parameters
- ✅ Can be augmented with real data later

### Why Streamlit Dashboard?

**Decision:** Build interactive dashboard instead of CLI scripts

**Rationale:**
- ✅ Visual reward function designer (intuitive tuning)
- ✅ Real-time training monitoring (live metrics)
- ✅ No code required for parameter changes
- ✅ Easy model comparison (charts, tables)
- ✅ Research-friendly (iterate quickly)

**Trade-offs:**
- ⚠️ Not suitable for production deployment (use API instead)
- ⚠️ Single-user (not multi-tenant)
- ✅ Perfect for research/experimentation phase

### Why PPO?

**Decision:** Use PPO instead of other RL algorithms (DQN, A3C, SAC)

**Rationale:**
- ✅ Stable training (clipped objective)
- ✅ Sample efficient (on-policy but reuses data)
- ✅ Works well for discrete action spaces
- ✅ Good for high-dimensional state spaces
- ✅ Industry standard for robotics/games

### Why Background Threading?

**Decision:** Run training in separate thread from dashboard

**Rationale:**
- ✅ Dashboard stays responsive during training
- ✅ Live metrics updates (auto-refresh)
- ✅ Pause/resume/stop controls
- ✅ Better user experience

**Implementation:**
```python
# Main thread: Streamlit UI
while True:
    render_dashboard()
    display_metrics()

# Background thread: Training
Thread(target=session.train).start()
```

---

## 🚧 Known Limitations

1. **Training Duration:**
   - Local CPU: Slow for large runs (>1000 episodes)
   - Solution: Use Google Colab GPU

2. **Session Persistence:**
   - Dashboard requires server running
   - Training stops if server crashes
   - Solution: Colab notebooks with auto-save

3. **Multi-User:**
   - Dashboard is single-user
   - No concurrent training sessions
   - Solution: Deploy separate instances or use Colab

4. **Real Data Integration:**
   - Currently only synthetic data
   - Real FOZ data requires custom loaders
   - Solution: Add data loaders in future

5. **Deployment:**
   - No production API (dashboard only)
   - Manual model loading for inference
   - Solution: Build FastAPI service layer

---

## 🔮 Future Enhancements

### Short-term (1-3 months)
- [ ] Google Colab training notebook
- [ ] Hyperparameter auto-tuning (Optuna integration)
- [ ] A/B testing framework for reward functions
- [ ] Curriculum learning implementation
- [ ] Multi-location comparison (Venice vs Gold Coast)

### Medium-term (3-6 months)
- [ ] Real FOZ data integration (historical floods)
- [ ] Multi-agent support (coordinated responses)
- [ ] Explainable AI (why agent took action)
- [ ] Transfer learning (pre-trained models)
- [ ] Ensemble models (multiple agents voting)

### Long-term (6-12 months)
- [ ] Production API deployment (FastAPI)
- [ ] Real-time data integration (weather feeds)
- [ ] Model serving infrastructure (Vertex AI)
- [ ] Web-based public demo
- [ ] Research paper publication

---

## 📚 Important Files

### Core Implementation

- `models/ppo_agent.py` - PPO algorithm, neural networks, training loop
- `environments/foz_environment.py` - Flood simulation, reward calculation
- `dashboard/utils/trainer.py` - Training session management, metrics tracking
- `utils/config_manager.py` - YAML config loading, inheritance, overrides

### Configuration

- `config/base.yaml` - Default configuration template (comprehensive)
- `config/experiments/*.yaml` - Experiment-specific configs
- `requirements.txt` - Python dependencies

### Documentation

- `README.md` - Project overview, installation, usage
- `DASHBOARD_README.md` - Dashboard feature guide
- `DASHBOARD_SETUP_COMPLETE.md` - Setup completion summary
- `CLAUDE.md` - This file (AI assistant context)

### Utilities

- `verify_dashboard.py` - Check installation and dependencies
- `run_dashboard.sh` - Startup script for dashboard
- `test_config_system.py` - Test config inheritance

---

## 🎓 Learning Resources

### Understanding the Code

**Start here:**
1. `environments/foz_environment.py` - See how floods are simulated
2. `models/ppo_agent.py` - Understand PPO training loop
3. `config/base.yaml` - All configurable parameters
4. `dashboard/pages/2_Training.py` - Training visualization

**Key Concepts:**
- **FOZ (Flood Operational Zone):** Geographic unit for flood management
- **PPO (Proximal Policy Optimization):** RL algorithm for policy gradient
- **GAE (Generalized Advantage Estimation):** Better advantage calculation
- **Action Masking:** Prevent invalid actions (resource constraints)
- **Curriculum Learning:** Progressive difficulty increase

### External References

- **PPO Paper:** https://arxiv.org/abs/1707.06347
- **Streamlit Docs:** https://docs.streamlit.io
- **PyTorch RL Tutorial:** https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

---

## 🤝 Working with This Project

### For Claude (or other AI assistants):

**When asked to modify code:**
1. Read relevant files first (use Read tool)
2. Understand existing patterns and style
3. Test changes don't break dashboard
4. Update this CLAUDE.md if architecture changes

**When asked about training:**
1. Check hardware (local CPU vs Colab GPU recommendation)
2. Suggest appropriate config (quick_test vs production)
3. Explain expected training time
4. Point to relevant dashboard page

**When asked about configurations:**
1. Start with config/base.yaml
2. Use inheritance for experiments
3. Validate reward weights make sense
4. Test with quick_test.yaml first

**When asked about errors:**
1. Check if dashboard is running (port 8501)
2. Verify dependencies installed
3. Look for missing imports
4. Check gitignored files (runs/, checkpoints/)

### Common User Questions

**Q: How do I train a model?**
A: Dashboard → Configuration → adjust params → Training → Start

**Q: Why is training slow?**
A: Local CPU is slow. Use Google Colab GPU (15-30 mins vs 2-3 hours)

**Q: How do I tune the reward function?**
A: Dashboard → Configuration → Reward Function tab → adjust sliders

**Q: Where are checkpoints saved?**
A: `runs/[experiment_name]/checkpoints/*.pt` (gitignored)

**Q: Can I train multiple models at once?**
A: Not on dashboard. Use Google Colab for parallel experiments

**Q: How do I compare models?**
A: Dashboard → Evaluation → Load each checkpoint → Export CSVs → Compare

---

## 🔐 Security & Privacy

- ✅ No external data collection
- ✅ No API keys required
- ✅ All data stays local (except Colab uploads)
- ✅ Synthetic data only (no real population data)
- ⚠️ Dashboard runs on localhost (not secured for public access)

---

## 📞 Support

**For setup issues:**
1. Run `python3 verify_dashboard.py`
2. Check `DASHBOARD_README.md` troubleshooting section
3. Review this file for architecture questions

**For training issues:**
1. Check metrics in dashboard (episode rewards, lives lost)
2. Try `config/experiments/quick_test.yaml` first
3. Verify GPU availability if using Colab

**For configuration questions:**
1. Review `config/base.yaml` for all parameters
2. Check reward function weights (lives_saved should be highest)
3. Test with small episodes first

---

## 📝 Version History

**2025-10-01:** Initial CLAUDE.md created
- Dashboard operational (4 pages)
- PPO agent implemented
- FOZ environment with synthetic data
- Configuration system complete
- Local training working
- Google Colab integration planned

---

**End of CLAUDE.md**
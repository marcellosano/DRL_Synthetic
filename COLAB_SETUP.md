# Google Colab Setup Guide

## 🎯 Quick Start

Train your DRL models with **free GPU acceleration** on Google Colab!

**Speed Comparison:**
- Local (CPU): 2-3 hours for 1000 episodes
- Colab (Tesla T4 GPU): 15-30 minutes for 1000 episodes

**Cost:** FREE (with limits) or $10/month (Colab Pro)

---

## 📋 Prerequisites

1. **GitHub Account** - To host your code
2. **Google Account** - For Colab access
3. **Repository pushed to GitHub** - Your DRL_Synthetic code

---

## 🚀 Setup Steps

### Step 1: Push Code to GitHub

```bash
# If you haven't already created a GitHub repo:
# 1. Go to https://github.com/new
# 2. Create repository (e.g., "DRL_Synthetic")
# 3. Copy the remote URL

# Add remote and push
cd /home/msano/Projects/DRL_Synthetic
git remote add origin https://github.com/YOUR_USERNAME/DRL_Synthetic.git
git branch -M main
git push -u origin main
```

**Important:** Replace `YOUR_USERNAME` with your actual GitHub username!

### Step 2: Upload Colab Notebook to GitHub

The notebook is already created at `colab_training.ipynb`

```bash
# It's already committed! Just push if you added remote:
git push origin main
```

### Step 3: Open in Google Colab

**Two Ways:**

**Option A: Direct Link (Easiest)**
1. Go to: https://colab.research.google.com/
2. Click "GitHub" tab
3. Enter: `YOUR_USERNAME/DRL_Synthetic`
4. Click on `colab_training.ipynb`

**Option B: From GitHub**
1. Go to your GitHub repo
2. Navigate to `colab_training.ipynb`
3. Click "Open in Colab" badge (if you add one, see below)

### Step 4: Update Notebook with Your GitHub URL

**In the Colab notebook, find this line in Cell 2:**
```python
!git clone https://github.com/YOUR_USERNAME/DRL_Synthetic.git
```

**Replace `YOUR_USERNAME` with your actual GitHub username**

Save the notebook:
- File → Save a copy in Drive (first time)
- Or File → Save (if already saved)

### Step 5: Enable GPU

1. In Colab: Runtime → Change runtime type
2. Hardware accelerator → **GPU**
3. GPU type → **T4** (free) or V100/A100 (Colab Pro)
4. Click **Save**

### Step 6: Run Training!

**Execute cells in order:**
1. **Cell 1:** Setup & Installation (~2 minutes)
2. **Cell 2:** Configuration Selection
3. **Cell 3:** Training (~15-30 minutes depending on config)
4. **Cell 4:** Visualize Results
5. **Cell 5:** Download Model

---

## 📊 Workflow

### Complete Workflow: Local ↔ Colab

```
┌────────────────────────────────────┐
│ LOCAL (Your Machine)               │
│                                    │
│ 1. Configure experiment            │
│    Dashboard → Configuration       │
│    Adjust reward weights, params   │
│    Save as experiment_v5.yaml      │
└────────────────────────────────────┘
              ↓ git push
┌────────────────────────────────────┐
│ GITHUB                             │
│                                    │
│ 2. Code synced to cloud            │
│    config/experiments/             │
│         experiment_v5.yaml         │
└────────────────────────────────────┘
              ↓ git clone (in Colab)
┌────────────────────────────────────┐
│ GOOGLE COLAB                       │
│                                    │
│ 3. Train with GPU                  │
│    Select: experiment_v5           │
│    Run → Train → Download          │
│    Time: 15-30 minutes            │
└────────────────────────────────────┘
              ↓ download .zip
┌────────────────────────────────────┐
│ LOCAL (Your Machine)               │
│                                    │
│ 4. Evaluate & Deploy               │
│    Dashboard → Evaluation          │
│    Load model.pt                   │
│    Compare performance             │
│    Deploy if satisfied             │
└────────────────────────────────────┘
```

---

## 🎓 Training on Colab

### Select Configuration

In Cell 2, change this line:
```python
selected_config = 'quick_test'  # Change to train different configs
```

**Available configs:**
- `'quick_test'` - Fast test (100 episodes, ~3-5 minutes)
- `'reward_tuning'` - Reward optimization (1000 episodes, ~15-20 minutes)
- `'hyperparameter_sweep'` - Parameter exploration (defined in config)
- `'base'` - Full training (1000 episodes)

### Monitor Training

Training progress updates every 10 episodes showing:
- Episode number and progress bar
- Current reward
- Lives lost
- Policy and value loss

**Example output:**
```
🎓 Training Progress: 45.0%
   Episode: 450/1000
   Reward: 245.67
   Lives Lost: 2
   Policy Loss: 0.0234
   Value Loss: 12.3456

███████████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░
```

### After Training

Cell 4 shows 4 charts:
- Episode Rewards (with moving average)
- Lives Lost over time
- Policy Loss
- Cumulative Damage

Cell 5 downloads a .zip file containing:
- `model.pt` - Trained model checkpoint
- `metrics.json` - All episode metrics
- `config.yaml` - Configuration used
- `summary.json` - Training summary

---

## 💾 Using Downloaded Models

### Extract and Load in Dashboard

1. **Extract the zip file**
   ```bash
   cd ~/Downloads
   unzip colab_results_*.zip
   ```

2. **Copy to project**
   ```bash
   cp colab_results_*/model.pt /home/msano/Projects/DRL_Synthetic/runs/colab_run/checkpoints/
   ```

3. **Load in dashboard**
   - Dashboard → Evaluation
   - Select checkpoint: `runs/colab_run/checkpoints/model.pt`
   - Click "Load Model"
   - Click "Run Evaluation"

---

## 🔄 Batch Training (Advanced)

Train multiple configurations in one session:

**In Cell 6 (optional), uncomment and modify:**
```python
configs_to_train = [
    'config/experiments/quick_test.yaml',
    'config/experiments/reward_tuning.yaml',
    'config/experiments/baseline_v2.yaml',
]

for config_path in configs_to_train:
    session = TrainingSession(config_path)
    session.train()
    # Results automatically saved
```

This trains all configs sequentially. Download all results at the end.

---

## ⚙️ Colab Settings

### Runtime Limits

**Free Tier:**
- 12 hour maximum session
- ~50 hours GPU/week
- Lower priority (may wait for GPU)
- Session disconnects if idle

**Colab Pro ($10/month):**
- 24 hour sessions
- Unlimited GPU access (within reasonable use)
- Priority GPU allocation
- Background execution

### Best Practices

1. **Save checkpoints frequently** (already configured every 100 episodes)
2. **Download results immediately** after training
3. **Don't leave idle** - sessions disconnect after inactivity
4. **Close when done** - free up resources for others

### Reconnecting

If disconnected:
1. Runtime → Connect
2. Re-run setup cell (Cell 1)
3. Training progress is lost, but checkpoints may be saved
4. Check `runs/` directory for partial results

---

## 🐛 Troubleshooting

### "No GPU available"

**Solution:**
- Runtime → Change runtime type → GPU
- Restart runtime if needed
- Try different time of day (less congestion)

### "git clone failed"

**Solution:**
- Check GitHub URL is correct
- Make sure repo is public (or add credentials)
- Manually visit GitHub to verify repo exists

### "Import errors"

**Solution:**
- Re-run Cell 1 (setup)
- Check all dependencies installed
- Restart runtime: Runtime → Restart runtime

### "Out of memory"

**Solution:**
- Reduce `batch_size` in config (e.g., 32 → 16)
- Use smaller network (reduce `hidden_dims`)
- Free up memory: Runtime → Restart runtime

### "Session disconnected"

**Prevention:**
- Colab Pro for 24h sessions
- Download results immediately
- Keep browser tab active

**Recovery:**
- Reconnect and re-run setup
- Check if checkpoints saved in `runs/`

---

## 📈 Performance Tips

### Faster Training

1. **Use appropriate config:**
   - `quick_test.yaml` for testing (100 episodes)
   - Production configs for serious runs

2. **Optimize batch size:**
   - Larger batch = faster but more VRAM
   - T4 GPU: 64-128 batch size works well

3. **Reduce unnecessary logging:**
   - Log every 10 episodes instead of every episode

### Cost Optimization

**Free Tier Strategy:**
- Use for production runs only (not testing)
- Test configurations locally first
- Train during off-peak hours
- Close session when done

**When to Upgrade to Pro:**
- Training > 12 hours
- Need multiple runs per day
- Require priority GPU access
- Want background execution

---

## 📚 Additional Resources

### Colab Tips

- **Keyboard Shortcuts:** Ctrl+M H (show shortcuts)
- **GPU Stats:** `!nvidia-smi` to check GPU usage
- **File Browser:** Click folder icon on left
- **Mount Google Drive:** To save results automatically

### Example: Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')

# Save checkpoints to Drive
checkpoint_path = '/content/drive/MyDrive/DRL_Models/model.pt'
```

---

## 🎯 Quick Reference

### Common Tasks

| Task | Command |
|------|---------|
| **Check GPU** | `!nvidia-smi` |
| **Restart Runtime** | Runtime → Restart runtime |
| **Change GPU Type** | Runtime → Change runtime type |
| **Download File** | `files.download('filename')` |
| **Upload File** | Click upload button in file browser |
| **Clear Output** | Click ⋮ menu → Clear output |

### Training Commands

```python
# Quick test (5 minutes)
selected_config = 'quick_test'

# Production run (20 minutes)
selected_config = 'reward_tuning'

# Custom config (if you created one)
selected_config = 'my_experiment'  # Must exist in config/experiments/
```

---

## ✅ Checklist

Before starting:
- [ ] Code pushed to GitHub
- [ ] GitHub URL updated in notebook
- [ ] GPU enabled in Colab
- [ ] Configuration selected
- [ ] Ready to wait 15-30 minutes

After training:
- [ ] Results visualized
- [ ] Model downloaded
- [ ] Metrics exported
- [ ] Session closed (if done)

---

## 🤝 Need Help?

1. Check [CLAUDE.md](CLAUDE.md) for project architecture
2. Review [DASHBOARD_README.md](DASHBOARD_README.md) for dashboard usage
3. Check Colab's built-in tutorials: Help → Search notebooks

---

**Happy Training! 🚀🌊**
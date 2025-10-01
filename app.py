"""
DRL Coastal Emergency Warning System - Streamlit Dashboard
Main application entry point
"""

import streamlit as st
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Page configuration
st.set_page_config(
    page_title="DRL Coastal EWS",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'training_manager' not in st.session_state:
    from dashboard.utils.trainer import TrainingManager
    st.session_state.training_manager = TrainingManager()

if 'config_manager' not in st.session_state:
    from utils.config_manager import ConfigManager
    st.session_state.config_manager = ConfigManager()

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .status-running {
        color: #28a745;
        font-weight: bold;
    }
    .status-stopped {
        color: #dc3545;
        font-weight: bold;
    }
    .status-paused {
        color: #ffc107;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.markdown("# 🌊 DRL Coastal EWS")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "⚙️ Configuration", "🚀 Training", "📊 Evaluation", "🎯 Deployment"],
    index=0
)

st.sidebar.markdown("---")

# Display active session info
if st.session_state.training_manager.active_session:
    session = st.session_state.training_manager.get_active_session()
    if session:
        st.sidebar.markdown("### 📍 Active Session")
        st.sidebar.markdown(f"**Run:** {session.run.name}")
        st.sidebar.markdown(f"**Episode:** {session.current_episode}/{session.total_episodes}")

        if session.is_training:
            st.sidebar.markdown('<p class="status-running">● Training</p>', unsafe_allow_html=True)
        elif session.is_paused:
            st.sidebar.markdown('<p class="status-paused">● Paused</p>', unsafe_allow_html=True)
        else:
            st.sidebar.markdown('<p class="status-stopped">● Stopped</p>', unsafe_allow_html=True)

st.sidebar.markdown("---")

# System info
st.sidebar.markdown("### 💻 System Info")
import torch
device = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
st.sidebar.markdown(f"**Device:** {device}")
st.sidebar.markdown(f"**PyTorch:** {torch.__version__}")

# Main content routing
if page == "🏠 Home":
    st.markdown('<p class="main-header">🌊 DRL Coastal Emergency Warning System</p>', unsafe_allow_html=True)

    st.markdown("""
    ### Welcome to the Training Dashboard

    This dashboard provides complete control over your Deep Reinforcement Learning system
    for coastal flood emergency management.

    #### 🎯 Features

    - **⚙️ Configuration Hub**: Set up training parameters, reward functions, and action spaces
    - **🚀 Training Control**: Launch and monitor training runs with real-time metrics
    - **📊 Evaluation**: Test models on synthetic scenarios and analyze performance
    - **🎯 Deployment**: Deploy trained models for live inference

    #### 🚀 Quick Start

    1. Go to **Configuration** to set up your experiment parameters
    2. Navigate to **Training** to start a training run
    3. Monitor live metrics and adjust as needed
    4. Evaluate your trained model in the **Evaluation** page
    5. Deploy for inference in the **Deployment** page

    ---
    """)

    # Display recent runs
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 📈 Recent Runs")
        runs_dir = Path("runs")
        if runs_dir.exists():
            recent_runs = sorted(runs_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True)[:5]
            if recent_runs:
                for run_dir in recent_runs:
                    st.markdown(f"- {run_dir.name}")
            else:
                st.info("No runs yet")
        else:
            st.info("No runs directory found")

    with col2:
        st.markdown("### 📝 Available Configs")
        config_dir = Path("config/experiments")
        if config_dir.exists():
            configs = sorted(config_dir.glob("*.yaml"))
            for config in configs:
                st.markdown(f"- {config.stem}")
        else:
            st.warning("No experiment configs found")

    with col3:
        st.markdown("### 💾 Saved Models")
        models_dir = Path("models")
        if models_dir.exists():
            model_files = list(models_dir.glob("**/*.pt"))
            st.markdown(f"Total: {len(model_files)} checkpoints")
        else:
            st.info("No models saved yet")

    st.markdown("---")

    # Quick actions
    st.markdown("### ⚡ Quick Actions")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🎯 Start Quick Test", use_container_width=True):
            st.switch_page("dashboard/pages/2_Training.py")

    with col2:
        if st.button("📊 View Last Run", use_container_width=True):
            st.switch_page("dashboard/pages/3_Evaluation.py")

    with col3:
        if st.button("⚙️ Configure New Run", use_container_width=True):
            st.switch_page("dashboard/pages/1_Configuration.py")

elif page == "⚙️ Configuration":
    # Import configuration page
    import importlib.util
    spec = importlib.util.spec_from_file_location("config_page", "dashboard/pages/1_Configuration.py")
    config_page = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_page)

elif page == "🚀 Training":
    # Import training page
    import importlib.util
    spec = importlib.util.spec_from_file_location("training_page", "dashboard/pages/2_Training.py")
    training_page = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(training_page)

elif page == "📊 Evaluation":
    # Import evaluation page
    import importlib.util
    spec = importlib.util.spec_from_file_location("eval_page", "dashboard/pages/3_Evaluation.py")
    eval_page = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(eval_page)

elif page == "🎯 Deployment":
    # Import deployment page
    import importlib.util
    spec = importlib.util.spec_from_file_location("deploy_page", "dashboard/pages/4_Deployment.py")
    deploy_page = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(deploy_page)
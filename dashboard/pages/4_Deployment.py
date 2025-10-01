"""
Deployment & Monitoring Page
Deploy models for live inference and monitor real-time performance
"""

import streamlit as st
import plotly.graph_objects as go
import sys
from pathlib import Path
import numpy as np
import time
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from environments.foz_environment import FOZEnvironment
from models.ppo_agent import PPOAgent
from utils.config_manager import ConfigManager

st.set_page_config(page_title="Deployment", page_icon="🎯", layout="wide")

# Initialize
if 'config_manager' not in st.session_state:
    st.session_state.config_manager = ConfigManager()

if 'deployed_agent' not in st.session_state:
    st.session_state.deployed_agent = None

if 'deployed_env' not in st.session_state:
    st.session_state.deployed_env = None

if 'inference_running' not in st.session_state:
    st.session_state.inference_running = False

if 'inference_history' not in st.session_state:
    st.session_state.inference_history = {
        'timestamps': [],
        'states': [],
        'actions': [],
        'rewards': [],
        'lives_at_risk': [],
        'water_levels': []
    }

if 'current_episode_data' not in st.session_state:
    st.session_state.current_episode_data = None

st.markdown("# 🎯 Deployment & Live Inference")
st.markdown("Deploy trained models and monitor real-time performance on synthetic data streams")
st.markdown("---")

# Deployment setup
st.markdown("### 🚀 Model Deployment")

col1, col2, col3 = st.columns([2, 2, 1])

with col1:
    # Find all checkpoint files
    runs_dir = Path("runs")
    checkpoint_paths = []

    if runs_dir.exists():
        for run_dir in runs_dir.iterdir():
            if run_dir.is_dir():
                checkpoint_dir = run_dir / "checkpoints"
                if checkpoint_dir.exists():
                    for ckpt in checkpoint_dir.glob("*.pt"):
                        checkpoint_paths.append(str(ckpt))

    if checkpoint_paths:
        selected_checkpoint = st.selectbox(
            "Select Model to Deploy",
            ["None"] + checkpoint_paths
        )
    else:
        st.warning("No checkpoints found")
        selected_checkpoint = "None"

with col2:
    config_files = list(Path("config").rglob("*.yaml"))
    config_options = ["config/base.yaml"] + [str(f) for f in config_files if f.name != "base.yaml"]

    deploy_config = st.selectbox(
        "Environment Configuration",
        config_options,
        index=0
    )

with col3:
    st.markdown("###")  # Spacing
    auto_run = st.checkbox("Auto-run Episodes", value=False)

# Deploy/Undeploy buttons
col1, col2 = st.columns(2)

with col1:
    if st.button("🚀 Deploy Model", use_container_width=True):
        if selected_checkpoint != "None":
            try:
                # Load config
                config = st.session_state.config_manager.load_config(deploy_config)

                # Create environment
                env = FOZEnvironment(deploy_config)

                # Create agent
                agent = PPOAgent(
                    state_dim=env.state_dim,
                    action_dim=env.action_dim,
                    config=config
                )

                # Load checkpoint
                agent.load(selected_checkpoint)

                st.session_state.deployed_agent = agent
                st.session_state.deployed_env = env
                st.session_state.deploy_config = config
                st.session_state.deployed_checkpoint = selected_checkpoint

                # Reset environment
                state = env.reset()
                st.session_state.current_episode_data = {
                    'state': state,
                    'step': 0,
                    'done': False,
                    'total_reward': 0,
                    'actions_taken': []
                }

                st.success(f"✅ Model deployed: {Path(selected_checkpoint).name}")

            except Exception as e:
                st.error(f"❌ Error deploying model: {e}")
        else:
            st.warning("Please select a checkpoint")

with col2:
    if st.button("🔴 Undeploy Model", use_container_width=True):
        st.session_state.deployed_agent = None
        st.session_state.deployed_env = None
        st.session_state.inference_running = False
        st.session_state.current_episode_data = None
        st.info("Model undeployed")

st.markdown("---")

# Deployment status
if st.session_state.deployed_agent:
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Status", "🟢 Deployed")

    with col2:
        model_name = Path(st.session_state.deployed_checkpoint).stem
        st.metric("Model", model_name)

    with col3:
        if st.session_state.current_episode_data:
            step = st.session_state.current_episode_data['step']
            st.metric("Current Step", f"{step}")
        else:
            st.metric("Current Step", "N/A")

    with col4:
        if st.session_state.current_episode_data:
            total_reward = st.session_state.current_episode_data['total_reward']
            st.metric("Episode Reward", f"{total_reward:.2f}")
        else:
            st.metric("Episode Reward", "N/A")

    st.markdown("---")

    # Inference controls
    tab1, tab2, tab3 = st.tabs([
        "🎮 Live Control",
        "📊 Real-time Monitoring",
        "📈 Performance Analytics"
    ])

    # ========================================================================
    # TAB 1: Live Control
    # ========================================================================
    with tab1:
        st.markdown("### 🎮 Inference Control")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if st.button("▶️ Step", use_container_width=True):
                if st.session_state.current_episode_data and not st.session_state.current_episode_data['done']:
                    agent = st.session_state.deployed_agent
                    env = st.session_state.deployed_env
                    episode_data = st.session_state.current_episode_data

                    # Get action
                    action_mask = env.get_valid_actions()
                    action, _, _ = agent.select_action(episode_data['state'], action_mask)

                    # Take step
                    next_state, reward, done, info = env.step(action)

                    # Update episode data
                    episode_data['state'] = next_state
                    episode_data['step'] += 1
                    episode_data['done'] = done
                    episode_data['total_reward'] += reward
                    episode_data['actions_taken'].append(action)

                    # Record history
                    st.session_state.inference_history['timestamps'].append(datetime.now())
                    st.session_state.inference_history['states'].append(next_state)
                    st.session_state.inference_history['actions'].append(action)
                    st.session_state.inference_history['rewards'].append(reward)
                    st.session_state.inference_history['lives_at_risk'].append(info.get('lives_lost', 0))

                    if done:
                        st.info("Episode completed!")

                    st.rerun()

        with col2:
            if st.button("⏭️ Run Episode", use_container_width=True):
                if st.session_state.current_episode_data:
                    agent = st.session_state.deployed_agent
                    env = st.session_state.deployed_env

                    with st.spinner("Running episode..."):
                        while not st.session_state.current_episode_data['done']:
                            episode_data = st.session_state.current_episode_data

                            action_mask = env.get_valid_actions()
                            action, _, _ = agent.select_action(episode_data['state'], action_mask)

                            next_state, reward, done, info = env.step(action)

                            episode_data['state'] = next_state
                            episode_data['step'] += 1
                            episode_data['done'] = done
                            episode_data['total_reward'] += reward
                            episode_data['actions_taken'].append(action)

                    st.success(f"Episode completed! Total reward: {episode_data['total_reward']:.2f}")
                    st.rerun()

        with col3:
            if st.button("🔄 Reset Episode", use_container_width=True):
                env = st.session_state.deployed_env
                state = env.reset()

                st.session_state.current_episode_data = {
                    'state': state,
                    'step': 0,
                    'done': False,
                    'total_reward': 0,
                    'actions_taken': []
                }

                st.info("Episode reset")
                st.rerun()

        with col4:
            if st.button("🗑️ Clear History", use_container_width=True):
                st.session_state.inference_history = {
                    'timestamps': [],
                    'states': [],
                    'actions': [],
                    'rewards': [],
                    'lives_at_risk': [],
                    'water_levels': []
                }
                st.info("History cleared")
                st.rerun()

        st.markdown("---")

        # Current environment state
        if st.session_state.current_episode_data:
            st.markdown("### 🌍 Current Environment State")

            env = st.session_state.deployed_env
            episode_data = st.session_state.current_episode_data

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Time Step", f"{env.time_step}/{env.max_steps}")

            with col2:
                st.metric("Resources", f"{env.resources}/100")

            with col3:
                st.metric("Lives Lost", f"{env.cumulative_lives_lost}")

            with col4:
                st.metric("Cumulative Damage", f"${env.cumulative_damage:.0f}")

            # Hazard information
            if env.current_hazard:
                st.markdown("#### 🌊 Current Hazard")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Intensity", f"{env.current_hazard['intensity']:.2f}m")

                with col2:
                    arrival = max(0, env.current_hazard['arrival_time'])
                    st.metric("Arrival Time", f"{arrival} steps")

                with col3:
                    affected = len(env.current_hazard['affected_zones'])
                    st.metric("Affected Zones", f"{affected}")

            # At-risk zones
            st.markdown("#### ⚠️ Top At-Risk Zones")

            at_risk_zones = sorted(
                env.zones.values(),
                key=lambda z: z.current_water_level * z.population,
                reverse=True
            )[:5]

            if at_risk_zones[0].current_water_level > 0:
                risk_data = []
                for zone in at_risk_zones:
                    if zone.current_water_level > 0:
                        risk_data.append({
                            'Zone': zone.name,
                            'Population': zone.population,
                            'Water Level': f"{zone.current_water_level:.2f}m",
                            'Evacuated': '✅' if zone.is_evacuated else '❌',
                            'Protected': '✅' if zone.has_protection else '❌'
                        })

                if risk_data:
                    import pandas as pd
                    st.dataframe(pd.DataFrame(risk_data), use_container_width=True, hide_index=True)
            else:
                st.info("No zones currently at risk")

    # ========================================================================
    # TAB 2: Real-time Monitoring
    # ========================================================================
    with tab2:
        st.markdown("### 📊 Live Performance Metrics")

        if st.session_state.inference_history['actions']:
            history = st.session_state.inference_history

            # Rewards over time
            fig_rewards = go.Figure()
            fig_rewards.add_trace(go.Scatter(
                x=list(range(len(history['rewards']))),
                y=history['rewards'],
                mode='lines+markers',
                name='Reward',
                line=dict(color='blue', width=2)
            ))
            fig_rewards.update_layout(
                title='Rewards Over Time',
                xaxis_title='Step',
                yaxis_title='Reward',
                height=300
            )
            st.plotly_chart(fig_rewards, use_container_width=True)

            # Actions frequency
            import pandas as pd
            action_counts = pd.Series(history['actions']).value_counts().sort_index()

            action_names = {
                0: "Do Nothing",
                1: "Evacuate Zone",
                2: "Sandbag Deployment",
                3: "Activate Flood Gates",
                4: "Emergency Alert"
            }

            fig_actions = go.Figure()
            fig_actions.add_trace(go.Bar(
                x=[action_names.get(i, f"Action {i}") for i in action_counts.index],
                y=action_counts.values,
                marker_color='teal'
            ))
            fig_actions.update_layout(
                title='Action Distribution',
                xaxis_title='Action',
                yaxis_title='Count',
                height=300
            )
            st.plotly_chart(fig_actions, use_container_width=True)

            # Lives at risk tracking
            if history['lives_at_risk']:
                fig_lives = go.Figure()
                fig_lives.add_trace(go.Scatter(
                    x=list(range(len(history['lives_at_risk']))),
                    y=history['lives_at_risk'],
                    mode='lines+markers',
                    name='Lives Lost',
                    line=dict(color='red', width=2),
                    fill='tozeroy'
                ))
                fig_lives.update_layout(
                    title='Lives Lost Over Time',
                    xaxis_title='Step',
                    yaxis_title='Lives Lost',
                    height=300
                )
                st.plotly_chart(fig_lives, use_container_width=True)

        else:
            st.info("No inference history yet. Run some steps to see live metrics!")

    # ========================================================================
    # TAB 3: Performance Analytics
    # ========================================================================
    with tab3:
        st.markdown("### 📈 Deployment Analytics")

        if st.session_state.inference_history['actions']:
            history = st.session_state.inference_history

            col1, col2, col3 = st.columns(3)

            with col1:
                total_reward = sum(history['rewards'])
                st.metric("Total Reward", f"{total_reward:.2f}")

            with col2:
                total_steps = len(history['actions'])
                st.metric("Total Steps", f"{total_steps}")

            with col3:
                if history['lives_at_risk']:
                    total_lives_lost = sum(history['lives_at_risk'])
                    st.metric("Total Lives Lost", f"{total_lives_lost}")

            # Response time analysis
            st.markdown("#### ⏱️ Decision Making")

            action_sequence = history['actions']
            non_zero_actions = [i for i, a in enumerate(action_sequence) if a != 0]

            if non_zero_actions:
                first_action_step = non_zero_actions[0]
                st.metric("First Action Taken at Step", f"{first_action_step}")

                action_frequency = len(non_zero_actions) / len(action_sequence) * 100
                st.metric("Action Rate", f"{action_frequency:.1f}%")

            # Action effectiveness
            st.markdown("#### 🎯 Action Effectiveness")

            # Calculate reward per action type
            import pandas as pd
            action_rewards = {}
            for action, reward in zip(history['actions'], history['rewards']):
                if action not in action_rewards:
                    action_rewards[action] = []
                action_rewards[action].append(reward)

            action_names = {
                0: "Do Nothing",
                1: "Evacuate Zone",
                2: "Sandbag Deployment",
                3: "Activate Flood Gates",
                4: "Emergency Alert"
            }

            effectiveness_data = []
            for action_id, rewards in action_rewards.items():
                effectiveness_data.append({
                    'Action': action_names.get(action_id, f"Action {action_id}"),
                    'Count': len(rewards),
                    'Mean Reward': np.mean(rewards),
                    'Total Reward': sum(rewards)
                })

            df_effectiveness = pd.DataFrame(effectiveness_data)
            st.dataframe(df_effectiveness, use_container_width=True, hide_index=True)

        else:
            st.info("No data yet. Deploy model and run inference to see analytics!")

else:
    st.info("👆 Deploy a model to start live inference")

# Auto-run episodes
if auto_run and st.session_state.deployed_agent and st.session_state.current_episode_data:
    if not st.session_state.current_episode_data['done']:
        time.sleep(0.5)
        st.rerun()
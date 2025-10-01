"""
Training Control Center Page
Launch training, monitor real-time metrics, and control training runs
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path
import time
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dashboard.utils.trainer import TrainingManager, TrainingSession

st.set_page_config(page_title="Training Control", page_icon="🚀", layout="wide")

# Initialize
if 'training_manager' not in st.session_state:
    st.session_state.training_manager = TrainingManager()

if 'auto_refresh' not in st.session_state:
    st.session_state.auto_refresh = False

st.markdown("# 🚀 Training Control Center")
st.markdown("Launch and monitor training runs with real-time metrics")
st.markdown("---")

# Training setup section
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    config_files = list(Path("config").rglob("*.yaml"))
    config_options = ["config/base.yaml"] + [str(f) for f in config_files if f.name != "base.yaml"]

    # Find quick_test.yaml index in config_options
    default_index = 0
    if "config/experiments/quick_test.yaml" in config_options:
        default_index = config_options.index("config/experiments/quick_test.yaml")

    selected_config = st.selectbox(
        "Select Configuration",
        config_options,
        index=default_index
    )

with col2:
    run_name = st.text_input(
        "Run Name (optional)",
        value="",
        placeholder="auto-generated"
    )

with col3:
    st.markdown("###")  # Spacing
    use_gpu = st.checkbox("Use GPU", value=True)

# Control buttons
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    if st.button("🚀 Start Training", use_container_width=True):
        try:
            # Create new session
            session = st.session_state.training_manager.create_session(
                selected_config,
                run_name if run_name else None
            )

            # Start training in background
            session.start_async()

            st.success(f"✅ Training started: {session.run.name}")
            st.session_state.auto_refresh = True

        except Exception as e:
            st.error(f"❌ Error starting training: {e}")

with col2:
    if st.button("⏸️ Pause", use_container_width=True):
        session = st.session_state.training_manager.get_active_session()
        if session and session.is_training:
            session.pause()
            st.info("Training paused")
        else:
            st.warning("No active training session")

with col3:
    if st.button("▶️ Resume", use_container_width=True):
        session = st.session_state.training_manager.get_active_session()
        if session and session.is_paused:
            session.resume()
            st.info("Training resumed")
        else:
            st.warning("No paused session to resume")

with col4:
    if st.button("⏹️ Stop", use_container_width=True):
        session = st.session_state.training_manager.get_active_session()
        if session:
            session.stop()
            st.session_state.auto_refresh = False
            st.warning("Training stopped")
        else:
            st.warning("No active session")

with col5:
    auto_refresh = st.checkbox("Auto Refresh", value=st.session_state.auto_refresh)
    st.session_state.auto_refresh = auto_refresh

st.markdown("---")

# Get active session
session = st.session_state.training_manager.get_active_session()

if session:
    # Status header
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        status_text = "🟢 Running" if session.is_training else "🔴 Stopped"
        if session.is_paused:
            status_text = "🟡 Paused"
        st.metric("Status", status_text)

    with col2:
        progress = (session.current_episode / session.total_episodes) * 100 if session.total_episodes > 0 else 0
        st.metric("Progress", f"{progress:.1f}%")

    with col3:
        st.metric("Episode", f"{session.current_episode}/{session.total_episodes}")

    with col4:
        if session.current_metrics:
            last_reward = session.current_metrics.get('reward', 0)
            st.metric("Last Reward", f"{last_reward:.2f}")
        else:
            st.metric("Last Reward", "N/A")

    with col5:
        if session.current_metrics:
            lives_lost = session.current_metrics.get('lives_lost', 0)
            st.metric("Lives Lost", f"{lives_lost}")
        else:
            st.metric("Lives Lost", "N/A")

    # Progress bar
    st.progress(progress / 100)

    st.markdown("---")

    # Metrics visualization
    if session.metrics['episode_rewards']:
        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 Rewards & Performance",
            "💀 Lives & Safety",
            "🎯 Policy Metrics",
            "📊 Statistics"
        ])

        # ====================================================================
        # TAB 1: Rewards and Performance
        # ====================================================================
        with tab1:
            # Prepare data
            episodes = list(range(len(session.metrics['episode_rewards'])))
            rewards = session.metrics['episode_rewards']

            # Calculate moving average
            window = min(10, len(rewards))
            if len(rewards) >= window:
                rewards_ma = pd.Series(rewards).rolling(window=window).mean().tolist()
            else:
                rewards_ma = rewards

            # Create subplot
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Episode Rewards', 'Episode Length', 'Cumulative Reward', 'Resources Used'),
                vertical_spacing=0.12,
                horizontal_spacing=0.1
            )

            # Episode rewards
            fig.add_trace(
                go.Scatter(x=episodes, y=rewards, mode='lines', name='Reward',
                          line=dict(color='lightblue', width=1), opacity=0.5),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=episodes, y=rewards_ma, mode='lines', name='MA(10)',
                          line=dict(color='blue', width=2)),
                row=1, col=1
            )

            # Episode length
            if session.metrics['episode_lengths']:
                fig.add_trace(
                    go.Scatter(x=episodes, y=session.metrics['episode_lengths'],
                              mode='lines', name='Length', line=dict(color='green')),
                    row=1, col=2
                )

            # Cumulative reward
            cumulative_rewards = np.cumsum(rewards)
            fig.add_trace(
                go.Scatter(x=episodes, y=cumulative_rewards,
                          mode='lines', name='Cumulative', line=dict(color='purple')),
                row=2, col=1
            )

            # Resources used
            if session.metrics['resources_used']:
                fig.add_trace(
                    go.Scatter(x=episodes, y=session.metrics['resources_used'],
                              mode='lines', name='Resources', line=dict(color='orange')),
                    row=2, col=2
                )

            fig.update_layout(height=600, showlegend=True)
            fig.update_xaxes(title_text="Episode", row=2, col=1)
            fig.update_xaxes(title_text="Episode", row=2, col=2)

            st.plotly_chart(fig, use_container_width=True)

        # ====================================================================
        # TAB 2: Lives and Safety Metrics
        # ====================================================================
        with tab2:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Lives Lost per Episode', 'Lives Saved', 'Cumulative Damage', 'False Alarms'),
                vertical_spacing=0.12,
                horizontal_spacing=0.1
            )

            # Lives lost
            if session.metrics['lives_lost']:
                fig.add_trace(
                    go.Scatter(x=episodes, y=session.metrics['lives_lost'],
                              mode='lines+markers', name='Lives Lost',
                              line=dict(color='red', width=2)),
                    row=1, col=1
                )

            # Lives saved
            if session.metrics['lives_saved']:
                fig.add_trace(
                    go.Scatter(x=episodes, y=session.metrics['lives_saved'],
                              mode='lines+markers', name='Lives Saved',
                              line=dict(color='green', width=2)),
                    row=1, col=2
                )

            # Cumulative damage
            if session.metrics['cumulative_damage']:
                fig.add_trace(
                    go.Scatter(x=episodes, y=session.metrics['cumulative_damage'],
                              mode='lines', name='Damage',
                              line=dict(color='orange', width=2)),
                    row=2, col=1
                )

            # False alarms
            if session.metrics['false_alarms']:
                fig.add_trace(
                    go.Scatter(x=episodes, y=session.metrics['false_alarms'],
                              mode='lines+markers', name='False Alarms',
                              line=dict(color='yellow', width=2)),
                    row=2, col=2
                )

            fig.update_layout(height=600, showlegend=True)
            fig.update_xaxes(title_text="Episode", row=2, col=1)
            fig.update_xaxes(title_text="Episode", row=2, col=2)

            st.plotly_chart(fig, use_container_width=True)

        # ====================================================================
        # TAB 3: Policy Metrics
        # ====================================================================
        with tab3:
            fig = make_subplots(
                rows=1, cols=3,
                subplot_titles=('Policy Loss', 'Value Loss', 'Entropy'),
                horizontal_spacing=0.1
            )

            # Policy loss
            if session.metrics['policy_loss']:
                fig.add_trace(
                    go.Scatter(x=episodes, y=session.metrics['policy_loss'],
                              mode='lines', name='Policy Loss',
                              line=dict(color='red', width=2)),
                    row=1, col=1
                )

            # Value loss
            if session.metrics['value_loss']:
                fig.add_trace(
                    go.Scatter(x=episodes, y=session.metrics['value_loss'],
                              mode='lines', name='Value Loss',
                              line=dict(color='blue', width=2)),
                    row=1, col=2
                )

            # Entropy
            if session.metrics['entropy']:
                fig.add_trace(
                    go.Scatter(x=episodes, y=session.metrics['entropy'],
                              mode='lines', name='Entropy',
                              line=dict(color='green', width=2)),
                    row=1, col=3
                )

            fig.update_layout(height=400, showlegend=False)
            fig.update_xaxes(title_text="Episode", row=1, col=1)
            fig.update_xaxes(title_text="Episode", row=1, col=2)
            fig.update_xaxes(title_text="Episode", row=1, col=3)

            st.plotly_chart(fig, use_container_width=True)

        # ====================================================================
        # TAB 4: Statistics
        # ====================================================================
        with tab4:
            st.markdown("### 📊 Training Statistics")

            summary = session.get_metrics_summary()

            if summary:
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("#### Rewards")
                    if 'episode_rewards' in summary:
                        df_rewards = pd.DataFrame({
                            'Metric': ['Mean', 'Std', 'Min', 'Max', 'Last'],
                            'Value': [
                                f"{summary['episode_rewards']['mean']:.2f}",
                                f"{summary['episode_rewards']['std']:.2f}",
                                f"{summary['episode_rewards']['min']:.2f}",
                                f"{summary['episode_rewards']['max']:.2f}",
                                f"{summary['episode_rewards']['last']:.2f}"
                            ]
                        })
                        st.dataframe(df_rewards, use_container_width=True, hide_index=True)

                    st.markdown("#### Lives Lost")
                    if 'lives_lost' in summary:
                        df_lives = pd.DataFrame({
                            'Metric': ['Mean', 'Std', 'Min', 'Max', 'Last'],
                            'Value': [
                                f"{summary['lives_lost']['mean']:.2f}",
                                f"{summary['lives_lost']['std']:.2f}",
                                f"{summary['lives_lost']['min']:.0f}",
                                f"{summary['lives_lost']['max']:.0f}",
                                f"{summary['lives_lost']['last']:.0f}"
                            ]
                        })
                        st.dataframe(df_lives, use_container_width=True, hide_index=True)

                with col2:
                    st.markdown("#### Damage")
                    if 'cumulative_damage' in summary:
                        df_damage = pd.DataFrame({
                            'Metric': ['Mean', 'Std', 'Min', 'Max', 'Last'],
                            'Value': [
                                f"${summary['cumulative_damage']['mean']:.0f}",
                                f"${summary['cumulative_damage']['std']:.0f}",
                                f"${summary['cumulative_damage']['min']:.0f}",
                                f"${summary['cumulative_damage']['max']:.0f}",
                                f"${summary['cumulative_damage']['last']:.0f}"
                            ]
                        })
                        st.dataframe(df_damage, use_container_width=True, hide_index=True)

                    st.markdown("#### Policy Loss")
                    if 'policy_loss' in summary:
                        df_policy = pd.DataFrame({
                            'Metric': ['Mean', 'Std', 'Min', 'Max', 'Last'],
                            'Value': [
                                f"{summary['policy_loss']['mean']:.4f}",
                                f"{summary['policy_loss']['std']:.4f}",
                                f"{summary['policy_loss']['min']:.4f}",
                                f"{summary['policy_loss']['max']:.4f}",
                                f"{summary['policy_loss']['last']:.4f}"
                            ]
                        })
                        st.dataframe(df_policy, use_container_width=True, hide_index=True)

            # Export metrics
            st.markdown("---")
            st.markdown("### 💾 Export Metrics")

            col1, col2 = st.columns(2)

            with col1:
                if st.button("💾 Save Checkpoint", use_container_width=True):
                    session.save_checkpoint(f"manual_episode_{session.current_episode}")
                    st.success("Checkpoint saved!")

            with col2:
                # Export to CSV
                if st.button("📥 Download Metrics CSV", use_container_width=True):
                    df = pd.DataFrame(session.metrics)
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"{session.run.name}_metrics.csv",
                        mime="text/csv"
                    )

    else:
        st.info("No metrics yet. Start training to see real-time data!")

else:
    st.info("👆 No active training session. Start a new training run above.")

# Auto-refresh
if st.session_state.auto_refresh and session and session.is_training:
    time.sleep(2)
    st.rerun()
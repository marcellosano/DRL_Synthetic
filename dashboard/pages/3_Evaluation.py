"""
Testing & Evaluation Page
Run inference, evaluate models, and compare performance
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from environments.foz_environment import FOZEnvironment
from models.ppo_agent import PPOAgent
from utils.config_manager import ConfigManager

st.set_page_config(page_title="Evaluation", page_icon="📊", layout="wide")

# Initialize
if 'config_manager' not in st.session_state:
    st.session_state.config_manager = ConfigManager()

if 'eval_results' not in st.session_state:
    st.session_state.eval_results = None

if 'loaded_agent' not in st.session_state:
    st.session_state.loaded_agent = None

st.markdown("# 📊 Testing & Evaluation")
st.markdown("Evaluate trained models on synthetic test scenarios")
st.markdown("---")

# Model selection
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
            "Select Model Checkpoint",
            ["None"] + checkpoint_paths
        )
    else:
        st.warning("No checkpoints found. Train a model first.")
        selected_checkpoint = "None"

with col2:
    config_files = list(Path("config").rglob("*.yaml"))
    config_options = ["config/base.yaml"] + [str(f) for f in config_files if f.name != "base.yaml"]

    eval_config = st.selectbox(
        "Evaluation Configuration",
        config_options,
        index=0
    )

with col3:
    num_eval_episodes = st.number_input(
        "Episodes",
        min_value=1,
        max_value=100,
        value=10,
        step=1
    )

# Load model button
col1, col2 = st.columns([1, 3])

with col1:
    if st.button("📂 Load Model", use_container_width=True):
        if selected_checkpoint != "None":
            try:
                # Load config
                config = st.session_state.config_manager.load_config(eval_config)

                # Create environment to get dimensions
                env = FOZEnvironment(eval_config)

                # Create agent
                agent = PPOAgent(
                    state_dim=env.state_dim,
                    action_dim=env.action_dim,
                    config=config
                )

                # Load checkpoint
                agent.load(selected_checkpoint)

                st.session_state.loaded_agent = agent
                st.session_state.loaded_env = env
                st.session_state.loaded_config = config

                st.success(f"✅ Model loaded: {Path(selected_checkpoint).name}")

            except Exception as e:
                st.error(f"❌ Error loading model: {e}")
        else:
            st.warning("Please select a checkpoint")

with col2:
    if st.button("🧪 Run Evaluation", use_container_width=True):
        if st.session_state.loaded_agent:
            with st.spinner("Running evaluation..."):
                agent = st.session_state.loaded_agent
                env = st.session_state.loaded_env

                # Run evaluation episodes
                eval_results = {
                    'episode_rewards': [],
                    'episode_lengths': [],
                    'lives_lost': [],
                    'cumulative_damage': [],
                    'resources_used': [],
                    'actions_taken': []
                }

                progress_bar = st.progress(0)

                for ep in range(num_eval_episodes):
                    state = env.reset()
                    done = False
                    episode_reward = 0
                    episode_actions = []

                    while not done:
                        action_mask = env.get_valid_actions()
                        action, _, _ = agent.select_action(state, action_mask)
                        state, reward, done, info = env.step(action)

                        episode_reward += reward
                        episode_actions.append(action)

                    # Store results
                    eval_results['episode_rewards'].append(episode_reward)
                    eval_results['episode_lengths'].append(env.time_step)
                    eval_results['lives_lost'].append(env.cumulative_lives_lost)
                    eval_results['cumulative_damage'].append(env.cumulative_damage)
                    eval_results['resources_used'].append(100 - env.resources)
                    eval_results['actions_taken'].append(episode_actions)

                    progress_bar.progress((ep + 1) / num_eval_episodes)

                st.session_state.eval_results = eval_results
                st.success(f"✅ Evaluation complete! Ran {num_eval_episodes} episodes")

        else:
            st.warning("Please load a model first")

st.markdown("---")

# Display results
if st.session_state.eval_results:
    results = st.session_state.eval_results

    # Summary metrics
    st.markdown("### 📈 Performance Summary")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        mean_reward = np.mean(results['episode_rewards'])
        std_reward = np.std(results['episode_rewards'])
        st.metric(
            "Mean Reward",
            f"{mean_reward:.2f}",
            delta=f"±{std_reward:.2f}"
        )

    with col2:
        mean_lives_lost = np.mean(results['lives_lost'])
        st.metric(
            "Mean Lives Lost",
            f"{mean_lives_lost:.1f}",
            delta=None,
            delta_color="inverse"
        )

    with col3:
        mean_damage = np.mean(results['cumulative_damage'])
        st.metric(
            "Mean Damage",
            f"${mean_damage:.0f}",
            delta=None,
            delta_color="inverse"
        )

    with col4:
        mean_length = np.mean(results['episode_lengths'])
        st.metric(
            "Mean Episode Length",
            f"{mean_length:.1f}"
        )

    with col5:
        mean_resources = np.mean(results['resources_used'])
        st.metric(
            "Mean Resources Used",
            f"{mean_resources:.1f}"
        )

    st.markdown("---")

    # Detailed visualizations
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Performance Metrics",
        "🎯 Action Analysis",
        "📈 Episode Comparison",
        "💾 Export Results"
    ])

    # ========================================================================
    # TAB 1: Performance Metrics
    # ========================================================================
    with tab1:
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Episode Rewards', 'Lives Lost', 'Cumulative Damage', 'Resources Used'),
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )

        episodes = list(range(len(results['episode_rewards'])))

        # Episode rewards
        fig.add_trace(
            go.Bar(x=episodes, y=results['episode_rewards'],
                   name='Reward', marker_color='lightblue'),
            row=1, col=1
        )

        # Add mean line
        fig.add_trace(
            go.Scatter(x=episodes, y=[mean_reward] * len(episodes),
                      mode='lines', name='Mean',
                      line=dict(color='red', dash='dash', width=2)),
            row=1, col=1
        )

        # Lives lost
        fig.add_trace(
            go.Bar(x=episodes, y=results['lives_lost'],
                   name='Lives Lost', marker_color='red'),
            row=1, col=2
        )

        # Cumulative damage
        fig.add_trace(
            go.Bar(x=episodes, y=results['cumulative_damage'],
                   name='Damage', marker_color='orange'),
            row=2, col=1
        )

        # Resources used
        fig.add_trace(
            go.Bar(x=episodes, y=results['resources_used'],
                   name='Resources', marker_color='green'),
            row=2, col=2
        )

        fig.update_layout(height=600, showlegend=False)
        fig.update_xaxes(title_text="Episode", row=2, col=1)
        fig.update_xaxes(title_text="Episode", row=2, col=2)

        st.plotly_chart(fig, use_container_width=True)

        # Distribution analysis
        st.markdown("### 📊 Distribution Analysis")

        col1, col2 = st.columns(2)

        with col1:
            # Reward distribution
            fig_dist = go.Figure()
            fig_dist.add_trace(go.Histogram(
                x=results['episode_rewards'],
                nbinsx=20,
                name='Rewards',
                marker_color='lightblue'
            ))
            fig_dist.update_layout(
                title='Reward Distribution',
                xaxis_title='Reward',
                yaxis_title='Count',
                height=300
            )
            st.plotly_chart(fig_dist, use_container_width=True)

        with col2:
            # Lives lost distribution
            fig_lives = go.Figure()
            fig_lives.add_trace(go.Histogram(
                x=results['lives_lost'],
                nbinsx=20,
                name='Lives Lost',
                marker_color='red'
            ))
            fig_lives.update_layout(
                title='Lives Lost Distribution',
                xaxis_title='Lives Lost',
                yaxis_title='Count',
                height=300
            )
            st.plotly_chart(fig_lives, use_container_width=True)

    # ========================================================================
    # TAB 2: Action Analysis
    # ========================================================================
    with tab2:
        st.markdown("### 🎯 Action Frequency Analysis")

        # Count action frequencies across all episodes
        all_actions = []
        for actions in results['actions_taken']:
            all_actions.extend(actions)

        action_counts = pd.Series(all_actions).value_counts().sort_index()

        # Action names (default)
        action_names = {
            0: "Do Nothing",
            1: "Evacuate Zone",
            2: "Sandbag Deployment",
            3: "Activate Flood Gates",
            4: "Emergency Alert"
        }

        # Create bar chart
        fig_actions = go.Figure()
        fig_actions.add_trace(go.Bar(
            x=[action_names.get(i, f"Action {i}") for i in action_counts.index],
            y=action_counts.values,
            marker_color='teal'
        ))
        fig_actions.update_layout(
            title='Action Frequency Across All Episodes',
            xaxis_title='Action',
            yaxis_title='Count',
            height=400
        )
        st.plotly_chart(fig_actions, use_container_width=True)

        # Action sequence analysis
        st.markdown("### 📜 Action Sequences")

        selected_episode = st.selectbox(
            "Select Episode",
            list(range(len(results['actions_taken'])))
        )

        episode_actions = results['actions_taken'][selected_episode]
        episode_action_names = [action_names.get(a, f"Action {a}") for a in episode_actions]

        # Display as timeline
        fig_timeline = go.Figure()
        fig_timeline.add_trace(go.Scatter(
            x=list(range(len(episode_actions))),
            y=episode_actions,
            mode='lines+markers',
            marker=dict(size=10, color=episode_actions, colorscale='Viridis'),
            line=dict(width=2)
        ))
        fig_timeline.update_layout(
            title=f'Action Sequence - Episode {selected_episode}',
            xaxis_title='Step',
            yaxis_title='Action ID',
            height=400
        )
        st.plotly_chart(fig_timeline, use_container_width=True)

        # Episode metrics
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Reward", f"{results['episode_rewards'][selected_episode]:.2f}")

        with col2:
            st.metric("Lives Lost", f"{results['lives_lost'][selected_episode]}")

        with col3:
            st.metric("Damage", f"${results['cumulative_damage'][selected_episode]:.0f}")

    # ========================================================================
    # TAB 3: Episode Comparison
    # ========================================================================
    with tab3:
        st.markdown("### 📈 Episode-by-Episode Comparison")

        # Create comprehensive comparison table
        comparison_df = pd.DataFrame({
            'Episode': list(range(len(results['episode_rewards']))),
            'Reward': results['episode_rewards'],
            'Lives Lost': results['lives_lost'],
            'Damage': [f"${d:.0f}" for d in results['cumulative_damage']],
            'Length': results['episode_lengths'],
            'Resources Used': results['resources_used']
        })

        st.dataframe(comparison_df, use_container_width=True, hide_index=True)

        # Best and worst episodes
        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 🏆 Best Episodes (by Reward)")
            best_indices = np.argsort(results['episode_rewards'])[-3:][::-1]
            best_df = comparison_df.iloc[best_indices]
            st.dataframe(best_df, use_container_width=True, hide_index=True)

        with col2:
            st.markdown("#### ⚠️ Worst Episodes (by Lives Lost)")
            worst_indices = np.argsort(results['lives_lost'])[-3:][::-1]
            worst_df = comparison_df.iloc[worst_indices]
            st.dataframe(worst_df, use_container_width=True, hide_index=True)

    # ========================================================================
    # TAB 4: Export Results
    # ========================================================================
    with tab4:
        st.markdown("### 💾 Export Evaluation Results")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 📊 Export Metrics")

            # Create comprehensive results
            export_data = {
                'summary': {
                    'num_episodes': len(results['episode_rewards']),
                    'mean_reward': float(mean_reward),
                    'std_reward': float(std_reward),
                    'mean_lives_lost': float(mean_lives_lost),
                    'mean_damage': float(mean_damage),
                    'mean_episode_length': float(mean_length)
                },
                'episodes': results
            }

            # Convert to JSON
            json_str = json.dumps(export_data, indent=2, default=lambda x: float(x) if isinstance(x, np.number) else list(x))

            st.download_button(
                label="📥 Download JSON",
                data=json_str,
                file_name="evaluation_results.json",
                mime="application/json",
                use_container_width=True
            )

        with col2:
            st.markdown("#### 📊 Export CSV")

            csv = comparison_df.to_csv(index=False)

            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name="evaluation_results.csv",
                mime="text/csv",
                use_container_width=True
            )

        st.markdown("---")

        # Save to disk
        st.markdown("#### 💾 Save Results to Project")

        save_name = st.text_input(
            "Results Name",
            value=f"eval_{Path(selected_checkpoint).stem if selected_checkpoint != 'None' else 'model'}"
        )

        if st.button("💾 Save Results", use_container_width=True):
            results_dir = Path("results/evaluations")
            results_dir.mkdir(parents=True, exist_ok=True)

            save_path = results_dir / f"{save_name}.json"

            with open(save_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=lambda x: float(x) if isinstance(x, np.number) else list(x))

            st.success(f"✅ Results saved to: {save_path}")

else:
    st.info("👆 Load a model and run evaluation to see results")
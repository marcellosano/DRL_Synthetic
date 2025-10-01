"""
Configuration Hub Page
Interactive parameter editor, reward function designer, and action space configuration
"""

import streamlit as st
import yaml
import sys
from pathlib import Path
import copy

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from utils.config_manager import ConfigManager

st.set_page_config(page_title="Configuration Hub", page_icon="⚙️", layout="wide")

# Initialize
if 'config_manager' not in st.session_state:
    st.session_state.config_manager = ConfigManager()

if 'current_config' not in st.session_state:
    st.session_state.current_config = None

if 'config_modified' not in st.session_state:
    st.session_state.config_modified = False

st.markdown("# ⚙️ Configuration Hub")
st.markdown("Configure training parameters, reward functions, and action spaces")
st.markdown("---")

# Configuration selector
col1, col2 = st.columns([3, 1])

with col1:
    config_files = list(Path("config").rglob("*.yaml"))
    config_options = ["config/base.yaml"] + [str(f) for f in config_files if f.name != "base.yaml"]

    selected_config = st.selectbox(
        "Select Configuration Template",
        config_options,
        index=0
    )

with col2:
    if st.button("📂 Load Config", use_container_width=True):
        st.session_state.current_config = st.session_state.config_manager.load_config(selected_config)
        st.session_state.config_modified = False
        st.success(f"Loaded: {selected_config}")

# Tabs for different configuration sections
if st.session_state.current_config:
    config = st.session_state.current_config

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎓 Training Parameters",
        "💰 Reward Function",
        "🎯 Action Space",
        "🌍 Environment",
        "💾 Save & Export"
    ])

    # ========================================================================
    # TAB 1: Training Parameters
    # ========================================================================
    with tab1:
        st.markdown("### 🎓 Training Hyperparameters")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### PPO Parameters")

            learning_rate = st.number_input(
                "Learning Rate",
                value=float(config['training']['learning_rate']),
                format="%.6f",
                min_value=0.0,
                max_value=0.01,
                step=0.0001
            )

            clip_epsilon = st.slider(
                "Clip Epsilon",
                min_value=0.1,
                max_value=0.5,
                value=float(config['training']['clip_epsilon']),
                step=0.05
            )

            gamma = st.slider(
                "Gamma (Discount Factor)",
                min_value=0.9,
                max_value=0.999,
                value=float(config['training']['gamma']),
                step=0.001
            )

            gae_lambda = st.slider(
                "GAE Lambda",
                min_value=0.9,
                max_value=0.99,
                value=float(config['training']['gae_lambda']),
                step=0.01
            )

        with col2:
            st.markdown("#### Training Schedule")

            episodes = st.number_input(
                "Total Episodes",
                value=int(config['training']['episodes']),
                min_value=10,
                max_value=100000,
                step=10
            )

            batch_size = st.selectbox(
                "Batch Size",
                options=[16, 32, 64, 128, 256],
                index=[16, 32, 64, 128, 256].index(config['training']['batch_size'])
            )

            update_epochs = st.slider(
                "Update Epochs",
                min_value=1,
                max_value=10,
                value=int(config['training']['update_epochs']),
                step=1
            )

            max_steps_per_episode = st.number_input(
                "Max Steps per Episode",
                value=int(config['training']['max_steps_per_episode']),
                min_value=10,
                max_value=1000,
                step=10
            )

        st.markdown("---")
        st.markdown("#### Loss Coefficients")

        col1, col2, col3 = st.columns(3)

        with col1:
            value_loss_coef = st.slider(
                "Value Loss Coefficient",
                min_value=0.1,
                max_value=1.0,
                value=float(config['training']['value_loss_coef']),
                step=0.1
            )

        with col2:
            entropy_coef = st.slider(
                "Entropy Coefficient",
                min_value=0.0,
                max_value=0.1,
                value=float(config['training']['entropy_coef']),
                step=0.01
            )

        with col3:
            gradient_clip = st.slider(
                "Gradient Clipping",
                min_value=0.1,
                max_value=5.0,
                value=float(config['training']['gradient_clip']),
                step=0.1
            )

        # Update config
        config['training']['learning_rate'] = learning_rate
        config['training']['clip_epsilon'] = clip_epsilon
        config['training']['gamma'] = gamma
        config['training']['gae_lambda'] = gae_lambda
        config['training']['episodes'] = episodes
        config['training']['batch_size'] = batch_size
        config['training']['update_epochs'] = update_epochs
        config['training']['max_steps_per_episode'] = max_steps_per_episode
        config['training']['value_loss_coef'] = value_loss_coef
        config['training']['entropy_coef'] = entropy_coef
        config['training']['gradient_clip'] = gradient_clip

    # ========================================================================
    # TAB 2: Reward Function
    # ========================================================================
    with tab2:
        st.markdown("### 💰 Reward Function Designer")
        st.markdown("Adjust weights to shape agent behavior")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Core Objectives")

            lives_saved_weight = st.slider(
                "Lives Saved Weight 👥",
                min_value=0.0,
                max_value=50.0,
                value=float(config['reward']['lives_saved_weight']),
                step=1.0,
                help="Reward for saving lives (higher = prioritize life safety)"
            )

            infrastructure_damage_weight = st.slider(
                "Infrastructure Damage Penalty 🏗️",
                min_value=-20.0,
                max_value=0.0,
                value=float(config['reward']['infrastructure_damage_weight']),
                step=1.0,
                help="Penalty for infrastructure damage (more negative = avoid damage)"
            )

            economic_cost_weight = st.slider(
                "Economic Cost Weight 💵",
                min_value=-1.0,
                max_value=0.0,
                value=float(config['reward']['economic_cost_weight']),
                step=0.05,
                help="Penalty for economic costs"
            )

        with col2:
            st.markdown("#### Behavioral Incentives")

            early_warning_bonus = st.slider(
                "Early Warning Bonus ⏰",
                min_value=0.0,
                max_value=20.0,
                value=float(config['reward']['early_warning_bonus']),
                step=1.0,
                help="Bonus for taking action early"
            )

            false_alarm_penalty = st.slider(
                "False Alarm Penalty 🚨",
                min_value=-10.0,
                max_value=0.0,
                value=float(config['reward']['false_alarm_penalty']),
                step=0.5,
                help="Penalty for unnecessary warnings"
            )

            quick_response_bonus = st.slider(
                "Quick Response Bonus ⚡",
                min_value=0.0,
                max_value=10.0,
                value=float(config['reward']['quick_response_bonus']),
                step=0.5,
                help="Bonus for fast decision-making"
            )

        st.markdown("---")

        # Reward preview
        st.markdown("#### 📊 Reward Function Preview")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Lives Priority", f"{lives_saved_weight:.1f}", help="Higher is better")

        with col2:
            total_penalties = abs(infrastructure_damage_weight) + abs(economic_cost_weight) + abs(false_alarm_penalty)
            st.metric("Total Penalties", f"{total_penalties:.1f}", help="Cost of actions")

        with col3:
            total_bonuses = early_warning_bonus + quick_response_bonus
            st.metric("Total Bonuses", f"{total_bonuses:.1f}", help="Performance incentives")

        # Multi-objective weights
        st.markdown("#### ⚖️ Multi-Objective Weights")
        st.markdown("Balance between safety, economy, and efficiency (must sum to 1.0)")

        col1, col2, col3 = st.columns(3)

        with col1:
            obj_safety = st.slider(
                "Safety Weight",
                min_value=0.0,
                max_value=1.0,
                value=float(config['reward']['objective_weights']['safety']),
                step=0.1
            )

        with col2:
            obj_economy = st.slider(
                "Economy Weight",
                min_value=0.0,
                max_value=1.0,
                value=float(config['reward']['objective_weights']['economy']),
                step=0.1
            )

        with col3:
            obj_efficiency = st.slider(
                "Efficiency Weight",
                min_value=0.0,
                max_value=1.0,
                value=float(config['reward']['objective_weights']['efficiency']),
                step=0.1
            )

        # Validate weights sum to 1.0
        total_weight = obj_safety + obj_economy + obj_efficiency
        if abs(total_weight - 1.0) > 0.01:
            st.warning(f"⚠️ Weights sum to {total_weight:.2f}, should be 1.0")

        # Update config
        config['reward']['lives_saved_weight'] = lives_saved_weight
        config['reward']['infrastructure_damage_weight'] = infrastructure_damage_weight
        config['reward']['economic_cost_weight'] = economic_cost_weight
        config['reward']['early_warning_bonus'] = early_warning_bonus
        config['reward']['false_alarm_penalty'] = false_alarm_penalty
        config['reward']['quick_response_bonus'] = quick_response_bonus
        config['reward']['objective_weights']['safety'] = obj_safety
        config['reward']['objective_weights']['economy'] = obj_economy
        config['reward']['objective_weights']['efficiency'] = obj_efficiency

    # ========================================================================
    # TAB 3: Action Space
    # ========================================================================
    with tab3:
        st.markdown("### 🎯 Action Space Configuration")
        st.markdown("Define available actions and their properties")

        # Default actions
        default_actions = [
            {"id": 0, "name": "do_nothing", "cost": 0, "description": "Monitor situation"},
            {"id": 1, "name": "evacuate_zone", "cost": 50, "description": "Evacuate high-risk zones"},
            {"id": 2, "name": "sandbag_deployment", "cost": 20, "description": "Deploy sandbags"},
            {"id": 3, "name": "activate_flood_gates", "cost": 30, "description": "Close flood barriers"},
            {"id": 4, "name": "emergency_alert", "cost": 10, "description": "Issue public alert"},
        ]

        st.markdown("#### Available Actions")

        for i, action in enumerate(default_actions):
            with st.expander(f"Action {action['id']}: {action['name']}", expanded=(i == 0)):
                col1, col2, col3 = st.columns([2, 1, 2])

                with col1:
                    action_name = st.text_input(
                        "Action Name",
                        value=action['name'],
                        key=f"action_name_{i}"
                    )

                with col2:
                    action_cost = st.number_input(
                        "Cost",
                        value=action['cost'],
                        min_value=0,
                        max_value=100,
                        key=f"action_cost_{i}"
                    )

                with col3:
                    action_desc = st.text_input(
                        "Description",
                        value=action['description'],
                        key=f"action_desc_{i}"
                    )

        # Action masking settings
        st.markdown("---")
        st.markdown("#### Action Masking")

        use_action_masking = st.checkbox(
            "Enable Action Masking",
            value=config['network'].get('use_action_masking', True),
            help="Prevent agent from selecting invalid actions"
        )

        mask_invalid_actions = st.checkbox(
            "Mask Resource-Constrained Actions",
            value=config['network'].get('mask_invalid_actions', True),
            help="Automatically mask actions that exceed available resources"
        )

        config['network']['use_action_masking'] = use_action_masking
        config['network']['mask_invalid_actions'] = mask_invalid_actions

    # ========================================================================
    # TAB 4: Environment
    # ========================================================================
    with tab4:
        st.markdown("### 🌍 Environment Configuration")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Grid Settings")

            grid_size = st.slider(
                "Grid Size",
                min_value=10,
                max_value=50,
                value=int(config['environment']['grid_size']),
                step=5
            )

            population_density = st.slider(
                "Population Density",
                min_value=0.0,
                max_value=1.0,
                value=float(config['environment']['population_density']),
                step=0.1
            )

            vulnerable_population_ratio = st.slider(
                "Vulnerable Population Ratio",
                min_value=0.0,
                max_value=1.0,
                value=float(config['environment']['vulnerable_population_ratio']),
                step=0.05
            )

        with col2:
            st.markdown("#### Hazard Parameters")

            max_storms = st.slider(
                "Max Concurrent Storms",
                min_value=1,
                max_value=10,
                value=int(config['environment']['max_storms']),
                step=1
            )

            storm_spawn_probability = st.slider(
                "Storm Spawn Probability",
                min_value=0.0,
                max_value=0.5,
                value=float(config['environment']['storm_spawn_probability']),
                step=0.05
            )

            evacuation_time = st.slider(
                "Evacuation Time (steps)",
                min_value=1,
                max_value=10,
                value=int(config['environment']['evacuation_time']),
                step=1
            )

        # Update config
        config['environment']['grid_size'] = grid_size
        config['environment']['population_density'] = population_density
        config['environment']['vulnerable_population_ratio'] = vulnerable_population_ratio
        config['environment']['max_storms'] = max_storms
        config['environment']['storm_spawn_probability'] = storm_spawn_probability
        config['environment']['evacuation_time'] = evacuation_time

    # ========================================================================
    # TAB 5: Save & Export
    # ========================================================================
    with tab5:
        st.markdown("### 💾 Save Configuration")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Save as New Config")

            new_config_name = st.text_input(
                "Configuration Name",
                value="my_custom_config",
                help="Name for the new configuration file"
            )

            save_location = st.selectbox(
                "Save Location",
                ["config/experiments/", "config/"],
                index=0
            )

            if st.button("💾 Save Configuration", use_container_width=True):
                save_path = Path(save_location) / f"{new_config_name}.yaml"
                save_path.parent.mkdir(parents=True, exist_ok=True)

                with open(save_path, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False, sort_keys=False)

                st.success(f"✅ Configuration saved to: {save_path}")
                st.session_state.config_modified = False

        with col2:
            st.markdown("#### Export Configuration")

            st.markdown("Download current configuration as YAML")

            config_yaml = yaml.dump(config, default_flow_style=False, sort_keys=False)

            st.download_button(
                label="📥 Download YAML",
                data=config_yaml,
                file_name=f"{new_config_name}.yaml",
                mime="text/yaml",
                use_container_width=True
            )

        st.markdown("---")

        # Preview current config
        with st.expander("👁️ Preview Full Configuration"):
            st.code(config_yaml, language='yaml')

    st.session_state.current_config = config
    st.session_state.config_modified = True

else:
    st.info("👆 Please select and load a configuration file to get started")
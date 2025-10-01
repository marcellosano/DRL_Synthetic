"""
Training wrapper for dashboard integration
Provides real-time metrics and control for PPO training
"""

import torch
import numpy as np
import json
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Callable
from pathlib import Path
import threading

import sys
sys.path.append('/home/msano/Projects/DRL_Synthetic')

from environments.foz_environment import FOZEnvironment
from models.ppo_agent import PPOAgent
from utils.config_manager import ConfigManager


class TrainingSession:
    """Manages a single training session with metrics tracking"""

    def __init__(self, config_path: str, run_name: Optional[str] = None):
        """
        Initialize training session

        Args:
            config_path: Path to config YAML file
            run_name: Optional custom run name
        """
        self.config_manager = ConfigManager()
        self.config = self.config_manager.load_config(config_path)

        # Create run
        self.run = self.config_manager.create_experiment_run(
            config_path, run_name=run_name
        )
        self.run_dir = Path(self.run.run_dir)

        # Create environment
        self.env = FOZEnvironment(config_path)

        # Create agent
        self.agent = PPOAgent(
            state_dim=self.env.state_dim,
            action_dim=self.env.action_dim,
            config=self.config
        )

        # Training state
        self.current_episode = 0
        self.total_episodes = self.config['training']['episodes']
        self.is_training = False
        self.should_stop = False
        self.is_paused = False

        # Metrics storage
        self.metrics = {
            'episode_rewards': [],
            'episode_lengths': [],
            'lives_saved': [],
            'lives_lost': [],
            'cumulative_damage': [],
            'false_alarms': [],
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'resources_used': []
        }

        # Real-time metrics (for live updates)
        self.current_metrics = {}

        # Callbacks
        self.callbacks = []

        # Metrics file
        self.metrics_file = self.run_dir / 'metrics.json'
        self.checkpoint_dir = self.run_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(exist_ok=True)

    def add_callback(self, callback: Callable):
        """Add callback function to be called after each episode"""
        self.callbacks.append(callback)

    def save_metrics(self):
        """Save metrics to disk"""
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)

    def load_metrics(self):
        """Load metrics from disk"""
        if self.metrics_file.exists():
            with open(self.metrics_file, 'r') as f:
                self.metrics = json.load(f)

    def train_episode(self) -> Dict:
        """Train a single episode"""
        state = self.env.reset()
        done = False
        episode_reward = 0
        episode_length = 0

        episode_lives_saved = 0
        episode_false_alarms = 0

        while not done:
            # Get valid actions
            action_mask = self.env.get_valid_actions()

            # Select action
            action, log_prob, value = self.agent.select_action(state, action_mask)

            # Take step
            next_state, reward, done, info = self.env.step(action)

            # Store transition
            self.agent.store_transition(
                state, action, log_prob, reward, value, done, action_mask
            )

            # Update metrics
            episode_reward += reward
            episode_length += 1

            # Track lives and false alarms
            if 'lives_lost' in info:
                episode_lives_saved += (info.get('population_at_risk', 0) - info['lives_lost'])

            state = next_state

        # Perform PPO update
        update_metrics = self.agent.update()

        # Compile episode metrics
        episode_metrics = {
            'episode': self.current_episode,
            'reward': episode_reward,
            'length': episode_length,
            'lives_saved': episode_lives_saved,
            'lives_lost': self.env.cumulative_lives_lost,
            'cumulative_damage': self.env.cumulative_damage,
            'false_alarms': episode_false_alarms,
            'resources_used': 100 - self.env.resources,
            **update_metrics
        }

        return episode_metrics

    def train(self):
        """Main training loop"""
        self.is_training = True
        self.should_stop = False

        print(f"Starting training: {self.run.name}")
        print(f"Episodes: {self.total_episodes}")
        print(f"Run directory: {self.run_dir}")

        start_time = time.time()

        for episode in range(self.current_episode, self.total_episodes):
            if self.should_stop:
                print("Training stopped by user")
                break

            # Wait if paused
            while self.is_paused:
                time.sleep(0.1)

            self.current_episode = episode

            # Train episode
            episode_metrics = self.train_episode()

            # Store metrics
            for key, value in episode_metrics.items():
                if key in self.metrics:
                    self.metrics[key].append(value)

            # Update current metrics for dashboard
            self.current_metrics = episode_metrics

            # Update run in config manager
            self.config_manager.update_run_metrics(
                self.run.name,
                {k: float(v) if isinstance(v, (int, float, np.number)) else v
                 for k, v in episode_metrics.items()}
            )

            # Call callbacks
            for callback in self.callbacks:
                callback(episode_metrics)

            # Save periodically
            if (episode + 1) % self.config['experiment']['save_frequency'] == 0:
                self.save_checkpoint(f"episode_{episode + 1}")
                self.save_metrics()

            # Log progress
            if (episode + 1) % self.config['experiment']['log_frequency'] == 0:
                elapsed = time.time() - start_time
                print(f"Episode {episode + 1}/{self.total_episodes} | "
                      f"Reward: {episode_metrics['reward']:.2f} | "
                      f"Lives Lost: {episode_metrics['lives_lost']} | "
                      f"Time: {elapsed:.1f}s")

        # Final save
        self.save_checkpoint("final")
        self.save_metrics()

        self.is_training = False
        print(f"Training completed! Total time: {time.time() - start_time:.1f}s")

    def start_async(self):
        """Start training in background thread"""
        self.training_thread = threading.Thread(target=self.train, daemon=True)
        self.training_thread.start()

    def stop(self):
        """Stop training"""
        self.should_stop = True

    def pause(self):
        """Pause training"""
        self.is_paused = True

    def resume(self):
        """Resume training"""
        self.is_paused = False

    def save_checkpoint(self, name: str):
        """Save model checkpoint"""
        checkpoint_path = self.checkpoint_dir / f"{name}.pt"
        self.agent.save(str(checkpoint_path))
        print(f"Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, name: str):
        """Load model checkpoint"""
        checkpoint_path = self.checkpoint_dir / f"{name}.pt"
        if checkpoint_path.exists():
            self.agent.load(str(checkpoint_path))
            print(f"Checkpoint loaded: {checkpoint_path}")
        else:
            print(f"Checkpoint not found: {checkpoint_path}")

    def get_metrics_summary(self) -> Dict:
        """Get summary statistics of training metrics"""
        if not self.metrics['episode_rewards']:
            return {}

        summary = {}
        for key, values in self.metrics.items():
            if values:
                summary[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'last': values[-1]
                }

        return summary

    def evaluate(self, num_episodes: int = 10) -> Dict:
        """Evaluate agent performance"""
        eval_metrics = {
            'rewards': [],
            'lives_saved': [],
            'lives_lost': [],
            'damage': []
        }

        for _ in range(num_episodes):
            state = self.env.reset()
            done = False
            episode_reward = 0

            while not done:
                action_mask = self.env.get_valid_actions()
                action, _, _ = self.agent.select_action(state, action_mask)
                state, reward, done, info = self.env.step(action)
                episode_reward += reward

            eval_metrics['rewards'].append(episode_reward)
            eval_metrics['lives_lost'].append(self.env.cumulative_lives_lost)
            eval_metrics['damage'].append(self.env.cumulative_damage)

        # Compute statistics
        return {
            'mean_reward': np.mean(eval_metrics['rewards']),
            'std_reward': np.std(eval_metrics['rewards']),
            'mean_lives_lost': np.mean(eval_metrics['lives_lost']),
            'mean_damage': np.mean(eval_metrics['damage'])
        }


class TrainingManager:
    """Manages multiple training sessions"""

    def __init__(self):
        self.sessions = {}
        self.active_session = None

    def create_session(self, config_path: str, run_name: Optional[str] = None) -> TrainingSession:
        """Create new training session"""
        session = TrainingSession(config_path, run_name)
        self.sessions[session.run.name] = session
        self.active_session = session.run.name
        return session

    def get_session(self, name: str) -> Optional[TrainingSession]:
        """Get training session by name"""
        return self.sessions.get(name)

    def get_active_session(self) -> Optional[TrainingSession]:
        """Get currently active session"""
        if self.active_session:
            return self.sessions.get(self.active_session)
        return None

    def list_sessions(self) -> List[str]:
        """List all session names"""
        return list(self.sessions.keys())
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any, Optional

class BaseEnvironment(ABC):
    """Abstract base class for DRL environments"""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.state = None
        self.step_count = 0
        self.max_steps = self.config.get('max_steps', 24)
        self.done = False

    @abstractmethod
    def reset(self) -> np.ndarray:
        """Reset environment to initial state"""
        self.step_count = 0
        self.done = False
        return self.state

    @abstractmethod
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute action and return (state, reward, done, info)"""
        pass

    @abstractmethod
    def get_valid_actions(self) -> np.ndarray:
        """Return mask of valid actions"""
        pass

    @abstractmethod
    def render(self, mode: str = 'human'):
        """Render the environment"""
        pass

    @property
    def state_dim(self) -> int:
        """Return state dimension"""
        return self.state.shape[0] if self.state is not None else 0

    @property
    def action_dim(self) -> int:
        """Return action dimension"""
        return len(self.get_valid_actions())
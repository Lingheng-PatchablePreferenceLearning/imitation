"""
IsaacLab wrapper for Stable-Baselines3 compatible environments.

This wrapper adapts IsaacLab's GPU-based vectorized environments to work
with libraries like Imitation and Stable-Baselines3.
"""

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvObs, VecEnvStepReturn
from typing import Any, Callable, Optional, Sequence

from isaaclab.envs import DirectRLEnv
from isaaclab.envs import ManagerBasedRLEnv as isaaclab_ManagerBasedRLEnv
from isaaclab_csiro_hri.lab.envs import ManagerBasedRLEnv

import pdb; pdb.set_trace()
class IsaacLabVecEnvWrapper(VecEnv):
    """
    Wrapper to make IsaacLab environments compatible with SB3 and Imitation libraries.
    
    IsaacLab environments are already vectorized and use PyTorch tensors on GPU.
    This wrapper:
    - Handles conversion between PyTorch tensors and NumPy arrays
    - Implements the VecEnv interface expected by SB3/Imitation
    - Manages episode statistics tracking
    """
    
    def __init__(self, env: ManagerBasedRLEnv | DirectRLEnv):
        """
        Args:
            env: An IsaacLab environment instance (ManagerBasedRLEnv or DirectRLEnv)
        """
        # check that input is valid
        if not isinstance(env.unwrapped, ManagerBasedRLEnv) and not isinstance(env.unwrapped, DirectRLEnv) and not isinstance(env.unwrapped, isaaclab_ManagerBasedRLEnv):
            raise ValueError(
                "The environment must be inherited from ManagerBasedRLEnv or DirectRLEnv. Environment type:"
                f" {type(env)}"
            )
        self.env = env
        self.num_envs = env.unwrapped.num_envs
        self.device = env.unwrapped.device
        import pdb; pdb.set_trace()
        # Get observation and action spaces
        # IsaacLab uses batched spaces, we need single env spaces
        observation_space = env.unwrapped.single_observation_space
        action_space = env.unwrapped.single_action_space
        
        # Initialize VecEnv
        super().__init__(self.num_envs, observation_space, action_space)
        
        # Episode tracking
        self.episode_count = 0
        self.episode_returns = torch.zeros(self.num_envs, device=self.device)
        self.episode_lengths = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)
        
    def reset(self) -> VecEnvObs:
        """Reset all environments."""
        obs_dict, info = self.env.reset()
        
        # Reset episode tracking
        self.episode_returns.zero_()
        self.episode_lengths.zero_()
        
        # Convert observation to numpy
        if isinstance(obs_dict, dict):
            # Use "policy" observation group if available, otherwise use the first group
            obs_key = "policy" if "policy" in obs_dict else list(obs_dict.keys())[0]
            obs = obs_dict[obs_key]
        else:
            obs = obs_dict
            
        return self._to_numpy(obs)
    
    def step_async(self, actions: np.ndarray) -> None:
        """Send actions to the environment (async)."""
        # Convert numpy actions to torch tensors
        self.actions = torch.from_numpy(actions).to(self.device)
    
    def step_wait(self) -> VecEnvStepReturn:
        """Wait for the step to complete and return results."""
        # Execute actions
        obs_dict, rewards, terminated, truncated, infos = self.env.step(self.actions)
        
        # Update episode tracking
        self.episode_returns += rewards
        self.episode_lengths += 1
        
        # Handle episode end
        dones = terminated | truncated
        if dones.any():
            # Log episode info
            for idx in torch.where(dones)[0]:
                idx_item = idx.item()
                # Add episode info to infos dict
                if not isinstance(infos, dict):
                    infos = {}
                if "episode" not in infos:
                    infos["episode"] = {}
                    
                # Store episode statistics
                if idx_item not in infos["episode"]:
                    infos["episode"][idx_item] = {}
                    
                infos["episode"][idx_item]["r"] = self.episode_returns[idx].item()
                infos["episode"][idx_item]["l"] = self.episode_lengths[idx].item()
                
                # Reset tracking for done episodes
                self.episode_returns[idx] = 0
                self.episode_lengths[idx] = 0
        
        # Convert observation to numpy
        if isinstance(obs_dict, dict):
            obs_key = "policy" if "policy" in obs_dict else list(obs_dict.keys())[0]
            obs = obs_dict[obs_key]
        else:
            obs = obs_dict
        
        return (
            self._to_numpy(obs),
            self._to_numpy(rewards),
            self._to_numpy(dones),
            infos,
        )
    
    def close(self) -> None:
        """Close the environment."""
        self.env.close()
    
    def seed(self, seed: Optional[int] = None) -> Sequence[Optional[int]]:
        """Set the random seed."""
        if seed is not None:
            self.env.unwrapped.seed(seed)
        return [seed] * self.num_envs
    
    def env_is_wrapped(self, wrapper_class: type, indices: Sequence[int] = None) -> Sequence[bool]:
        """Check if environment is wrapped."""
        return [False] * self.num_envs
    
    def env_method(
        self,
        method_name: str,
        *method_args,
        indices: Sequence[int] = None,
        **method_kwargs,
    ) -> Sequence[Any]:
        """Call a method on the environment."""
        raise NotImplementedError("env_method is not implemented for IsaacLab environments")
    
    def get_attr(self, attr_name: str, indices: Sequence[int] = None) -> Sequence[Any]:
        """Get an attribute from the environment."""
        return [getattr(self.env, attr_name)] * self.num_envs
    
    def set_attr(self, attr_name: str, value: Any, indices: Sequence[int] = None) -> None:
        """Set an attribute in the environment."""
        setattr(self.env, attr_name, value)
    
    def _to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert PyTorch tensor to NumPy array."""
        return tensor.detach().cpu().numpy()
    
    @property
    def unwrapped(self):
        """Return the unwrapped environment."""
        return self.env.unwrapped


def make_isaaclab_env(task_name: str, num_envs: int = 8, device: str = "cuda:0", headless: bool = True):
    """
    Create an IsaacLab environment wrapped for use with Imitation/SB3.
    
    Args:
        task_name: Name of the IsaacLab task (e.g., "Isaac-Ant-v0")
        num_envs: Number of parallel environments
        device: Device to run simulation on
        headless: Whether to run in headless mode (no GUI)
        
    Returns:
        Wrapped IsaacLab environment compatible with SB3/Imitation
    """
    import gymnasium as gym
    
    # Import IsaacLab and initialize AppLauncher (required for Isaac Sim)
    from isaaclab.app import AppLauncher
    
    # Create launcher configuration
    app_launcher = AppLauncher(headless=headless)
    simulation_app = app_launcher.app
    
    # Import IsaacLab tasks to register environments (must be after AppLauncher)
    try:
        import isaaclab_tasks  # noqa: F401
    except ImportError:
        raise ImportError(
            "IsaacLab tasks not found. Please ensure isaaclab_tasks is installed."
        )
    
    # Parse environment configuration
    from isaaclab_tasks.utils import parse_env_cfg
    env_cfg = parse_env_cfg(task_name, device=device, num_envs=num_envs, use_fabric=False)
    
    # Create the environment
    env = gym.make(task_name, cfg=env_cfg)
    import pdb; pdb.set_trace()
    # Wrap it for SB3/Imitation compatibility
    wrapped_env = IsaacLabVecEnvWrapper(env)
    wrapped_env.simulation_app = simulation_app  # Store for cleanup
    
    return wrapped_env

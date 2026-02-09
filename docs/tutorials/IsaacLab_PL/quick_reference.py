"""
Quick Start Guide: Preference Learning with Isaac-Ant-v0
=========================================================

This is a minimal example showing the key components needed to use
Preference Learning from the Imitation library with IsaacLab environments.
"""

# ============================================================================
# Key Components
# ============================================================================

"""
1. WRAPPER: IsaacLabVecEnvWrapper
   - Converts IsaacLab's GPU-based vectorized environment to SB3-compatible format
   - Handles PyTorch tensor <-> NumPy array conversions
   - Tracks episode statistics
   
2. ENVIRONMENT: Isaac-Ant-v0
   - Locomotion task with quadruped ant robot
   - Already vectorized (runs multiple parallel envs on GPU)
   - Observation space: Box(36,) - joint positions, velocities, body state
   - Action space: Box(8,) - joint torques
   
3. PREFERENCE LEARNING ALGORITHM
   - RandomFragmenter: Samples trajectory fragments randomly
   - SyntheticGatherer: Generates preferences (replace with human feedback)
   - PreferenceModel: Converts rewards to preference probabilities
   - BasicRewardTrainer: Trains reward network from preferences
   - AgentTrainer: Trains RL policy using learned reward
"""

# ============================================================================
# Minimal Working Example
# ============================================================================

import numpy as np
from stable_baselines3 import PPO
from imitation.algorithms import preference_comparisons
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm
from isaaclab_vec_env_wrapper import make_isaaclab_env

# Create environment
venv = make_isaaclab_env("Isaac-Ant-v0", num_envs=8, device="cuda:0")

# Create reward network
reward_net = BasicRewardNet(
    venv.observation_space,
    venv.action_space,
    normalize_input_layer=RunningNorm
)

# Create RL agent
agent = PPO("MlpPolicy", venv, seed=0, verbose=1)

# Set up preference comparisons
rng = np.random.default_rng(0)
trajectory_generator = preference_comparisons.AgentTrainer(
    algorithm=agent,
    reward_fn=reward_net,
    venv=venv,
    rng=rng,
)

pref_comparisons = preference_comparisons.PreferenceComparisons(
    trajectory_generator,
    reward_net,
    num_iterations=5,
    fragmenter=preference_comparisons.RandomFragmenter(rng=rng),
    preference_gatherer=preference_comparisons.SyntheticGatherer(rng=rng),
    fragment_length=100,
)

# Train
pref_comparisons.train(total_timesteps=50_000, total_comparisons=2000)

# Clean up
venv.close()

# ============================================================================
# Key Parameters to Tune
# ============================================================================

"""
ENVIRONMENT:
- num_envs: Number of parallel environments (4-16 typical for GPU)
- device: "cuda:0" for GPU, "cpu" for CPU

PREFERENCE LEARNING:
- num_iterations: How many times to alternate between reward and policy training
- fragment_length: Length of trajectory segments to compare (50-200 typical)
- total_comparisons: Number of preference pairs (1000-5000 typical)
- total_timesteps: Total environment interactions (50k-1M typical)

PPO AGENT:
- learning_rate: 1e-4 to 3e-4 typical
- n_steps: Steps per env before update (1024-4096 typical)
- batch_size: Minibatch size (64-256 typical)
- gamma: Discount factor (0.99 for locomotion)

REWARD NETWORK:
- Use RunningNorm for input normalization (helps stability)
- BasicRewardNet works well for most tasks
- Can use deeper networks for complex tasks
"""

# ============================================================================
# Common Issues and Solutions
# ============================================================================

"""
ISSUE: CUDA out of memory
SOLUTION: Reduce num_envs or use smaller batch_size

ISSUE: Training is too slow
SOLUTION: 
  - Increase num_envs (more parallelization)
  - Use GPU (device="cuda:0")
  - Reduce total_timesteps for faster experimentation

ISSUE: Policy not learning
SOLUTION:
  - Increase total_comparisons (more training signal)
  - Increase num_iterations (more learning cycles)
  - Check reward accuracy (should be >90%)
  - Try different PPO hyperparameters

ISSUE: Reward model overfitting
SOLUTION:
  - Increase fragment diversity (more trajectories)
  - Reduce reward network complexity
  - Add regularization to reward trainer

ISSUE: Unstable training
SOLUTION:
  - Use RunningNorm for observations and rewards
  - Reduce learning rates
  - Increase batch sizes
  - Use smaller initial_epoch_multiplier
"""

# ============================================================================
# Next Steps
# ============================================================================

"""
1. SAVE AND LOAD MODELS
   - Save reward network: torch.save(reward_net.state_dict(), "reward.pt")
   - Save policy: agent.save("policy.zip")
   - Load: agent = PPO.load("policy.zip", env=venv)

2. VISUALIZE TRAINING
   - Use TensorBoard: tensorboard --logdir ./logs
   - Plot preferences: visualize preference accuracy over time
   - Record videos: Use gym wrappers to record episodes

3. HUMAN FEEDBACK
   - Replace SyntheticGatherer with custom gatherer
   - Implement UI for trajectory comparison
   - Store human preferences for reuse

4. ADVANCED TECHNIQUES
   - Active learning: Use uncertainty-based fragment selection
   - Ensemble reward models: Train multiple networks
   - Reward regularization: Add smoothness constraints
   - Multi-task learning: Share reward models across tasks

5. DEPLOYMENT
   - Export learned policy for real robots
   - Fine-tune with real-world data
   - Add safety constraints
"""

print("Quick reference guide loaded!")
print("See the full scripts for complete implementations.")

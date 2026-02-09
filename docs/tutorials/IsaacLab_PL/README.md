# Preference Learning with IsaacLab Isaac-Ant-v0

This directory contains scripts and utilities to run Preference Learning from the Imitation library with IsaacLab's Isaac-Ant-v0 locomotion task.

## Overview

Preference Learning is a form of reward learning where instead of providing explicit reward signals, the algorithm learns from comparisons between trajectory fragments. This is particularly useful for tasks where specifying a reward function is difficult, but comparing trajectories is easier.

## Files

- **`isaaclab_vec_env_wrapper.py`**: Wrapper that makes IsaacLab environments compatible with Stable-Baselines3 and the Imitation library. Handles conversion between PyTorch tensors (GPU) and NumPy arrays (CPU).

- **`train_preference_learning_ant.py`**: Main script that trains a locomotion policy for the Isaac-Ant-v0 task using preference comparisons.

- **`test_wrapper.py`**: Test script to verify the wrapper works correctly before running the full training.

- **`PL_train_demo.py`**: Original demo script using standard Gym environments (HalfCheetah-v4).

- **`PL_train_IsaacLab.py`**: Earlier version of the IsaacLab integration script.

## Requirements

Make sure you have the following installed:

1. **IsaacLab** (with Isaac Sim)
2. **Imitation library**
3. **Stable-Baselines3**
4. **PyTorch** (for GPU acceleration)

## Setup

1. Ensure IsaacLab is properly installed and the Isaac Sim simulator is accessible.

2. Install the Imitation library and dependencies:
   ```bash
   pip install imitation stable-baselines3
   ```

3. Make sure the IsaacLab tasks are importable:
   ```python
   import isaaclab_tasks
   ```

## Usage

### Step 1: Test the Wrapper

First, verify that the wrapper works correctly with your IsaacLab installation:

```bash
python test_wrapper.py
```

This will:
- Create the Isaac-Ant-v0 environment
- Test reset and step functions
- Verify NumPy/PyTorch conversions
- Check compatibility with Stable-Baselines3

### Step 2: Run Preference Learning Training

Once the wrapper test passes, run the main training script:

```bash
python train_preference_learning_ant.py
```

This will:
1. Create 8 parallel Isaac-Ant-v0 environments
2. Initialize a reward network
3. Set up preference comparison components
4. Train an initial PPO agent
5. Iteratively:
   - Collect trajectory fragments
   - Generate preference comparisons
   - Train the reward model
   - Update the policy
6. Train a final policy using the learned reward
7. Evaluate the learned policy

### Configuration

You can modify these parameters in `train_preference_learning_ant.py`:

- **`NUM_ENVS`**: Number of parallel environments (default: 8). Adjust based on GPU memory.
- **`DEVICE`**: Device for simulation (`"cuda:0"` or `"cpu"`)
- **`SEED`**: Random seed for reproducibility
- **Preference Learning Parameters**:
  - `total_timesteps`: Total environment interaction steps (default: 50,000)
  - `total_comparisons`: Number of preference comparisons (default: 2,000)
  - `num_iterations`: Reward model training iterations (default: 5)
  - `fragment_length`: Length of trajectory fragments (default: 100)
- **PPO Parameters**: Learning rate, batch size, etc.

## How It Works

### IsaacLab Environment Wrapper

IsaacLab environments are GPU-accelerated and already vectorized, which differs from standard Gym environments. The `IsaacLabVecEnvWrapper` class bridges this gap by:

1. **Tensor Conversion**: Converting PyTorch tensors (GPU) to NumPy arrays (CPU) for compatibility with SB3/Imitation
2. **VecEnv Interface**: Implementing the `step_async`/`step_wait` pattern expected by vectorized environments
3. **Episode Tracking**: Maintaining episode statistics (returns, lengths) across parallel environments
4. **Observation Handling**: Extracting the "policy" observation group from IsaacLab's multi-group observation structure

### Preference Learning Pipeline

1. **Initialization**: Create reward network and PPO agent
2. **Trajectory Collection**: Agent interacts with environment to generate trajectories
3. **Fragment Sampling**: Random fragments are sampled from trajectories
4. **Preference Generation**: Synthetic preferences are generated (in practice, these would come from human feedback)
5. **Reward Training**: Reward model is trained to predict preferences
6. **Policy Update**: Agent is trained using the learned reward function
7. **Iteration**: Steps 2-6 repeat for multiple iterations

## Troubleshooting

### CUDA Out of Memory
- Reduce `NUM_ENVS` (number of parallel environments)
- Use `DEVICE="cpu"` for CPU-only execution (slower)

### Import Errors
```python
ImportError: No module named 'isaaclab_tasks'
```
- Ensure IsaacLab is properly installed
- Check that `PYTHONPATH` includes IsaacLab directories

### Environment Registration Issues
```python
gym.error.UnregisteredEnv: No registered env with id 'Isaac-Ant-v0'
```
- Make sure to import isaaclab_tasks: `import isaaclab_tasks`
- Check IsaacLab installation

### Slow Training
- Increase `NUM_ENVS` for better parallelization (if GPU memory allows)
- Verify GPU is being used: check CUDA availability
- Reduce `total_timesteps` or `total_comparisons` for faster iteration

## Expected Results

After training, you should see:
- Increasing episode returns as the ant learns to walk forward
- High reward accuracy (>95%) indicating good preference prediction
- Stable policy that can locomote effectively

The Isaac-Ant task requires the agent to learn:
- Forward locomotion
- Balance and stability
- Coordinated leg movement

## Comparison with Standard Environments

The main differences when using IsaacLab vs standard Gym environments:

| Feature | Standard Gym | IsaacLab |
|---------|-------------|----------|
| Vectorization | External (SubprocVecEnv) | Built-in (GPU) |
| Backend | NumPy (CPU) | PyTorch (GPU) |
| Rendering | OpenGL | Isaac Sim (photorealistic) |
| Physics | Varies | PhysX (GPU-accelerated) |
| Speed | Slower | Much faster with GPU |

## References

- **Imitation Library**: https://github.com/HumanCompatibleAI/imitation
- **IsaacLab**: https://github.com/isaac-sim/IsaacLab
- **Stable-Baselines3**: https://github.com/DLR-RM/stable-baselines3
- **Preference Learning Paper**: "Deep Reinforcement Learning from Human Preferences" (Christiano et al., 2017)

## License

This code follows the licenses of the respective libraries:
- Imitation: MIT License
- IsaacLab: BSD-3-Clause License
- Stable-Baselines3: MIT License

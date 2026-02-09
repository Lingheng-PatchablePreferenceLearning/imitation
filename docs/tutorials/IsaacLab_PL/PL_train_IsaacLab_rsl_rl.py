import random
from imitation.algorithms import preference_comparisons
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm
from imitation.util.util import make_vec_env
from imitation.policies.base import FeedForward32Policy, NormalizeFeaturesExtractor
import gymnasium as gym
from stable_baselines3 import PPO
import numpy as np

# rng = np.random.default_rng(0)
# ******************************************************
# Make vectorized environment
# *****************************************************
# # Gym environment
# # venv = make_vec_env("Pendulum-v1", rng=rng)
# venv = make_vec_env("HalfCheetah-v4", rng=rng)

# IsaacLab environment
"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import git

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip


# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--use_ground_truth_reward", action="store_true", default=False, 
                    help="Use ground truth reward for trajectory collection instead of learned reward")
parser.add_argument("--exploration_frac", type=float, default=0.05,
                    help="Fraction of trajectories to collect with exploration (default: 0.05)")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
from datetime import datetime
from rsl_rl.runners.runner import Runner
from rsl_rl.algorithms import PPO, TD3
from rsl_rl.runners.callbacks import make_first_cb, make_wandb_cb, make_interval_cb, make_save_model_cb

import wandb
# from wandb_config import WANDB_API_KEY, WANDB_ENTITY
# os.environ["WANDB_API_KEY"] = WANDB_API_KEY
LOG_WANDB = True # True (set to False during development)

from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml

import isaaclab_tasks  # noqa: F401
import isaaclab_csiro_hri.tasks
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
# from isaaclab_tasks.utils.wrappers.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_csiro_hri.rl.rsl_rl import RslRlVecEnvWrapper

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, "rsl_rl_alg_branch_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    # agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    # agent_cfg.max_iterations = (
    #     args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    # )
    print(agent_cfg)
    
    # specify directory for logging experiments 
    #   Note: we assume there is a data/ folder in the root of the git repo
    repo = git.Repo('.', search_parent_directories=True)
    git_root_path = repo.working_tree_dir    # get the root directory of the git repo
    superproject_root_path = repo.git.rev_parse("--show-superproject-working-tree") # get the root directory of the git super repo if any
    print(f"[INFO] Git root directory: {git_root_path}")
    print(f"[INFO] Git superproject root directory: {superproject_root_path}")
    log_root_path = os.path.join(superproject_root_path, "data", "logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)    # os.path.abspath will prepend the current working directory to the path
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)
    
    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    print(env.action_space)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    
    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)
    print("Breaking point")

    # set seed of the environment
    env.seed(agent_cfg.seed)

    rng = np.random.default_rng(0)

    # ******************************************************
    # Create reward network
    # ******************************************************
    # # Gym environment
    # observation_space = venv.observation_space
    # action_space = venv.action_space
    
    # IsaacLab environment
    observation_space = env.unwrapped.single_observation_space['policy']
    action_space = env.unwrapped.single_action_space

    reward_net = BasicRewardNet(
        observation_space, action_space, normalize_input_layer=RunningNorm
    )
    
    # ******************************************************
    # Set up preference comparison components
    # ******************************************************
    # 1. Fragmenter
    fragmenter = preference_comparisons.RandomFragmenter(
        warning_threshold=0,
        rng=rng,
    )
    # 2. Gatherer
    gatherer = preference_comparisons.SyntheticGatherer(rng=rng)
    # 3. Preference model and trainer
    preference_model = preference_comparisons.PreferenceModel(reward_net)
    reward_trainer = preference_comparisons.BasicRewardTrainer(
        preference_model=preference_model,
        loss=preference_comparisons.CrossEntropyRewardLoss(),
        epochs=3,
        rng=rng,
    )

    # ******************************************************
    # Create RL agent
    # ******************************************************
    # create runner from rsl-rl
    print(agent_cfg)
    env_kwargs = dict(name=args_cli.task, cfg=env_cfg)
    #
    agent = agent_cfg.alg_class(env, device=agent_cfg.device, **agent_cfg.agent_kwargs)

    config = dict(
        agent_kwargs=agent_cfg.agent_kwargs,
        env_kwargs=env_kwargs,
        runner_kwargs=agent_cfg.runner_kwargs,
    )
    wandb_learn_config = dict(
        config=config,
        # entity=WANDB_ENTITY,
        group=f"{agent_cfg.alg_class.__name__}_{args_cli.task}",
        project="rsl_rl-alg_branch",
        name=agent_cfg.run_name,
        tags=[agent_cfg.alg_class.__name__, args_cli.task, "train"],
    )
    
    runner = Runner(env, agent, log_dir=log_dir, device=agent_cfg.device, **agent_cfg.runner_kwargs)
    runner._learn_cb.append(Runner._log)    # Print log to console
    runner._learn_cb.append(make_first_cb(make_save_model_cb(log_dir)))
    runner._learn_cb.append(make_interval_cb(make_save_model_cb(log_dir), agent_cfg.save_interval))
    # runner._learn_cb = [lambda *args, **kwargs: Runner._log(*args, prefix=f"{alg_class.__name__}_{env_name}", **kwargs)]
    
    if LOG_WANDB:
        # Create wandb run once and reuse it across multiple runner.learn() calls
        wandb_run = wandb.init(**{'dir':os.path.join(superproject_root_path, "data"), **wandb_learn_config})
        
        # Create a custom callback that uses the existing run (don't let it create a new one)
        def wandb_cb(runner, stat):
            total_episodes = len(stat['returns_original'])
            current_iter = stat["current_iteration"]
            
            # Use window size that scales with number of environments
            # This ensures balanced representation: 10 episodes per environment
            episodes_per_env = 10
            recent_window_size = episodes_per_env * runner.env.num_envs
            
            # Calculate cumulative averages (all episodes - smooth curves)
            cumulative_reward = sum(stat["returns"]) / len(stat["returns"]) if len(stat["returns"]) > 0 else 0.0
            cumulative_reward_original = sum(stat["returns_original"]) / len(stat["returns_original"]) if len(stat["returns_original"]) > 0 else 0.0
            cumulative_steps = sum(stat["lengths"]) / len(stat["lengths"]) if len(stat["lengths"]) > 0 else 0.0
            
            # Calculate recent averages (last N episodes - shows recent performance)
            recent_returns = stat["returns"][-recent_window_size:] if len(stat["returns"]) > 0 else []
            recent_returns_original = stat["returns_original"][-recent_window_size:] if len(stat["returns_original"]) > 0 else []
            recent_lengths = stat["lengths"][-recent_window_size:] if len(stat["lengths"]) > 0 else []
            
            recent_reward = sum(recent_returns) / len(recent_returns) if len(recent_returns) > 0 else 0.0
            recent_reward_original = sum(recent_returns_original) / len(recent_returns_original) if len(recent_returns_original) > 0 else 0.0
            recent_steps = sum(recent_lengths) / len(recent_lengths) if len(recent_lengths) > 0 else 0.0
            
            log_dict = {}
            for stat_key, stat_item in stat.items():
                if 'returns_separate_terms_' in stat_key:
                    # For reward terms, also log both cumulative and recent
                    all_items = stat_item if len(stat_item) > 0 else []
                    recent_items = stat_item[-recent_window_size:] if len(stat_item) > 0 else []
                    log_dict['mean_{}_cumulative'.format(stat_key)] = sum(all_items) / len(all_items) if len(all_items) > 0 else 0.0
                    log_dict['mean_{}_recent'.format(stat_key)] = sum(recent_items) / len(recent_items) if len(recent_items) > 0 else 0.0
            
            total_steps = stat["current_iteration"] * runner.env.num_envs * runner._num_steps_per_env
            training_time = stat["training_time"]
            
            # Log both cumulative (smooth) and recent (responsive) metrics
            log_dict["mean_rewards_cumulative"] = cumulative_reward
            log_dict["mean_rewards_recent"] = recent_reward
            log_dict["mean_rewards_original_cumulative"] = cumulative_reward_original
            log_dict["mean_rewards_original_recent"] = recent_reward_original
            log_dict["mean_steps_cumulative"] = cumulative_steps
            log_dict["mean_steps_recent"] = recent_steps
            log_dict["training_steps"] = total_steps
            log_dict["training_time"] = training_time
            log_dict["episode_count"] = total_episodes
            
            wandb_run.log(log_dict, step=total_steps)
        
        def safe_wandb_cb(*args, **kwargs):
            try:
                return wandb_cb(*args, **kwargs)
            except Exception as e:
                print(f"[WARNING] WandB logging failed: {e}")
                raise ValueError("Stopping execution due to WandB logging failure.")
        
        runner._learn_cb.append(safe_wandb_cb)
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # save resume path before creating a new log_dir
    if agent_cfg.resume:
        # get path to previous checkpoint
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # 
    agent = runner
    venv = env
    
    # Choose reward function for trajectory collection
    if args_cli.use_ground_truth_reward:
        print("[INFO] Using GROUND TRUTH reward for trajectory collection")
        # Use None to keep original environment rewards (ground truth)
        trajectory_reward_fn = None
    else:
        print("[INFO] Using LEARNED reward for trajectory collection")
        # Use the learned reward network
        trajectory_reward_fn = reward_net
    
    trajectory_generator = preference_comparisons.AgentTrainer(
        algorithm=agent,
        reward_fn=trajectory_reward_fn,
        venv=venv,
        exploration_frac=args_cli.exploration_frac,
        rng=rng,
    )
    print(f"[INFO] Using exploration_frac={args_cli.exploration_frac} ({args_cli.exploration_frac*100:.1f}% of trajectories with exploration)")
    
    pref_comparisons = preference_comparisons.PreferenceComparisons(
        trajectory_generator,
        reward_net,
        num_iterations= 60,  # Set to 60 for better performance
        fragmenter=fragmenter,
        preference_gatherer=gatherer,
        reward_trainer=reward_trainer,
        fragment_length=100,
        transition_oversampling=1,
        initial_comparison_frac=0.01, # 0.1,
        allow_variable_horizon=True,  # TODO: WARNING: variable horizon episodes leak information about the reward via termination condition, and can seriously confound evaluation.
        initial_epoch_multiplier=4,
        query_schedule="hyperbolic",
    )

    # The steps and comparisions is shceduled for num_iterations preference learning iterations.
    #   e.g., with num_iterations=5, total_timesteps=50_000, total_comparisons=2000,
    #   each preference learning iteration will have approximately:
    #       400 comparisons (2000 / 5)      # Depends on the initial_comparison_frac and query_schedule
    #       400 comparison x 2 fragments x 100 fragment_length = 80000 timesteps (including agent steps and exploration steps)
    #       10_000 timesteps (50_000 / 5)   # RL agent training timesteps per iteration
    #   
    # Note: to test code set total_timesteps and total_comparisons to small values
    pref_comparisons.train(
        total_timesteps=1000*60, # 50_000, (training time step for RL is total_timesteps / num_iterations)
        total_comparisons= 12000, # 2000,
    )

    # Finish wandb run after all training is complete
    if LOG_WANDB:
        wandb_run.finish()
    import pdb; pdb.set_trace()
    # **************************************************************
    # Train an RL agent on the learned reward
    # **************************************************************

    from imitation.rewards.reward_wrapper import RewardVecEnvWrapper

    learned_reward_venv = RewardVecEnvWrapper(venv, reward_net.predict_processed)

    learner = PPO(
        seed=0,
        policy=FeedForward32Policy,
        policy_kwargs=dict(
            features_extractor_class=NormalizeFeaturesExtractor,
            features_extractor_kwargs=dict(normalize_class=RunningNorm),
        ),
        env=learned_reward_venv,
        batch_size=64,
        ent_coef=0.01,
        n_epochs=10,
        n_steps=2048 // learned_reward_venv.num_envs,
        clip_range=0.1,
        gae_lambda=0.95,
        gamma=0.97,
        learning_rate=2e-3,
    )
    learner.learn(100_000)  # Note: set to 100_000 to train a proficient expert

    # # *************************************************************
    # agent = agent_cfg.alg_class(learned_reward_venv, device=agent_cfg.device, **agent_cfg.agent_kwargs)
    # config = dict(
    #     agent_kwargs=agent_cfg.agent_kwargs,
    #     env_kwargs=env_kwargs,
    #     runner_kwargs=agent_cfg.runner_kwargs,
    # )
    # wandb_learn_config = dict(
    #     config=config,
    #     # entity=WANDB_ENTITY,
    #     group=f"{agent_cfg.alg_class.__name__}_{args_cli.task}",
    #     project="rsl_rl-alg_branch",
    #     name="{}_{}".format(agent_cfg.run_name, 'learner'),
    #     tags=[agent_cfg.alg_class.__name__, args_cli.task, "train"],
    # )
    
    # learner = Runner(learned_reward_venv, agent, log_dir=log_dir, device=agent_cfg.device, **agent_cfg.runner_kwargs)
    # learner._learn_cb.append(Runner._log)    # Print log to console
    # learner._learn_cb.append(make_first_cb(make_save_model_cb(log_dir)))
    # learner._learn_cb.append(make_interval_cb(make_save_model_cb(log_dir), agent_cfg.save_interval))
    # # runner._learn_cb = [lambda *args, **kwargs: Runner._log(*args, prefix=f"{alg_class.__name__}_{env_name}", **kwargs)]
    
    # if LOG_WANDB:
    #     # Wrap wandb callback with error handling to prevent crashes if wandb backend fails
    #     wandb_cb = make_wandb_cb({'dir':os.path.join(superproject_root_path, "data"), **wandb_learn_config})
    #     def safe_wandb_cb(*args, **kwargs):
    #         try:
    #             return wandb_cb(*args, **kwargs)
    #         except Exception as e:
    #             print(f"[WARNING] WandB logging failed: {e}")
    #             return None
    #     learner._learn_cb.append(safe_wandb_cb)    # The default wandb logging directory is ./wandb
    # # write git state to logs
    # learner.add_git_repo_to_log(__file__)
    # # save resume path before creating a new log_dir
    # if agent_cfg.resume:
    #     # get path to previous checkpoint
    #     resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    #     print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    #     # load previously trained model
    #     learner.load(resume_path)

    # # run training
    # learner.learn(iterations=1000000)

    import pdb; pdb.set_trace()
    # *************************************************************
    # Evaluate the learned policy
    # *************************************************************
    from stable_baselines3.common.evaluation import evaluate_policy

    n_eval_episodes = 10
    reward_mean, reward_std = evaluate_policy(learner.policy, venv, n_eval_episodes)
    reward_stderr = reward_std / np.sqrt(n_eval_episodes)
    print(f"Reward: {reward_mean:.0f} +/- {reward_stderr:.0f}")
    



if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
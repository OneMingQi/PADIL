# Standard library imports
import argparse
import inspect
import os
import pickle
import random
import sys
from pathlib import Path

# Third-party library imports
import gym
import numpy as np
import yaml
import torch

# Add the parent directory of the current file to the system path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Custom module imports
from algorithms.rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from algorithms.rlkit.envs import get_env, get_envs
from algorithms.rlkit.envs.wrappers import (
    ProxyEnv,
    ScaledEnv,
    MinmaxEnv,
    NormalizedBoxEnv,
    EPS,
)
from algorithms.rlkit.launchers.launcher_util import setup_logger, set_seed
from algorithms.rlkit.torch.algorithms.adv_irl.adv_irl import AdvIRL
from algorithms.rlkit.torch.algorithms.adv_irl.disc_models.simple_disc_models import (
    MLPDisc,
)
from algorithms.rlkit.torch.algorithms.sac.sac_alpha import SoftActorCritic
from algorithms.rlkit.torch.common.networks import FlattenMlp
from algorithms.rlkit.torch.common.policies import ReparamTanhMultivariateGaussianPolicy
import algorithms.rlkit.torch.utils.pytorch_util as ptu


def experiment(variant):
    # Load demo listings from YAML file
    with open("experts/expert_data.yaml", "r") as file:
        listings = yaml.safe_load(file.read())

    # Get the path of the demo file
    demos_path = listings[variant["expert_name"]]["file_paths"][variant["expert_idx"]]
    print("demos_path", demos_path)

    # Load the demo data from the file
    with open(demos_path, "rb") as f:
        traj_list = pickle.load(f)

    # Randomly sample a number of trajectories
    traj_list = random.sample(traj_list, variant["traj_num"])

    # Stack the observations from all trajectories
    obs = np.vstack([traj_list[i]["observations"] for i in range(len(traj_list))])

    # Calculate the mean and standard deviation of the observations
    obs_mean, obs_std = np.mean(obs, axis=0), np.std(obs, axis=0)

    # Calculate the minimum and maximum of the observations
    obs_min, obs_max = np.min(obs, axis=0), np.max(obs, axis=0)

    # Get the environment specifications
    env_specs = variant["env_specs"]

    # Create the environment
    env = get_env(env_specs)
    env.seed(env_specs["eval_env_seed"])

    # Print some information about the environment
    print(f"\n\nEnvironment:")
    print(f"Name: {env_specs['env_name']}")
    print(f"Parameters: {env_specs['env_kwargs']}")
    print(f"Observation Space: {env.observation_space}")
    print(f"Action Space: {env.action_space}\n\n")

    # Create the replay buffer
    expert_replay_buffer = EnvReplayBuffer(
        variant["adv_irl_params"]["replay_buffer_size"],
        env,
        random_seed=np.random.randint(10000),
    )

    # Set the environment wrapper
    tmp_env_wrapper = env_wrapper = ProxyEnv  # Identical wrapper
    kwargs = {}
    wrapper_kwargs = {}

    # Check if the environment should be scaled with demo stats
    if variant["scale_env_with_demo_stats"]:
        print("\nWARNING: Using scale env wrapper")

        # Set the environment wrapper to ScaledEnv
        tmp_env_wrapper = env_wrapper = ScaledEnv

        # Set the wrapper arguments
        wrapper_kwargs = dict(
            obs_mean=obs_mean,
            obs_std=obs_std,
            acts_mean=None,
            acts_std=None,
        )

        # Scale the observations in each trajectory
        for i in range(len(traj_list)):
            traj_list[i]["observations"] = (traj_list[i]["observations"] - obs_mean) / (
                obs_std + EPS
            )
            traj_list[i]["next_observations"] = (
                traj_list[i]["next_observations"] - obs_mean
            ) / (obs_std + EPS)

    # Check if the environment should be min-max normalized with demo stats
    elif variant["minmax_env_with_demo_stats"]:
        print("\nWARNING: Using min max env wrapper")

        # Set the environment wrapper to MinmaxEnv
        tmp_env_wrapper = env_wrapper = MinmaxEnv

        # Set the wrapper arguments
        wrapper_kwargs = dict(obs_min=obs_min, obs_max=obs_max)

        # Normalize the observations in each trajectory
        for i in range(len(traj_list)):
            traj_list[i]["observations"] = (traj_list[i]["observations"] - obs_min) / (
                obs_max - obs_min + EPS
            )
            traj_list[i]["next_observations"] = (
                traj_list[i]["next_observations"] - obs_min
            ) / (obs_max - obs_min + EPS)

    # Get the observation space and action space of the environment
    obs_space = env.observation_space
    act_space = env.action_space

    # Assert that the observation space is not a dictionary and has only one dimension
    assert not isinstance(obs_space, gym.spaces.Dict)
    assert len(obs_space.shape) == 1

    # Assert that the action space has only one dimension
    assert len(act_space.shape) == 1

    # Wrap the environment
    env = env_wrapper(env, **wrapper_kwargs)

    # Create the training environment
    training_env = get_envs(
        env_specs, env_wrapper, wrapper_kwargs=wrapper_kwargs, **kwargs
    )
    training_env.seed(env_specs["training_env_seed"])

    # Add the trajectories to the expert replay buffer
    for i in range(len(traj_list)):
        expert_replay_buffer.add_path(
            traj_list[i], absorbing=variant["adv_irl_params"]["wrap_absorbing"], env=env
        )

    # Get the dimensions of the observation space and action space
    obs_dim = obs_space.shape[0]
    action_dim = act_space.shape[0]

    # Build the Q functions and policy
    net_size = variant["policy_net_size"]
    num_hidden = variant["policy_num_hidden_layers"]

    qf1 = FlattenMlp(
        hidden_sizes=num_hidden * [net_size],
        input_size=obs_dim + action_dim,
        output_size=1,
    )
    qf2 = FlattenMlp(
        hidden_sizes=num_hidden * [net_size],
        input_size=obs_dim + action_dim,
        output_size=1,
    )
    policy = ReparamTanhMultivariateGaussianPolicy(
        hidden_sizes=num_hidden * [net_size],
        obs_dim=obs_dim,
        action_dim=action_dim,
    )

    # Adjust the input dimension if the environment is absorbing
    if variant["adv_irl_params"]["wrap_absorbing"]:
        obs_dim += 1
    input_dim = obs_dim + action_dim
    if variant["adv_irl_params"]["state_only"]:
        input_dim = obs_dim + obs_dim

    # Build the discriminator model
    disc_model = MLPDisc(
        input_dim,
        num_layer_blocks=variant["disc_num_blocks"],
        hid_dim=variant["disc_hid_dim"],
        hid_act=variant["disc_hid_act"],
        use_bn=variant["disc_use_bn"],
        clamp_magnitude=variant["disc_clamp_magnitude"],
    )
    # Set up the Soft Actor-Critic (SAC) trainer
    trainer = SoftActorCritic(
        policy=policy, qf1=qf1, qf2=qf2, env=env, **variant["sac_params"]
    )

    # Create an instance of the Adversarial Inverse Reinforcement Learning (AdvIRL) algorithm
    algorithm = AdvIRL(
        env=env,
        training_env=training_env,
        exploration_policy=policy,
        discriminator=disc_model,
        policy_trainer=trainer,
        expert_replay_buffer=expert_replay_buffer,
        **variant["adv_irl_params"],
    )

    # If GPU is enabled, move the algorithm to the GPU device
    if ptu.gpu_enabled():
        algorithm.to(ptu.device)

    # Train the algorithm
    algorithm.train()

    # Ensure the directory exists
    os.makedirs("saved", exist_ok=True)

    filename = f'saved/zm_{variant["exp_name"]}_{variant["seed"]}_{variant["traj_num"]}_disc.yaml'
    torch.save(disc_model.state_dict(), filename)

    filename = f'saved/zm_{variant["exp_name"]}_{variant["seed"]}_{variant["traj_num"]}_policy.yaml'
    torch.save(policy.state_dict(), filename)

    # Return a signal indicating that the training is done
    signal = "Done"

    return signal


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiment", help="experiment specification file")
    parser.add_argument("-g", "--gpu", help="gpu id", type=int, default=0)
    args = parser.parse_args()

    # Load experiment specifications from the file
    print(args.experiment)
    with open(args.experiment, "r") as spec_file:
        spec_string = spec_file.read()
        exp_specs = yaml.safe_load(spec_string)

    # Set all seeds to the same value
    exp_specs["env_specs"]["eval_env_seed"] = exp_specs["env_specs"][
        "training_env_seed"
    ] = exp_specs["seed"]

    # Build the experiment suffix based on the experiment specifications
    exp_suffix = "--gp-{}--rs-{}--trajnum-{}".format(
        exp_specs["adv_irl_params"]["grad_pen_weight"],
        exp_specs["sac_params"]["reward_scale"],
        format(exp_specs["traj_num"]),
    )
    if not exp_specs["adv_irl_params"]["no_terminal"]:
        exp_suffix = "--terminal" + exp_suffix
    if exp_specs["adv_irl_params"]["wrap_absorbing"]:
        exp_suffix = "--absorbing" + exp_suffix
    if exp_specs["scale_env_with_demo_stats"]:
        exp_suffix = "--scale" + exp_suffix

    # If GPU is enabled, set GPU mode
    if exp_specs["using_gpus"] > 0:
        print("\n\nUSING GPU\n\n")
        ptu.set_gpu_mode(True, args.gpu)

    # Set up the logger
    exp_id = exp_specs["exp_id"]
    exp_prefix = exp_specs["exp_name"] + exp_suffix
    seed = exp_specs["seed"]
    set_seed(seed)
    setup_logger(exp_prefix=exp_prefix, exp_id=exp_id, variant=exp_specs, seed=seed)

    # Run experiment
    signal = experiment(exp_specs)
    print(signal)

import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from algorithms.rlkit.policies.base import ExplorationPolicy, Policy
from algorithms.rlkit.torch.common.distributions import (
    ReparamTanhMultivariateNormal,
    ReparamMultivariateNormalDiag,
)
from algorithms.rlkit.torch.common.networks import Mlp
from algorithms.rlkit.torch.core import PyTorchModule
from algorithms.rlkit.torch.utils.pytorch_util import device
from algorithms.rlkit.torch.utils.transform_layer import hsv2rgb
from diffusion.diffusion import Diffusion_conditional
from diffusion.utils import EMA


class diffusion_actor_policy(Diffusion_conditional):
    def __init__(
        self,
        policy_args,
        state_dim: int,
        action_dim,
        action_space,
        device: torch.device,
        diffusion_steps_needed=False,
        diffusion_n_timesteps=10,
    ):
        self.diffusion_steps_needed = diffusion_steps_needed
        self.device = device
        self.noise_ratio = policy_args["noise_ratio"]
        self.beta_schedule = policy_args["beta_schedule"]
        self.n_timesteps = diffusion_n_timesteps

        if action_space is None:
            self.action_scale = 1.0
            self.action_bias = 0.0
        else:
            self.action_scale = (action_space.high - action_space.low) / 2.0
            self.action_bias = (action_space.high + action_space.low) / 2.0

        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            noise_ratio=self.noise_ratio,
            beta_schedule=self.beta_schedule,
            n_timesteps=self.n_timesteps,
            diffusion_steps_needed=self.diffusion_steps_needed,
        )

        self.to(self.device)

    def get_action(self, state: np.ndarray, eval: bool = False) -> np.ndarray:
        """
        Samples an action based on the given state.

        Args:
            state: The state to sample the action from.
            eval: A boolean indicating whether the model is in evaluation mode.

        Returns:
            The sampled action.
        """
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)

        if self.diffusion_steps_needed == False:
            action = self(state, eval).cpu().data.numpy().flatten()
            action = action.clip(-1, 1)
            action = action * self.action_scale + self.action_bias
            return action
        else:
            action_tensor, _ = self(state, eval)
            action = action_tensor.cpu().data.numpy().flatten()
            action = action.clip(-1, 1)
            action = action * self.action_scale + self.action_bias
            return action

    def get_actions(self, states: np.ndarray, eval: bool = False) -> np.ndarray:
        """
        Samples actions based on the given states. With Diffusion steps.

        Args:
            states: The states to sample the actions from. Shape should be (batch_size, state_dim).
            eval: A boolean indicating whether the model is in evaluation mode.

        Returns:
            The sampled actions.
        """
        states = torch.FloatTensor(states).to(self.device)

        if self.diffusion_steps_needed == False:
            actions = self(states, eval).cpu().data.numpy()
            actions = actions.clip(-1, 1)
            actions = actions * self.action_scale + self.action_bias

            return actions

        else:
            actions_tensor, diffusion_actions_tensor = self(states, eval)

            actions = actions_tensor.cpu().data.numpy()
            diffusion_actions = diffusion_actions_tensor.cpu().data.numpy()

            actions = actions.clip(-1, 1)
            actions = actions * self.action_scale + self.action_bias

            return actions, diffusion_actions

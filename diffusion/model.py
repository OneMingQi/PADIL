import random
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion.utils import SinusoidalPosEmb, init_weights
from algorithms.rlkit.data_management.data_augmentation import random_rotation
from algorithms.rlkit.torch.utils.pytorch_util import rand


class MLP_base(nn.Module):
    """
    Base MLP Model.
    """

    def __init__(self, time_dim, hidden_size):
        super(MLP_base, self).__init__()

        self.time_mlp = self._build_time_mlp(time_dim, hidden_size)

    def _build_time_mlp(self, time_dim: int, hidden_size: int) -> nn.Module:
        """
        Build a Sequential model for time MLP with the given time dimension and hidden size.
        """
        return nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, hidden_size),
            nn.Mish(),
            nn.Linear(hidden_size, time_dim),
        )


class MLP_state_action(MLP_base):
    """
    The Model class that represents the main model.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_size: int = 256,
        time_dim: int = 32,
        rand: bool = False,
    ):
        """
        Initialize the Model with the given state dimension, action dimension, hidden size, and time dimension.
        """
        super(MLP_state_action, self).__init__(time_dim, hidden_size)

        self.rand = rand

        input_dim = state_dim + action_dim + time_dim
        self.layer = self._build_layer(input_dim, hidden_size, action_dim)
        self.apply(init_weights)

    def _build_layer(
        self, input_dim: int, hidden_size: int, action_dim: int
    ) -> nn.Module:
        """
        Build a Sequential model for the main layer with the given input dimension, hidden size, and action dimension.
        """
        return nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.Mish(),
            nn.Linear(hidden_size, hidden_size),
            nn.Mish(),
            nn.Linear(hidden_size, hidden_size),
            nn.Mish(),
            nn.Linear(hidden_size, action_dim),
        )

    def forward(
        self, x: torch.Tensor, time: torch.Tensor, state: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through the model. Returns the output tensor.
        """
        t = self.time_mlp(time)
        out = torch.cat([x, t, state], dim=-1)
        out = self.layer(out)

        if self.rand == True:
            out += torch.randn_like(out) * 0.1

        return out


class MLP_x_time(MLP_base):
    """
    MLP Model for x-time.
    """

    def __init__(self, x_dim, hid_dim, device, t_dim=16, rand=False):
        super(MLP_x_time, self).__init__(t_dim, hid_dim)
        self.device = device

        self.mid_layer = self._build_mid_layer(x_dim + t_dim, hid_dim)
        self.final_layer = nn.Linear(hid_dim, x_dim)
        self.rand = rand

        self.apply(init_weights)

    def _build_mid_layer(self, input_dim: int, hidden_size: int) -> nn.Module:
        """
        Build a Sequential model for the middle layer with the given input dimension and hidden size.
        """
        return nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.Mish(),
            nn.Linear(hidden_size, hidden_size),
            nn.Mish(),
            nn.Linear(hidden_size, hidden_size),
            nn.Mish(),
        )

    def forward(self, x, time):
        t = self.time_mlp(time)
        x = torch.cat([x, t], dim=1)

        out = self.mid_layer(x)
        out = self.final_layer(out)

        if self.rand == True:
            out += torch.randn_like(out) * 0.1

        return out

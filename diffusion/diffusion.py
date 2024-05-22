import copy
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion.model import MLP_state_action
from diffusion.utils import (
    cosine_beta_schedule,
    linear_beta_schedule,
    vp_beta_schedule,
    extract,
    Losses,
    Progress,
    Silent,
)
from algorithms.rlkit.torch.utils.pytorch_util import device


class Diffusion_base(nn.Module):
    """Base class for the diffusion process."""

    def __init__(
        self,
        x_dim: int,
        action_dim: int,
        model: nn.Module,
        beta_schedule: str = "linear",
        n_timesteps: int = 100,
        loss_type: str = "l2",
        clip_denoised: bool = True,
        predict_epsilon: bool = True,
        diffusion_steps_needed=False,
    ):
        """
        Initialize the Diffusion module.

        Args:
            x_dim: The dimension of the state.
            action_dim: The dimension of the action.
            model: The model to use for the diffusion process.
            beta_schedule: The beta schedule to use. Default is "linear".
            n_timesteps: The number of timesteps. Default is 100.
            loss_type: The type of loss to use. Default is "l2".
            clip_denoised: Whether to clip the denoised value. Default is True.
            predict_epsilon: Whether to predict epsilon. Default is True.
        """
        super(Diffusion_base, self).__init__()

        self.x_dim = x_dim
        self.action_dim = action_dim
        self.model = model
        self.diffusion_steps_needed = diffusion_steps_needed

        # Get betas and calculate alphas
        betas = self.get_betas(beta_schedule, n_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        # Register buffers for diffusion calculations
        self.register_diffusion_buffers(alphas_cumprod, alphas_cumprod_prev, betas)

        # Register buffers for posterior calculations
        self.register_posterior_buffers(
            alphas_cumprod, alphas_cumprod_prev, betas, alphas
        )

        # Initialize loss function
        self.loss_fn = Losses[loss_type]()

    def register_diffusion_buffers(self, alphas_cumprod, alphas_cumprod_prev, betas):
        """Register buffers needed for the diffusion calculations."""
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1)
        )

    def register_posterior_buffers(
        self, alphas_cumprod, alphas_cumprod_prev, betas, alphas
    ):
        """Register buffers needed for the posterior calculations."""
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(torch.clamp(posterior_variance, min=1e-20)),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

    def get_betas(self, beta_schedule: str, n_timesteps: int) -> torch.Tensor:
        """
        Get the betas based on the beta_schedule.

        Args:
            beta_schedule: The schedule to use for beta. Can be "linear", "cosine", or "vp".
            n_timesteps: The number of timesteps.

        Returns:
            A tensor of betas.

        Raises:
            ValueError: If beta_schedule is not "linear", "cosine", or "vp".
        """
        beta_schedule_func = {
            "linear": linear_beta_schedule,
            "cosine": cosine_beta_schedule,
            "vp": vp_beta_schedule,
        }.get(beta_schedule)

        if beta_schedule_func is None:
            raise ValueError(f"Unknown beta_schedule: {beta_schedule}")

        return beta_schedule_func(n_timesteps)

    def predict_start_from_noise(
        self, x_t: torch.Tensor, t: int, noise: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict the start from noise.
        If self.predict_epsilon, model output is (scaled) noise;
        otherwise, model predicts x0 directly.

        Args:
            x_t: The current state.
            t: The current timestep.
            noise: The noise tensor.

        Returns:
            The predicted start state.
        """
        if self.predict_epsilon:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def p_mean_variance(self, x, t, s):
        x_recon = self.predict_start_from_noise(x, t=t, noise=self.model(x, t, s))

        if self.clip_denoised:
            x_recon.clamp_(-1.0, 1.0)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance

    def q_posterior(
        self, x_start: torch.Tensor, x_t: torch.Tensor, t: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculate the posterior mean, variance, and log variance.

        Args:
            x_start: The start state.
            x_t: The current state.
            t: The current timestep.

        Returns:
            The posterior mean, variance, and log variance.
        """
        # Calculate the posterior mean using the start state and the current state
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )

        # Extract the posterior variance for the current timestep
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)

        # Extract the clipped posterior log variance for the current timestep
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )

        # Return the posterior mean, variance, and log variance
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    @torch.no_grad()
    def p_sample(
        self, x: torch.Tensor, t: torch.Tensor, s: torch.Tensor
    ) -> torch.Tensor:
        """
        Sample from the model distribution.

        Args:
            x: The current state.
            t: The current timestep.
            s: The current action.

        Returns:
            The sampled state.
        """
        b, *_, device = *x.shape, x.device

        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, s=s)

        noise = torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))

        return (
            model_mean
            + nonzero_mask * (0.5 * model_log_variance).exp() * noise * self.noise_ratio
        )

    @torch.no_grad()
    def p_sample_loop(
        self, state: torch.Tensor, shape: Tuple[int, int]
    ) -> torch.Tensor:
        """
        Sample from the model distribution in a loop.

        Args:
            state: The current state.
            shape: The shape of the sample.

        Returns:
            The sampled state.
        """
        device = self.betas.device

        batch_size = shape[0]
        x = torch.randn(shape, device=device)

        if self.diffusion_steps_needed == False:
            for i in reversed(range(0, self.n_timesteps)):
                timesteps = torch.full(
                    (batch_size,), i, device=device, dtype=torch.long
                )
                x = self.p_sample(x, timesteps, state)

            return x

        else:
            diffusion_actions = torch.full(
                (batch_size, self.n_timesteps, shape[1]),
                0.0,
                device=device,
                dtype=torch.long,
            )

            for i in reversed(range(0, self.n_timesteps)):
                timesteps = torch.full(
                    (batch_size,), i, device=device, dtype=torch.long
                )
                x = self.p_sample(x, timesteps, state)
                diffusion_actions[:, i, :] = x

            return x, diffusion_actions

    def q_sample(self, x_start, t, noise=None):
        """
        Sample from the Q distribution.

        Args:
            x_start: The start state.
            t: The current timestep.
            noise: The noise tensor. If None, random noise is used.

        Returns:
            The sampled state.
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )


class Diffusion_conditional(Diffusion_base):
    """A module for the diffusion process."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        noise_ratio: float,
        beta_schedule: str = "vp",
        n_timesteps: int = 100,
        loss_type: str = "l2",
        clip_denoised: bool = True,
        predict_epsilon: bool = True,
        max_action: float = -1.0,
        diffusion_steps_needed=False,
    ):
        """
        Initialize the Diffusion module.

        Args:
            state_dim: The dimension of the state.
            action_dim: The dimension of the action.
            noise_ratio: The ratio of noise to add to the state.
            beta_schedule: The beta schedule to use. Default is "vp".
            n_timesteps: The number of timesteps. Default is 1000.
            loss_type: The type of loss to use. Default is "l2".
            clip_denoised: Whether to clip the denoised value. Default is True.
            predict_epsilon: Whether to predict epsilon. Default is True.
        """
        super().__init__(
            x_dim=state_dim,
            action_dim=action_dim,
            model=MLP_state_action(state_dim, action_dim),
            beta_schedule=beta_schedule,
            n_timesteps=n_timesteps,
            loss_type=loss_type,
            clip_denoised=clip_denoised,
            predict_epsilon=predict_epsilon,
            diffusion_steps_needed=diffusion_steps_needed,
        )

        self.diffusion_steps_needed = diffusion_steps_needed
        self.max_noise_ratio = noise_ratio
        self.noise_ratio = noise_ratio
        self.max_action = max_action

    @torch.no_grad()
    def sample(self, state: torch.Tensor, eval: bool = False) -> torch.Tensor:
        """
        Sample an action from the model distribution.

        Args:
            state: The current state.
            eval: Whether to evaluate the model. If True, no noise is added.

        Returns:
            A tensor representing the sampled action.
        """
        # Set the noise ratio based on whether we are evaluating the model or not
        self.noise_ratio = 0 if eval else self.max_noise_ratio

        # Get the batch size from the state tensor
        batch_size = state.shape[0]
        # Create a tensor of shape (batch_size, self.action_dim)
        shape = (batch_size, self.action_dim)

        if self.diffusion_steps_needed == False:
            # Generate an action using the p_sample_loop method
            action = self.p_sample_loop(state, shape)

            # Clamp the action values between -1.0 and 1.0 if self.max_action is -1.0
            # Otherwise, clamp the action values between -self.max_action and self.max_action
            if self.max_action == -1.0:
                return action.clamp_(-1.0, 1.0)
            else:
                return action.clamp_(-self.max_action, self.max_action)

        else:
            # Generate an action and a series of diffusion actions using the p_sample_loop method
            action, diffusion_actions = self.p_sample_loop(state, shape)

            # Clamp the action and diffusion action values between -1.0 and 1.0 if self.max_action is -1.0
            # Otherwise, clamp the action and diffusion action values between -self.max_action and self.max_action
            if self.max_action == -1.0:
                return (
                    action.clamp_(-1.0, 1.0),
                    diffusion_actions,
                )
            else:
                return (
                    action.clamp_(-self.max_action, self.max_action),
                    diffusion_actions,
                )

    def forward(self, state: torch.Tensor, eval: bool = False) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            state: The current state.
            eval: Whether to evaluate the model. If True, no noise is added.

        Returns:
            The result of the sample method.
        """
        return self.sample(state, eval)

    def p_losses(
        self,
        x_start: torch.Tensor,
        state: torch.Tensor,
        t: int,
        weights: float = 1.0,
        disc_ddpm: bool = False,
    ) -> torch.Tensor:
        """
        Calculate the losses for the P distribution.

        Args:
            x_start: The start state.
            state: The current state.
            t: The current timestep.
            weights: The weights for the loss. Default is 1.0.

        Returns:
            The calculated loss.
        """
        # Generate random noise with the same shape as x_start
        noise = torch.randn_like(x_start)

        # Sample from the Q distribution
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        # Pass the noisy sample through the model
        x_recon = self.model(x_noisy, t, state)

        # Ensure the noise and reconstructed sample have the same shape
        assert noise.shape == x_recon.shape

        # Calculate the loss
        if self.predict_epsilon:
            if not disc_ddpm:
                loss = self.loss_fn(x_recon, noise, weights)
            else:
                l2_ = torch.mean(torch.pow((x_recon - noise), 2), dim=1)
                loss = torch.exp(-l2_)
        else:
            loss = self.loss_fn(x_recon, x_start, weights)

        return loss

    def loss(
        self,
        x: torch.Tensor,
        state: torch.Tensor,
        weights: float = 1.0,
        disc_ddpm: bool = False,
    ) -> torch.Tensor:
        """
        Calculate the total loss.

        Args:
            x: The current state.
            state: The current state.
            weights: The weights for the loss. Default is 1.0.

        Returns:
            The total loss.
        """
        # Get the batch size and generate random timesteps
        batch_size = len(x)

        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()

        # Calculate the losses for the P distribution
        return self.p_losses(x, state, t, weights, disc_ddpm)

    def set_num_steps_total(self, t):
        pass

    def calc_reward(self, batch_state, batch_action) -> torch.Tensor:
        device = batch_action.device
        batch_size = batch_action.shape[0]

        vb = []
        for t in list(range(0, self.n_timesteps))[::-1]:
            t_batch = torch.tensor([t] * batch_size, device=device)
            noise = torch.randn_like(batch_action)
            x_t = self.q_sample(x_start=batch_action, t=t_batch, noise=noise)

            with torch.no_grad():
                x_recon = self.model(x=x_t, time=t_batch, state=batch_state)
                l2_ = torch.pow((x_recon - noise), 2)
                loss_t = torch.exp(-l2_)
            vb.append(loss_t)

        vb = torch.stack(vb, dim=1)
        disc_cost1 = vb.sum(dim=1) / (self.n_timesteps)
        disc_cost1 = disc_cost1.sum(dim=1) / (batch_action.shape[1])
        return disc_cost1

import numpy as np
from collections import OrderedDict

import torch
import torch.optim as optim
from torch import nn
from torch import autograd
import torch.nn.functional as F

import algorithms.rlkit.torch.utils.pytorch_util as ptu
from algorithms.rlkit.torch.core import np_to_pytorch_batch
from algorithms.rlkit.torch.algorithms.torch_base_algorithm import TorchBaseAlgorithm
from algorithms.rlkit.torch.algorithms.adv_irl.utility.bypass_bn import (
    enable_running_stats,
    disable_running_stats,
)
from algorithms.rlkit.core.vistools import plot_2dhistogram
from diffusion.model import MLP_state_action
from discriminator.GDAIL_Disc import GDAIL_Disc
import random


class AdvIRL(TorchBaseAlgorithm):
    """
    The AdvIRL class inherits from the TorchBaseAlgorithm class.

    This class is used to implement the Adversarial Inverse Reinforcement Learning (AdvIRL) algorithm. Depending on the choice of reward function and size of the replay buffer, this class can be used to implement various versions of the AdvIRL algorithm such as Generative Adversarial Imitation Learning (GAIL), Discriminator Actor Critic (DAC), and Diffusion Discriminator Policy Adversarial Imitation Learning (DDPAIL).

    The AdvIRL class takes in parameters such as the mode of operation, the discriminator network, the policy trainer, and the expert replay buffer, among others. It also includes methods for training the model and updating the policy and discriminator.
    """

    def __init__(
        self,
        mode,
        discriminator,
        policy_trainer,
        expert_replay_buffer,
        state_only=False,
        disc_optim_batch_size=1024,
        policy_optim_batch_size=1024,
        policy_optim_batch_size_from_expert=0,
        num_update_loops_per_train_call=1,
        num_disc_updates_per_loop_iter=100,
        num_policy_updates_per_loop_iter=50,
        disc_lr=1e-3,
        disc_momentum=0.0,
        disc_optimizer_class=optim.Adam,
        use_grad_pen=True,
        grad_pen_weight=10,
        rew_clip_min=None,
        rew_clip_max=None,
        disc_ddpm=False,
        diffusion_n_timesteps=10,
        diffusion_steps_needed=False,
        **kwargs,
    ):
        """
        Initialize the AdvIRL class.

        Args:
            mode (str): The mode of the algorithm, could be 'ddpail_common', 'ddpail', 'gail', or 'ddpm'.
            discriminator (nn.Module): The discriminator network.
            policy_trainer (nn.Module): The policy trainer.
            expert_replay_buffer (ReplayBuffer): The replay buffer that stores the expert's data.
            state_only (bool): Whether to use only the state for the discriminator.
            disc_optim_batch_size (int): The batch size for the discriminator optimizer.
            policy_optim_batch_size (int): The batch size for the policy optimizer.
            policy_optim_batch_size_from_expert (int): The batch size for the policy optimizer from the expert's data.
            num_update_loops_per_train_call (int): The number of update loops per train call.
            num_disc_updates_per_loop_iter (int): The number of discriminator updates per loop iteration.
            num_policy_updates_per_loop_iter (int): The number of policy updates per loop iteration.
            disc_lr (float): The learning rate for the discriminator optimizer.
            disc_momentum (float): The momentum for the discriminator optimizer.
            disc_optimizer_class (torch.optim.Optimizer): The class of the discriminator optimizer.
            use_grad_pen (bool): Whether to use gradient penalty.
            grad_pen_weight (float): The weight for the gradient penalty.
            rew_clip_min (float): The minimum reward clip value.
            rew_clip_max (float): The maximum reward clip value.
            disc_ddpm (bool): Whether to use ddpm for the discriminator.
            kwargs: Other arguments.
        """
        # Define modes and their descriptions.
        modes = {
            "GAIL": "Generative Adversarial Imitation Learning with Soft Actor-Critic",
            "PADIL": "Process Adversarial Diffusion Imitation Learning",
        }

        # Check if the mode is valid.
        assert (
            mode in modes
        ), f"Invalid mode '{mode}'. Available modes are:\n" + "\n".join(
            f"{key}: {value}" for key, value in modes.items()
        )
        self.diffusion_steps_needed = diffusion_steps_needed

        super().__init__(
            diffusion_n_timesteps=diffusion_n_timesteps,
            diffusion_steps_needed=diffusion_steps_needed,
            **kwargs,
        )
        # Initialize the class variables
        self.mode = mode
        self.diffusion_n_timesteps = diffusion_n_timesteps
        self.disc_ddpm = disc_ddpm
        self.state_only = state_only
        self.expert_replay_buffer = expert_replay_buffer
        self.policy_trainer = policy_trainer
        self.policy_optim_batch_size = policy_optim_batch_size
        self.policy_optim_batch_size_from_expert = policy_optim_batch_size_from_expert
        self.discriminator = discriminator
        self.disc_optimizer = disc_optimizer_class(
            self.discriminator.parameters(), lr=disc_lr, betas=(disc_momentum, 0.999)
        )
        self.disc_optim_batch_size = disc_optim_batch_size
        self.bce = nn.BCEWithLogitsLoss()
        self.bec_without_logits = nn.BCELoss()
        self.bce_targets = torch.cat(
            [
                torch.ones(disc_optim_batch_size, 1),
                torch.zeros(disc_optim_batch_size, 1),
            ],
            dim=0,
        ).to(ptu.device)
        # loss = -[y*log(σ(x)) + (1-y)*log(1-σ(x))]
        self.use_grad_pen = use_grad_pen
        self.grad_pen_weight = grad_pen_weight
        self.num_update_loops_per_train_call = num_update_loops_per_train_call
        self.num_disc_updates_per_loop_iter = num_disc_updates_per_loop_iter
        self.num_policy_updates_per_loop_iter = num_policy_updates_per_loop_iter
        self.rew_clip_min = rew_clip_min
        self.rew_clip_max = rew_clip_max
        self.clip_min_rews = rew_clip_min is not None
        self.clip_max_rews = rew_clip_max is not None
        self.disc_eval_statistics = None
        self.select_timesteps = 1

    # -------------------------------
    # Training
    # -------------------------------

    def _do_training(self, epoch):
        """
        Perform the training loops for the current epoch.

        Args:
            epoch (int): The current epoch.
        """
        # Define the training methods for different modes

        disc_training_methods = {
            "GAIL": self._do_reward_training,
            "PADIL": self._do_reward_training,
        }

        policy_training_methods = {
            "GAIL": self._do_policy_training
            "PADIL": self._do_PADIL_policy_training,
        }

        # Perform the training loops for the current epoch
        for t in range(self.num_update_loops_per_train_call):
            # Perform the discriminator training
            for mode, method in disc_training_methods.items():
                if mode == self.mode:
                    for _ in range(self.num_disc_updates_per_loop_iter):
                        if self.mode == "GDAIL_V25":
                            self.select_timesteps = 1

                        method(epoch)
                    break

            # Perform the policy training
            for mode, method in policy_training_methods.items():
                if mode == self.mode:
                    for _ in range(self.num_policy_updates_per_loop_iter):
                        method(epoch)
                    break

    # -------------------------------
    # Training
    # -------------------------------

    # -------------------------------
    # Utils
    # -------------------------------

    def get_batch(self, batch_size, from_expert, keys=None):
        """
        Get a batch of data from the replay buffer.

        Args:
            batch_size (int): The size of the batch.
            from_expert (bool): Whether to get the batch from the expert's data.
            keys (list): The keys of the data to include in the batch.

        Returns:
            dict: The batch of data.
        """
        # Choose the appropriate replay buffer based on whether the batch should come from the expert
        if from_expert:
            buffer = self.expert_replay_buffer
        else:
            buffer = self.replay_buffer
        # Get a random batch from the chosen replay buffer
        batch = buffer.random_batch(batch_size, keys=keys)
        # Convert the batch from numpy arrays to PyTorch tensors
        batch = np_to_pytorch_batch(batch)

        return batch

    def evaluate(self, epoch):
        """
        Evaluate the algorithm at the given epoch.

        Args:
            epoch (int): The current epoch.
        """
        # Initialize the evaluation statistics
        self.eval_statistics = OrderedDict()
        # Add the discriminator evaluation statistics to the evaluation statistics
        self.eval_statistics.update(self.disc_eval_statistics)
        # Add the policy trainer's evaluation statistics to the evaluation statistics
        self.eval_statistics.update(self.policy_trainer.get_eval_statistics())
        # Call the parent class's evaluate method
        super().evaluate(epoch)

    def _end_epoch(self):
        """
        End the epoch for the policy trainer and reset the discriminator evaluation statistics.
        """
        # End the epoch for the policy trainer
        self.policy_trainer.end_epoch()
        self.disc_eval_statistics = None
        super()._end_epoch()

    def get_epoch_snapshot(self, epoch):
        """
        Get the snapshot of the current epoch.
        """
        snapshot = super().get_epoch_snapshot(epoch)
        snapshot.update(disc=self.discriminator)
        snapshot.update(self.policy_trainer.get_snapshot())
        return snapshot

    def to(self, device):
        """
        Move the BCE loss and the BCE targets to the specified device.
        """
        self.bce.to(ptu.device)
        self.bce_targets = self.bce_targets.to(ptu.device)
        super().to(device)

    @property
    def networks(self):
        """
        Return a list of all networks.
        """
        return [self.discriminator] + self.policy_trainer.networks

    # -------------------------------
    # Utils
    # -------------------------------

    # -------------------------------
    # Policy training
    # -------------------------------

    def _do_policy_training(self, epoch):

        if self.policy_optim_batch_size_from_expert > 0:
            policy_batch_from_policy_buffer = self.get_batch(
                self.policy_optim_batch_size - self.policy_optim_batch_size_from_expert,
                False,
            )
            policy_batch_from_expert_buffer = self.get_batch(
                self.policy_optim_batch_size_from_expert, True
            )
            policy_batch = {}
            for k in policy_batch_from_policy_buffer:
                policy_batch[k] = torch.cat(
                    [
                        policy_batch_from_policy_buffer[k],
                        policy_batch_from_expert_buffer[k],
                    ],
                    dim=0,
                )
        else:
            policy_batch = self.get_batch(self.policy_optim_batch_size, False)

        obs = policy_batch["observations"]
        acts = policy_batch["actions"]
        next_obs = policy_batch["next_observations"]

        if self.wrap_absorbing:
            # pass
            obs = torch.cat([obs, policy_batch["absorbing"][:, 0:1]], dim=-1)
            next_obs = torch.cat([next_obs, policy_batch["absorbing"][:, 1:]], dim=-1)


        self.discriminator.eval()
        if self.state_only:
            disc_input = torch.cat([obs, next_obs], dim=1)
        else:
            disc_input = torch.cat([obs, acts], dim=1)

        disc_logits = self.discriminator(disc_input).detach()
        self.discriminator.train()

        # compute the reward using the algorithm
        policy_batch["rewards"] = F.softplus(disc_logits, beta=-1)


        if self.clip_max_rews:
            policy_batch["rewards"] = torch.clamp(
                policy_batch["rewards"], max=self.rew_clip_max
            )
        if self.clip_min_rews:
            policy_batch["rewards"] = torch.clamp(
                policy_batch["rewards"], min=self.rew_clip_min
            )

        # policy optimization step
        self.policy_trainer.train_step(policy_batch)

        self.disc_eval_statistics["Disc Rew Mean"] = np.mean(
            ptu.get_numpy(policy_batch["rewards"])
        )
        self.disc_eval_statistics["Disc Rew Std"] = np.std(
            ptu.get_numpy(policy_batch["rewards"])
        )
        self.disc_eval_statistics["Disc Rew Max"] = np.max(
            ptu.get_numpy(policy_batch["rewards"])
        )
        self.disc_eval_statistics["Disc Rew Min"] = np.min(
            ptu.get_numpy(policy_batch["rewards"])
        )

    def _do_PADIL_policy_training(self, epoch):
        """
        Train the diffusion policy.
        """

        # If policy_optim_batch_size_from_expert is greater than 0, get batches from both the policy and expert buffer
        if self.policy_optim_batch_size_from_expert > 0:
            policy_batch_from_policy_buffer = self.get_batch(
                self.policy_optim_batch_size - self.policy_optim_batch_size_from_expert,
                False,
            )
            policy_batch_from_expert_buffer = self.get_batch(
                self.policy_optim_batch_size_from_expert, True
            )
            # Concatenate the batches from the policy and expert buffer
            policy_batch = {}
            for k in policy_batch_from_policy_buffer:
                policy_batch[k] = torch.cat(
                    [
                        policy_batch_from_policy_buffer[k],
                        policy_batch_from_expert_buffer[k],
                    ],
                    dim=0,
                )
        # If policy_optim_batch_size_from_expert is not greater than 0, get batch from the policy buffer only
        else:
            policy_batch = self.get_batch(self.policy_optim_batch_size, False)
            best_actions = (
                torch.as_tensor(self.replay_buffer.get_best_actions())
                .to(ptu.device)
                .float()
            )

        # Get observations, actions and next observations from the batch
        obs = policy_batch["observations"]
        acts = policy_batch["actions"]
        next_obs = policy_batch["next_observations"]
        policy_batch["best_actions"] = best_actions

        # If wrap_absorbing is True, concatenate the absorbing state to the observations
        if self.wrap_absorbing:
            obs = torch.cat([obs, policy_batch["absorbing"][:, 0:1]], dim=-1)
            next_obs = torch.cat([next_obs, policy_batch["absorbing"][:, 1:]], dim=-1)

        self.discriminator.eval()

        if self.state_only:
            disc_input = torch.cat([obs, next_obs], dim=1)
        else:
            disc_input = torch.cat([obs, acts], dim=1)

        disc_logits = self.discriminator(disc_input).detach()
        self.discriminator.train()

        # Compute the reward using the algorithm
        policy_batch["rewards"] = F.softplus(disc_logits, beta=-1)

        # If clip_max_rews is True, clip the rewards to the maximum reward
        if self.clip_max_rews:
            policy_batch["rewards"] = torch.clamp(
                policy_batch["rewards"], max=self.rew_clip_max
            )

        # If clip_min_rews is True, clip the rewards to the minimum reward
        if self.clip_min_rews:
            policy_batch["rewards"] = torch.clamp(
                policy_batch["rewards"], min=self.rew_clip_min
            )

        # Perform a policy optimization step
        self.policy_trainer.train_step(policy_batch)

        # Save some statistics for evaluation
        self.disc_eval_statistics["Disc Rew Mean"] = np.mean(
            ptu.get_numpy(policy_batch["rewards"])
        )
        self.disc_eval_statistics["Disc Rew Std"] = np.std(
            ptu.get_numpy(policy_batch["rewards"])
        )
        self.disc_eval_statistics["Disc Rew Max"] = np.max(
            ptu.get_numpy(policy_batch["rewards"])
        )
        self.disc_eval_statistics["Disc Rew Min"] = np.min(
            ptu.get_numpy(policy_batch["rewards"])
        )

   
    # -------------------------------
    # Reward training
    # -------------------------------

    def _do_reward_training(self, epoch):
        """
        Train the discriminator
        """
        # Reset the gradients of the discriminator optimizer
        self.disc_optimizer.zero_grad()

        # Define the keys for the batch
        keys = ["observations"]
        if self.state_only:
            keys.append("next_observations")
        else:
            keys.append("actions")
        if self.wrap_absorbing:
            keys.append("absorbing")

        # Get batch from expert and policy buffer
        expert_batch = self.get_batch(self.disc_optim_batch_size, True, keys)
        policy_batch = self.get_batch(self.disc_optim_batch_size, False, keys)

        # Get observations from the batch
        expert_obs = expert_batch["observations"]
        policy_obs = policy_batch["observations"]

        # If wrap_absorbing is True, concatenate the absorbing state to the observations
        if self.wrap_absorbing:
            expert_obs = torch.cat(
                [expert_obs, expert_batch["absorbing"][:, 0:1]], dim=-1
            )
            policy_obs = torch.cat(
                [policy_obs, policy_batch["absorbing"][:, 0:1]], dim=-1
            )

        # If state_only is True, concatenate the next observations to the observations
        # Otherwise, concatenate the actions to the observations
        if self.state_only:
            expert_next_obs = expert_batch["next_observations"]
            policy_next_obs = policy_batch["next_observations"]
            if self.wrap_absorbing:
                expert_next_obs = torch.cat(
                    [expert_next_obs, expert_batch["absorbing"][:, 1:]], dim=-1
                )
                policy_next_obs = torch.cat(
                    [policy_next_obs, policy_batch["absorbing"][:, 1:]], dim=-1
                )
            expert_disc_input = torch.cat([expert_obs, expert_next_obs], dim=1)
            policy_disc_input = torch.cat([policy_obs, policy_next_obs], dim=1)
        else:
            expert_acts = expert_batch["actions"]
            policy_acts = policy_batch["actions"]
            expert_disc_input = torch.cat([expert_obs, expert_acts], dim=1)
            policy_disc_input = torch.cat([policy_obs, policy_acts], dim=1)

        # Concatenate the expert and policy inputs and compute the discriminator output
        disc_input = torch.cat([expert_disc_input, policy_disc_input], dim=0)
        disc_logits = self.discriminator(disc_input)

        # Compute the discriminator predictions and the cross-entropy loss
        disc_preds = (disc_logits > 0).type(disc_logits.data.type())
        disc_ce_loss = self.bce(disc_logits, self.bce_targets)

        # Compute the accuracy of the discriminator
        accuracy = (disc_preds == self.bce_targets).type(torch.FloatTensor).mean()

        # If use_grad_pen is True, compute the gradient penalty loss
        if self.use_grad_pen:
            eps = ptu.rand(expert_obs.size(0), 1)
            eps.to(ptu.device)

            interp_obs = eps * expert_disc_input + (1 - eps) * policy_disc_input
            interp_obs = interp_obs.detach()
            interp_obs.requires_grad_(True)
            a = self.discriminator(interp_obs).sum()
            gradients = autograd.grad(
                outputs=self.discriminator(interp_obs).sum(),
                inputs=[interp_obs],
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )
            total_grad = gradients[0]

            # GP from Gulrajani et al.
            gradient_penalty = ((total_grad.norm(2, dim=1) - 1) ** 2).mean()
            disc_grad_pen_loss = gradient_penalty * self.grad_pen_weight

        else:
            disc_grad_pen_loss = 0.0

        # Compute the total loss and perform a discriminator optimization step
        disc_total_loss = disc_ce_loss + disc_grad_pen_loss
        disc_total_loss.backward()
        self.disc_optimizer.step()
        """
            Save some statistics for evaluation
            """
        if self.disc_eval_statistics is None:
            """
            Evaluation should set this to None.
            This way, these statistics are only computed for one batch.
            """
            self.disc_eval_statistics = OrderedDict()

            # Save the mean cross-entropy loss and the accuracy
            self.disc_eval_statistics["Disc CE Loss"] = np.mean(
                ptu.get_numpy(disc_ce_loss)
            )
            self.disc_eval_statistics["Disc Acc"] = np.mean(ptu.get_numpy(accuracy))

            # If gradient penalty is used, save the mean gradient penalty and the gradient penalty weight
            if self.use_grad_pen:
                self.disc_eval_statistics["Grad Pen"] = np.mean(
                    ptu.get_numpy(gradient_penalty)
                )
                self.disc_eval_statistics["Grad Pen W"] = np.mean(self.grad_pen_weight)

 
    # -------------------------------
    # Reward training
    # -------------------------------

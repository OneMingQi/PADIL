# Python standard libraries
import copy
import math
from typing import Optional, Tuple
from collections import OrderedDict

# Third-party libraries
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Local application imports
import algorithms.rlkit.torch.utils.pytorch_util as ptu
from algorithms.rlkit.core.trainer import Trainer
from algorithms.rlkit.core.eval_util import create_stats_ordered_dict
import itertools


class diffusion_policy_trainer(Trainer):
    def __init__(
        self,
        policy,
        q1,
        q2,
        env,
        reward_scale=1.0,
        discount=0.99,
        args=None,
    ):
        """
        Initialize the trainer.

        Args:
            policy: The policy to be trained.
            q1: The first Q-function.
            q2: The second Q-function.
            env: The environment in which the policy will be trained.
            args: The arguments for training.
        """
        self.reward_scale = reward_scale
        self.discount = discount

        self.diffusion_steps_needed = policy.diffusion_steps_needed

        self.eval_statistics = None

        action_space = env.action_space
        action_dim = np.prod(action_space.shape)

        self.action_gradient_steps = args["action_gradient_steps"]
        self.action_grad_norm = action_dim * args["ratio"]
        self.ac_grad_norm = args["ac_grad_norm"]

        self.action_dim = action_dim
        self.action_lr = args["action_lr"]

        if action_space is None:
            self.action_scale = 1.0
            self.action_bias = 0.0
        else:
            self.action_scale = (action_space.high - action_space.low) / 2.0
            self.action_bias = (action_space.high + action_space.low) / 2.0

        self.policy = policy
        self.actor_optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=args["diffusion_lr"], eps=1e-5
        )

        self.q1 = q1
        self.q2 = q2
        self.env = env

        self.step = 0
        self.tau = args["tau"]
        self.actor_target = copy.deepcopy(self.policy)
        self.update_actor_target_every = args["update_actor_target_every"]

        self.q1_target = copy.deepcopy(self.q1)
        self.q2_target = copy.deepcopy(self.q2)

        self.q1_optimizer = torch.optim.Adam(
            self.q1.parameters(), lr=args["critic_lr"], eps=1e-5
        )
        self.q2_optimizer = torch.optim.Adam(
            self.q2.parameters(), lr=args["critic_lr"], eps=1e-5
        )

    @property
    def networks(self):
        return [
            self.policy,
            self.q1,
            self.q2,
            self.q1_target,
            self.q2_target,
        ]

    def action_gradient(
        self,
        states,
        best_actions,
    ):
        """
        Computes the action gradient for the given batch size.

        Args:
            states: The states for which the action gradient is computed.
            best_actions: The best actions for the given states.

        Returns:
            None
        """
        # Extract states, best actions and indices from the diffusion memory

        # Create an Adam optimizer for the best actions'

        actions_optim = torch.optim.Adam([best_actions], lr=self.action_lr, eps=1e-5)

        # Enable gradient computation for the best actions
        best_actions.requires_grad_(True)

        for i in range(self.action_gradient_steps):
            # Compute Q values for the states and best actions
            q1 = self.q1(states, best_actions)
            q2 = self.q2(states, best_actions)

            # Compute the loss as the negative minimum of the Q values
            loss = -torch.min(q1, q2)

            # Zero the gradients of the optimizer
            actions_optim.zero_grad()

            # Compute the gradients of the loss
            loss.backward(torch.ones_like(loss))

            # Clip the gradients of the best actions if necessary
            if self.action_grad_norm > 0:
                nn.utils.clip_grad_norm_(
                    [best_actions], max_norm=self.action_grad_norm, norm_type=2
                )

            # Perform a step of the optimizer
            actions_optim.step()

            # Clamp the best actions to the range [-1, 1]
            best_actions.data.clamp_(-1.0, 1.0)

        # Disable gradient computation for the best actions
        best_actions.requires_grad_(False)

    def _update_target_network(self):
        """
        Perform soft update for the target networks.

        The parameters of the target networks are updated by taking a step
        towards the corresponding parameters of the original networks.

        Args:
            None

        Returns:
            None
        """
        # Perform soft update for the first Q-function's target network
        ptu.soft_update_from_to(self.q1, self.q1_target, self.tau)

        # Perform soft update for the second Q-function's target network
        ptu.soft_update_from_to(self.q2, self.q2_target, self.tau)

    def train_step(self, epoch):
        """
        Perform a training step.

        Args:
            epoch: The epoch data containing rewards, terminals, states, actions, next states, and best actions.

        Returns:
            None
        """
        # Extract data from the memory
        rewards = self.reward_scale * epoch["rewards"]
        terminals = epoch["terminals"]
        states = epoch["observations"]
        actions = epoch["actions"]
        next_states = epoch["next_observations"]
        best_actions = epoch["best_actions"]

        """Q Training"""
        # Zero out the gradients of the Q-function optimizers
        self.q1_optimizer.zero_grad()
        self.q2_optimizer.zero_grad()

        # Compute the current Q-values
        current_q1 = self.q1(states, actions)
        current_q2 = self.q2(states, actions)

        if self.diffusion_steps_needed:
            # Compute the next actions using the target policy
            next_actions, _ = self.actor_target(next_states, eval=False)
        else:
            next_actions = self.actor_target(next_states, eval=False)

        # Compute the target Q-values
        with torch.no_grad():  # Don't compute gradients for the following operations
            target_q1 = self.q1_target(next_states, next_actions)
            target_q2 = self.q2_target(next_states, next_actions)

            # Use the minimum of the two target Q-values
            min_target = torch.min(target_q1, target_q2)

            # Compute the target for the Q-function update
            q_target = rewards + (1 - terminals) * self.discount * min_target

        # Compute the Q-function loss
        q1_loss = 0.5 * F.mse_loss(current_q1, q_target)
        q2_loss = 0.5 * F.mse_loss(current_q2, q_target)

        # Backpropagate the loss gradients to the parameters
        q1_loss.backward()
        q2_loss.backward()

        # Clip the gradients if necessary
        if self.ac_grad_norm > 0:
            nn.utils.clip_grad_norm_(self.q1.parameters(), self.ac_grad_norm)
            nn.utils.clip_grad_norm_(self.q2.parameters(), self.ac_grad_norm)

        # Update the Q-function parameters
        self.q1_optimizer.step()
        self.q2_optimizer.step()

        """Policy Training"""
        # Compute the action gradient
        self.action_gradient(states, best_actions)
        epoch["best_actions"] = best_actions
        # Compute the policy loss
        actor_loss = self.policy.loss(best_actions, states)

        # Zero out the gradients of the policy optimizer
        self.actor_optimizer.zero_grad()

        # Backpropagate the loss gradients to the parameters
        actor_loss.backward()

        if self.ac_grad_norm > 0:
            nn.utils.clip_grad_norm_(
                self.policy.parameters(), max_norm=self.ac_grad_norm, norm_type=2
            )

        # Update the policy parameters
        self.actor_optimizer.step()

        """ Step Target network """
        # Update the target networks
        self._update_target_network()

        # Update the target policy network every self.update_actor_target_every steps
        if self.step % self.update_actor_target_every == 0:
            for param, target_param in zip(
                self.policy.parameters(), self.actor_target.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )

        # Increment the step counter
        self.step = self.step + 1

        """
        Save some statistics for eval
        """
        if self.eval_statistics is None:
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            self.eval_statistics = OrderedDict()
            self.eval_statistics["Reward Scale"] = self.reward_scale
            self.eval_statistics["QF1 Loss"] = np.mean(ptu.get_numpy(q1_loss))
            self.eval_statistics["QF2 Loss"] = np.mean(ptu.get_numpy(q2_loss))
            self.eval_statistics["Policy Loss"] = np.mean(ptu.get_numpy(actor_loss))

            # Add Q1 and Q2 predictions to the statistics
            self.eval_statistics.update(
                create_stats_ordered_dict(
                    "Q1 Predictions",
                    ptu.get_numpy(current_q1),
                )
            )
            self.eval_statistics.update(
                create_stats_ordered_dict(
                    "Q2 Predictions",
                    ptu.get_numpy(current_q2),
                )
            )

            # Add the gradient norms of the parameters of Q1, Q2, Q1 target, Q2 target, and policy to the statistics
            for name, param in itertools.chain(
                self.q1.named_parameters(),
                self.q2.named_parameters(),
                self.q1_target.named_parameters(),
                self.q2_target.named_parameters(),
                self.policy.named_parameters(),
            ):
                if param.grad is not None:
                    grad_norm = param.grad.data.norm()
                    self.eval_statistics[name] = ptu.get_numpy(grad_norm)

    def get_eval_statistics(self):
        """
        Get the evaluation statistics.

        Args:
            None

        Returns:
            The evaluation statistics.
        """
        return self.eval_statistics

    def end_epoch(self):
        """
        End the current epoch.

        Args:
            None

        Returns:
            None
        """
        self.eval_statistics = None

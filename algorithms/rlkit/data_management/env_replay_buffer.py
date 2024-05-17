from algorithms.rlkit.data_management.simple_replay_buffer import (
    SimpleReplayBuffer,
)
from gym.spaces import Box, Discrete, Tuple, Dict


class EnvReplayBuffer(SimpleReplayBuffer):
    def __init__(
        self,
        max_replay_buffer_size,
        env,
        random_seed=1995,
        disc_ddpm=False,
        diffusion_steps_needed=False,
    ):
        """
        :param max_replay_buffer_size:
        :param env:
        """
        self._ob_space = env.observation_space
        self._action_space = env.action_space
        self.ddpm = disc_ddpm

        self.diffusion_steps_needed = diffusion_steps_needed

        super().__init__(
            max_replay_buffer_size=max_replay_buffer_size,
            observation_dim=get_dim(self._ob_space),
            action_dim=get_dim(self._action_space),
            random_seed=random_seed,
            diffusion_steps_needed=self.diffusion_steps_needed,
        )

    def get_best_actions(self):
        return self.best_actions[self.indices]

    def renew_best_actions(self, better_action):
        self.best_actions[self.indices] = better_action

    def get_diffusion_actions(self):
        return self.diffusion_actions[self.indices]

    def add_sample(
        self,
        observation,
        action,
        reward,
        terminal,
        next_observation,
        diffusion_actions=None,
        **kwargs
    ):
        if self.diffusion_steps_needed == False:
            super(EnvReplayBuffer, self).add_sample(
                observation, action, reward, terminal, next_observation, **kwargs
            )
        else:
            super(EnvReplayBuffer, self).add_sample(
                observation,
                action,
                reward,
                terminal,
                next_observation,
                diffusion_actions=diffusion_actions,
                **kwargs
            )


def get_dim(space):
    """Get the dimension of a gym space."""

    # If the space is a Box, return its shape or size.
    if isinstance(space, Box):
        if len(space.low.shape) > 1:
            return space.low.shape
        return space.low.size

    # If the space is Discrete, return 1.
    elif isinstance(space, Discrete):
        return 1  # space.n

    # If the space is a Tuple, return the sum of the dimensions of its subspaces.
    elif isinstance(space, Tuple):
        return sum(get_dim(subspace) for subspace in space.spaces)

    # If the space is a Dict, return a dictionary where the keys are the names of the subspaces
    # and the values are the dimensions of the subspaces.
    elif isinstance(space, Dict):
        return {k: get_dim(v) for k, v in space.spaces.items()}

    # If the space has a flat_dim attribute, return its value.
    elif hasattr(space, "flat_dim"):
        return space.flat_dim

    # If the space is not any of the above types, raise a TypeError.
    else:
        raise TypeError("Unknown space: {}".format(space))

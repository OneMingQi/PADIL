from itertools import starmap
import numpy as np
import pickle
import abc


class ReplayBuffer(object, metaclass=abc.ABCMeta):
    """
    A class used to save and replay data.
    """

    @abc.abstractmethod
    def add_sample(
        self, observation, action, reward, next_observation, terminal, **kwargs
    ):
        """
        Add a transition tuple with best actions.
        """
        pass

    @abc.abstractmethod
    def terminate_episode(self):
        """
        Let the replay buffer know that the episode has terminated in case some
        special book-keeping has to happen.
        :return:
        """
        pass

    @abc.abstractmethod
    def num_steps_can_sample(self, **kwargs):
        """
        :return: # of unique items that can be sampled.
        """
        pass

    def add_path(self, path):
        """
        Add a path to the replay buffer.

        This default implementation naively goes through every step, but you
        may want to optimize this.

        NOTE: You should NOT call "terminate_episode" after calling add_path.
        It's assumed that this function handles the episode termination.

        :param path: Dict like one outputted by rlkit.samplers.util.rollout
        """
        for i, (
            obs,
            action,
            reward,
            next_obs,
            terminal,
            agent_info,
            env_info,
        ) in enumerate(
            zip(
                path["observations"],
                path["actions"],
                path["rewards"],
                path["next_observations"],
                path["terminals"],
                path["agent_infos"],
                path["env_infos"],
            )
        ):
            self.add_sample(
                obs,
                action,
                reward,
                next_obs,
                terminal,
                agent_info=agent_info,
                env_info=env_info,
            )
        self.terminate_episode()

    @abc.abstractmethod
    def random_batch(self, batch_size):
        """
        Return a batch of size `batch_size`.
        :param batch_size:
        :return:
        """
        pass


class SimpleReplayBuffer(ReplayBuffer):
    """
    THE MAX LENGTH OF AN EPISODE SHOULD BE STRICTLY SMALLER THAN THE max_replay_buffer_size
    OTHERWISE THERE IS A BUG IN TERMINATE_EPISODE

    # It's a bit memory inefficient to save the observations twice,
    # but it makes the code *much* easier since you no longer have to
    # worry about termination conditions.
    """

    def __init__(
        self,
        max_replay_buffer_size,
        observation_dim,
        action_dim,
        random_seed=1995,
        diffusion_steps_needed=False,
        diffusion_n_timesteps=10,
    ):
        self.indices = None
        self._np_rand_state = np.random.RandomState(random_seed)

        self.diffusion_steps_needed = diffusion_steps_needed

        self._observation_dim = observation_dim
        self._action_dim = action_dim
        self._max_replay_buffer_size = max_replay_buffer_size

        obs_dtype = np.uint8 if type(observation_dim) == tuple else np.float64

        self.diffusion_n_timesteps = diffusion_n_timesteps

        if isinstance(observation_dim, tuple):
            dims = [d for d in observation_dim]
            dims = [max_replay_buffer_size] + dims
            dims = tuple(dims)
            self._observations = np.zeros(dims, dtype=obs_dtype)
            self._next_obs = np.zeros(dims, dtype=obs_dtype)
        elif isinstance(observation_dim, dict):
            # assuming that this is a one-level dictionary
            self._observations = {}
            self._next_obs = {}

            for key, dims in observation_dim.items():
                if isinstance(dims, tuple):
                    dims = tuple([max_replay_buffer_size] + list(dims))
                else:
                    dims = (max_replay_buffer_size, dims)
                self._observations[key] = np.zeros(dims, dtype=obs_dtype)
                self._next_obs[key] = np.zeros(dims, dtype=obs_dtype)
        else:
            # else observation_dim is an integer
            self._observations = np.zeros((max_replay_buffer_size, observation_dim))
            self._next_obs = np.zeros((max_replay_buffer_size, observation_dim))

        self._actions = np.zeros((max_replay_buffer_size, action_dim))
        self.best_actions = np.zeros((max_replay_buffer_size, action_dim))

        if self.diffusion_steps_needed == True:
            self.diffusion_actions = np.zeros(
                (max_replay_buffer_size, self.diffusion_n_timesteps, action_dim)
            )

        # Make everything a 2D np array to make it easier for other code to
        # reason about the shape of the data
        self._rewards = np.zeros((max_replay_buffer_size, 1))
        # self._terminals[i] = a terminal was received at time i
        self._terminals = np.zeros((max_replay_buffer_size, 1), dtype="uint8")
        self._timeouts = np.zeros((max_replay_buffer_size, 1), dtype="uint8")
        # absorbing[0] is if obs was absorbing, absorbing[1] is if next_obs was absorbing
        self._absorbing = np.zeros((max_replay_buffer_size, 2))
        self._top = 0
        self._size = 0
        self._trajs = 0

        # keeping track of trajectory boundaries
        # assumption is trajectory lengths are AT MOST the length of the entire replay buffer
        self._cur_start = 0
        self._traj_endpoints = {}  # start->end means [start, end)

    def _np_randint(self, *args, **kwargs):
        rets = self._np_rand_state.randint(*args, **kwargs)
        return rets

    def _np_choice(self, *args, **kwargs):
        rets = self._np_rand_state.choice(*args, **kwargs)
        return rets

    def add_sample(
        self,
        observation,
        action,
        reward,
        terminal,
        next_observation,
        timeout=False,
        diffusion_actions=None,
        **kwargs
    ):
        self._actions[self._top] = action
        self.best_actions[self._top] = action

        if self.diffusion_steps_needed:
            if diffusion_actions is not None:
                self.diffusion_actions[self._top] = diffusion_actions
            else:
                self.diffusion_actions[self._top] = np.zeros(
                    (self.diffusion_n_timesteps, 1)
                )

        self._rewards[self._top] = reward
        self._terminals[self._top] = terminal
        self._timeouts[self._top] = timeout
        if "absorbing" in kwargs:
            self._absorbing[self._top] = kwargs["absorbing"]

        if terminal:
            next_start = (self._top + 1) % self._max_replay_buffer_size
            self._traj_endpoints[self._cur_start] = next_start
            self._cur_start = next_start

        if isinstance(self._observations, dict):
            for key, obs in observation.items():
                self._observations[key][self._top] = obs
            for key, obs in next_observation.items():
                self._next_obs[key][self._top] = obs
        else:
            self._observations[self._top] = observation
            self._next_obs[self._top] = next_observation

        self._advance()

    def save_data(self, save_name):
        if self.diffusion_steps_needed == True:
            save_dict = {
                "observations": self._observations[: self._top],
                "actions": self._actions[: self._top],
                "next_observations": self._next_obs[: self._top],
                "terminals": self._terminals[: self._top],
                "timeouts": self._timeouts[: self._top],
                "rewards": self._rewards[: self._top],
                "diffusion_actions": self.diffusion_actions[: self._top],
                "agent_infos": [None] * len(self._observations[: self._top]),
                "env_infos": [None] * len(self._observations[: self._top]),
            }
        else:
            save_dict = {
                "observations": self._observations[: self._top],
                "actions": self._actions[: self._top],
                "next_observations": self._next_obs[: self._top],
                "terminals": self._terminals[: self._top],
                "timeouts": self._timeouts[: self._top],
                "rewards": self._rewards[: self._top],
                "agent_infos": [None] * len(self._observations[: self._top]),
                "env_infos": [None] * len(self._observations[: self._top]),
            }

        with open(save_name, "wb") as f:
            pickle.dump(save_dict, f)

    def terminate_episode(self):
        if self._cur_start != self._top:
            # if they are equal it means that the previous state was terminal
            # and was handled so there is no need to handle it again
            # THERE WILL BE A BUG HERE IS max_replay_buffer_size
            # IS NOT STRICTLY LARGER THAN MAX EPISODE LENGTH
            self._traj_endpoints[self._cur_start] = self._top
            self._cur_start = self._top

    def add_path(self, path, absorbing=False, env=None):
        # add path using in expert data
        if not absorbing:
            for (
                ob,
                action,
                reward,
                next_ob,
                terminal,
                # agent_info,
                # env_info
            ) in zip(
                path["observations"],
                path["actions"],
                path["rewards"],
                path["next_observations"],
                path["terminals"],
                # path["agent_infos"],
                # path["env_infos"],
            ):
                self.add_sample(
                    observation=ob,
                    action=action,
                    reward=reward,
                    terminal=terminal,
                    next_observation=next_ob,
                    # agent_info=agent_info,
                    # env_info=env_info,
                )
        else:
            for (
                ob,
                action,
                reward,
                next_ob,
                terminal,
                # agent_info,
                # env_info
            ) in zip(
                path["observations"],
                path["actions"],
                path["rewards"],
                path["next_observations"],
                path["terminals"],
                # path["agent_infos"],
                # path["env_infos"],
            ):
                self.add_sample(
                    observation=ob,
                    action=action,
                    reward=reward,
                    terminal=np.array([False]),
                    next_observation=next_ob,
                    absorbing=np.array([0.0, 0.0]),
                    # agent_info=agent_info,
                    # env_info=env_info,
                )
                if terminal[0]:
                    print("add terminal")
                    self.add_sample(
                        observation=next_ob,
                        # action=action,
                        action=env.action_space.sample(),
                        reward=reward,
                        terminal=np.array([False]),
                        next_observation=np.zeros_like(next_ob),  # next_ob,
                        absorbing=np.array([0.0, 1.0]),
                        # agent_info=agent_info,
                        # env_info=env_info,
                    )
                    self.add_sample(
                        observation=np.zeros_like(next_ob),  # next_ob,
                        # action=action,
                        action=env.action_space.sample(),
                        reward=reward,
                        terminal=np.array([False]),
                        next_observation=np.zeros_like(next_ob),  # next_ob,
                        absorbing=np.array([1.0, 1.0]),
                        # agent_info=agent_info,
                        # env_info=env_info,
                    )

        self.terminate_episode()
        self._trajs += 1

    def get_traj_num(self):
        return self._trajs

    def get_all(self, keys=None, multi_step=False, step_num=1, **kwargs):
        indices = range(self._size)

        return self._get_batch_using_indices(
            indices, keys=keys, multi_step=multi_step, step_num=step_num, **kwargs
        )

    def _advance(self):
        if self._top in self._traj_endpoints:
            # this means that the step in the replay buffer
            # that we just overwrote was the start of a some
            # trajectory, so now the full trajectory is no longer
            # there and we should remove it
            del self._traj_endpoints[self._top]
        self._top = (self._top + 1) % self._max_replay_buffer_size
        if self._size < self._max_replay_buffer_size:
            self._size += 1

    def random_batch(
        self, batch_size, keys=None, multi_step=False, step_num=1, **kwargs
    ):
        indices = self._np_randint(0, self._size, batch_size)
        if multi_step:
            indices = self._np_randint(0, self._size - step_num, batch_size)
            # candidates = list(range(0, self._size))
            # for value in list(self._traj_endpoints.values()):
            #     for step in np.arange(0, step_num):
            #         candidates.remove((value-step-1)%self._size) # endpoints are like [:endpoints] that may do not live in candidates
            # indices = np.random.choice(candidates, batch_size)

        return self._get_batch_using_indices(
            indices, keys=keys, multi_step=multi_step, step_num=step_num, **kwargs
        )

    def _get_batch_using_indices(
        self, indices, keys=None, multi_step=False, step_num=1
    ):
        if self.diffusion_steps_needed == False:
            if keys is None:
                keys = set(
                    [
                        "observations",
                        "actions",
                        "rewards",
                        "terminals",
                        "next_observations",
                        "absorbing",
                    ]
                )
            if isinstance(self._observations, dict):
                obs_to_return = {}
                next_obs_to_return = {}
                for k in self._observations:
                    if "observations" in keys:
                        obs_to_return[k] = self._observations[k][indices]
                    if "next_observations" in keys:
                        next_obs_to_return[k] = self._next_obs[k][indices]
            else:
                obs_to_return = self._observations[indices]
                next_obs_to_return = self._next_obs[indices]

            self.indices = None

            self.indices = indices

            ret_dict = {}

            if "observations" in keys:
                ret_dict["observations"] = obs_to_return
            if "actions" in keys:
                ret_dict["actions"] = self._actions[indices]
            if "rewards" in keys:
                ret_dict["rewards"] = self._rewards[indices]
            if "terminals" in keys:
                ret_dict["terminals"] = self._terminals[indices]
            if "next_observations" in keys:
                ret_dict["next_observations"] = next_obs_to_return
            if "absorbing" in keys:
                ret_dict["absorbing"] = self._absorbing[indices]

            if multi_step:
                next_next_obs_return = [None] * step_num
                for i in np.arange(1, step_num + 1):
                    if isinstance(self._observations, dict):
                        next_next_obs_return[i - 1] = {}
                        for k in self._observations:
                            next_next_obs_return[i - 1][k] = self._next_obs[k][
                                (indices + i) % self._max_replay_buffer_size
                            ]
                    else:
                        next_next_obs_return[i - 1] = self._next_obs[
                            (indices + i) % self._max_replay_buffer_size
                        ]

                    for j, indice in enumerate(indices):
                        source_list = list(range(indice + 1, indice + i + 1))
                        target_list = list(self._traj_endpoints.values())
                        res = set(source_list) & set(target_list)
                        if (
                            len(res) > 0
                        ):  # there is a number in range(indice, indice+i+1) are a traj endpoint, this should be the last state of the traj
                            next_next_obs_return[i - 1][j] = self._next_obs[
                                list(res)[0] - 1
                            ]

                    ret_dict["next{}_observations".format(i)] = next_next_obs_return[
                        i - 1
                    ]

        else:
            if keys is None:
                keys = set(
                    [
                        "observations",
                        "actions",
                        "rewards",
                        "terminals",
                        "next_observations",
                        "absorbing",
                        "diffusion_actions",
                    ]
                )
            if isinstance(self._observations, dict):
                obs_to_return = {}
                next_obs_to_return = {}
                for k in self._observations:
                    if "observations" in keys:
                        obs_to_return[k] = self._observations[k][indices]
                    if "next_observations" in keys:
                        next_obs_to_return[k] = self._next_obs[k][indices]
            else:
                obs_to_return = self._observations[indices]
                next_obs_to_return = self._next_obs[indices]

            self.indices = None

            self.indices = indices

            ret_dict = {}

            if "observations" in keys:
                ret_dict["observations"] = obs_to_return
            if "actions" in keys:
                ret_dict["actions"] = self._actions[indices]
            if "diffusion_actions" in keys:
                ret_dict["diffusion_actions"] = self.diffusion_actions[indices]
            if "rewards" in keys:
                ret_dict["rewards"] = self._rewards[indices]
            if "terminals" in keys:
                ret_dict["terminals"] = self._terminals[indices]
            if "next_observations" in keys:
                ret_dict["next_observations"] = next_obs_to_return
            if "absorbing" in keys:
                ret_dict["absorbing"] = self._absorbing[indices]

            if multi_step:
                next_next_obs_return = [None] * step_num
                for i in np.arange(1, step_num + 1):
                    if isinstance(self._observations, dict):
                        next_next_obs_return[i - 1] = {}
                        for k in self._observations:
                            next_next_obs_return[i - 1][k] = self._next_obs[k][
                                (indices + i) % self._max_replay_buffer_size
                            ]
                    else:
                        next_next_obs_return[i - 1] = self._next_obs[
                            (indices + i) % self._max_replay_buffer_size
                        ]

                    for j, indice in enumerate(indices):
                        source_list = list(range(indice + 1, indice + i + 1))
                        target_list = list(self._traj_endpoints.values())
                        res = set(source_list) & set(target_list)
                        if (
                            len(res) > 0
                        ):  # there is a number in range(indice, indice+i+1) are a traj endpoint, this should be the last state of the traj
                            next_next_obs_return[i - 1][j] = self._next_obs[
                                list(res)[0] - 1
                            ]

                    ret_dict["next{}_observations".format(i)] = next_next_obs_return[
                        i - 1
                    ]

        # print(step_num, ret_dict.keys())
        return ret_dict

    def _get_segment(self, start, end, keys=None):
        if start < end or end == 0:
            if end == 0:
                end = self._max_replay_buffer_size
            return self._get_batch_using_indices(range(start, end), keys=keys)

        inds = list(range(start, self._max_replay_buffer_size)) + list(range(0, end))
        return self._get_batch_using_indices(inds, keys=keys)

    def _get_samples_from_traj(self, start, end, samples_per_traj, keys=None):
        # subsample a trajectory
        if start < end or end == 0:
            if end == 0:
                end = self._max_replay_buffer_size
            inds = range(start, end)
        else:
            inds = list(range(start, self._max_replay_buffer_size)) + list(
                range(0, end)
            )
        inds = self._np_choice(
            inds, size=samples_per_traj, replace=len(inds) < samples_per_traj
        )
        return self._get_batch_using_indices(inds, keys=keys)

    def sample_trajs(self, num_trajs, keys=None, samples_per_traj=None):
        # samples_per_traj of None mean use all of the samples
        keys_list = list(self._traj_endpoints.keys())
        starts = self._np_choice(
            keys_list, size=num_trajs, replace=len(keys_list) < num_trajs
        )
        ends = map(lambda k: self._traj_endpoints[k], starts)

        if samples_per_traj is None:
            return list(
                starmap(lambda s, e: self._get_segment(s, e, keys), zip(starts, ends))
            )
        else:
            return list(
                starmap(
                    lambda s, e: self._get_samples_from_traj(
                        s, e, samples_per_traj, keys
                    ),
                    zip(starts, ends),
                )
            )

    def num_steps_can_sample(self):
        return self._size

    def sample_all_trajs(
        self,
        keys=None,
        samples_per_traj=None,
    ):
        # samples_per_traj of None mean use all of the samples
        starts = list(self._traj_endpoints.keys())
        ends = map(lambda k: self._traj_endpoints[k], starts)

        if samples_per_traj is None:
            return list(
                starmap(lambda s, e: self._get_segment(s, e, keys), zip(starts, ends))
            )
        else:
            return list(
                starmap(
                    lambda s, e: self._get_samples_from_traj(
                        s, e, samples_per_traj, keys
                    ),
                    zip(starts, ends),
                )
            )

    def clear(self):
        if isinstance(self._observation_dim, tuple):
            dims = [d for d in self._observation_dim]
            dims = [self._max_replay_buffer_size] + dims
            dims = tuple(dims)
            self._observations = np.zeros(dims)
            self._next_obs = np.zeros(dims)
        elif isinstance(self._observation_dim, dict):
            # assuming that this is a one-level dictionary
            self._observations = {}
            self._next_obs = {}

            for key, dims in self._observation_dim.items():
                if isinstance(dims, tuple):
                    dims = tuple([self._max_replay_buffer_size] + list(dims))
                else:
                    dims = (self._max_replay_buffer_size, dims)
                self._observations[key] = np.zeros(dims)
                self._next_obs[key] = np.zeros(dims)
        else:
            # else observation_dim is an integer
            self._observations = np.zeros(
                (self._max_replay_buffer_size, self._observation_dim)
            )
            self._next_obs = np.zeros(
                (self._max_replay_buffer_size, self._observation_dim)
            )

        self._actions = np.zeros((self._max_replay_buffer_size, self._action_dim))
        self.best_actions = np.zeros((self._max_replay_buffer_size, self._action_dim))

        if self.diffusion_steps_needed == True:
            self.diffusion_actions = np.zeros(
                (
                    self.max_replay_buffer_size,
                    self.action_dim,
                    self.diffusion_n_timesteps,
                )
            )

        # Make everything a 2D np array to make it easier for other code to
        # reason about the shape of the data
        self._rewards = np.zeros((self._max_replay_buffer_size, 1))
        # self._terminals[i] = a terminal was received at time i
        self._terminals = np.zeros((self._max_replay_buffer_size, 1), dtype="uint8")
        self._timeouts = np.zeros((self._max_replay_buffer_size, 1), dtype="uint8")
        # absorbing[0] is if obs was absorbing, absorbing[1] is if next_obs was absorbing
        self._absorbing = np.zeros((self._max_replay_buffer_size, 2))
        self._top = 0
        self._size = 0
        self._trajs = 0
        self.indices = None
        # keeping track of trajectory boundaries
        # assumption is trajectory lengths are AT MOST the length of the entire replay buffer
        self._cur_start = 0
        self._traj_endpoints = {}  # start->end means [start, end)


def concat_nested_dicts(d1, d2):
    # two dicts that have the exact same nesting structure
    # and contain leaf values that are numpy arrays of the same
    # shape except for the first dimensions
    return {
        k: (
            np.concatenate((d1[k], d2[k]), axis=0)
            if not isinstance(d1[k], dict)
            else concat_nested_dicts(d1[k], d2[k])
        )
        for k in d1
    }

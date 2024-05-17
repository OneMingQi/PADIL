import numpy as np


def goal_distance(goal_a, goal_b, distance_threshold=0.05):
    return np.linalg.norm(goal_a - goal_b, ord=2, axis=-1)


def compute_reward(achieved, goal, info, distance_threshold=0.05):
    dis = goal_distance(achieved, goal)
    return -(dis > distance_threshold).astype(np.float32)


def goal_distance_obs(obs, distance_threshold=0.05):
    return goal_distance(
        obs["achieved_goal"], obs["desired_goal"], distance_threshold=distance_threshold
    )


def compute_distance(achieved, goal):
    return np.sqrt(np.sum(np.square(achieved - goal)))

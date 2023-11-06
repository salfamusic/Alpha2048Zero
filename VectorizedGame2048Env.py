import gym
import numpy as np
from gym.vector.utils import spaces

from Game2048Env import Game2048Env, MAX_TILE


class VectorizedGame2048Env(gym.Env):
    def __init__(self, num_envs):
        super(VectorizedGame2048Env, self).__init__()
        self.num_envs = num_envs
        self.envs = [Game2048Env() for _ in range(self.num_envs)]
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=MAX_TILE, shape=(num_envs, 4, 4, 5), dtype=np.uint16)

        self.reset()

    def reset(self, **kwargs):
        return np.array([env.reset() for env in self.envs]).reshape((self.num_envs, 4, 4, 5))

    def observation(self):
        return np.array([Game2048Env.make_observation(env.board, Game2048Env.is_game_over(env.board)) for env in self.envs]).reshape((self.num_envs, 4, 4, 5))

    def reset_single_game(self, index):
        """
        Resets a single game within the vectorized environment.

        Parameters:
        - index (int): The index of the game to reset.

        Returns:
        - numpy array: The initial state of the reset game.
        """
        return self.envs[index].reset()

    def step(self, actions):
        observations, rewards, dones, infos = [], [], [], []
        for env, action in zip(self.envs, actions):
            observation, reward, done, info = env.step(action)
            observations.append(observation)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
        return np.array(observations).reshape((self.num_envs,4,4,5)), np.array(rewards), np.array(dones), infos

    def render(self, mode='human'):
        for env in self.envs:
            env.render(mode=mode)

    def close(self):
        for env in self.envs:
            env.close()

    @classmethod
    def legal_actions_mask(cls, states):
        return np.array([Game2048Env.legal_actions_mask(state) for state in states])

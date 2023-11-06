import numpy as np


class PPOExperienceBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.logits = []
        self.dones = []

    def store(self, state, action, reward, next_state, logits, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.logits.append(logits)
        self.dones.append(done)

    def normalize_rewards_per_episode(self):
        start_idx = 0
        for idx, done in enumerate(self.dones):
            if done:
                self._normalize_rewards_slice(start_idx, idx + 1)
                start_idx = idx + 1

    def _normalize_rewards_slice(self, start_idx, end_idx):
        slice_rewards = self.rewards[start_idx:end_idx]
        if len(slice_rewards) == 0:
            return

        reward_mean = sum(slice_rewards) / len(slice_rewards)
        reward_std = (sum([(r - reward_mean) ** 2 for r in slice_rewards]) / len(slice_rewards)) ** 0.5

        if reward_std == 0:
            norm_rewards = [r - reward_mean for r in slice_rewards]
        else:
            norm_rewards = [(r - reward_mean) / reward_std for r in slice_rewards]

        self.rewards[start_idx:end_idx] = norm_rewards

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.logits = []
        self.dones = []

    def size(self):
        return len(self.states)
    
    def print_reward_mean(self):
        average_reward = np.mean(self.rewards)
        print(f"{average_reward=}")

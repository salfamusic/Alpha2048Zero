import math
import numpy as np
import random


class PPOExperienceBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.values = []
        self.next_values = []
        self.rewards = []
        self.next_states = []
        self.logits = []
        self.dones = []

    def store(self, state, action, value, reward, next_state, logits, done):
        if len(self.dones) != 0 and not self.dones[-1]:
            self.next_values.append(value)
        self.states.append(state)
        self.actions.append(action)
        self.values.append(value)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.logits.append(logits)
        self.dones.append(done)

    def clear(self):
        self.states = []
        self.actions = []
        self.values = []
        self.next_values = []
        self.rewards = []
        self.next_states = []
        self.logits = []
        self.dones = []

    def size(self):
        return len(self.states)
    
    def print_reward_mean(self):
        average_reward = np.mean(self.rewards)
        print(f"{average_reward=}")

    def get_num_batches(self, batch_size):
        """
        Generator that yields batches of experiences.

        Parameters:
        batch_size (int): The size of the batch to yield.
        """
        n_samples = self.size()

        return math.ceil(n_samples / batch_size)


    def get_batches(self, batch_size):
        """
        Generator that yields batches of experiences.

        Parameters:
        batch_size (int): The size of the batch to yield.
        """
        n_samples = self.size()
        indices = np.arange(n_samples)
        np.random.shuffle(indices)

        for start_idx in range(0, n_samples, batch_size):
            batch_indices = indices[start_idx:start_idx + batch_size]
            batch = (np.array(self.states)[batch_indices],
                     np.array(self.actions)[batch_indices],
                     np.array(self.values)[batch_indices],
                     np.array(self.next_values)[batch_indices],
                     np.array(self.rewards)[batch_indices],
                     np.array(self.next_states)[batch_indices],
                     np.array(self.logits)[batch_indices],
                     np.array(self.dones)[batch_indices])
            yield batch

    def slice(self, start_idx, end_idx):
        return (np.array(self.states)[start_idx:end_idx],
                     np.array(self.actions)[start_idx:end_idx],
                     np.array(self.values)[start_idx:end_idx],
                     np.array(self.next_values)[start_idx:end_idx],
                     np.array(self.rewards)[start_idx:end_idx],
                     np.array(self.next_states)[start_idx:end_idx],
                     np.array(self.logits)[start_idx:end_idx],
                     np.array(self.dones)[start_idx:end_idx])

    def finalize_episode_with_next_values(self, next_value):
        self.next_values.append(next_value)

    def copy(self):
        copied_buffer = PPOExperienceBuffer()

        copied_buffer.states = list(self.states)
        copied_buffer.actions = list(self.actions)
        copied_buffer.values = list(self.values)
        copied_buffer.next_values = list(self.next_values)
        copied_buffer.rewards = list(self.rewards)
        copied_buffer.next_states = list(self.next_states)
        copied_buffer.logits = list(self.logits)
        copied_buffer.dones = list(self.dones)

        return copied_buffer
    
    def concat(self, other):
        copied_buffer = PPOExperienceBuffer()

        copied_buffer.states = list(self.states) + list(other.states)
        copied_buffer.actions = list(self.actions) + list(other.actions)
        copied_buffer.values = list(self.values) + list(other.values)
        copied_buffer.next_values = list(self.next_values) + list(other.next_values)
        copied_buffer.rewards = list(self.rewards) + list(other.rewards)
        copied_buffer.next_states = list(self.next_states) + list(other.next_states)
        copied_buffer.logits = list(self.logits) + list(other.logits)
        copied_buffer.dones = list(self.dones) + list(other.dones)

        return copied_buffer

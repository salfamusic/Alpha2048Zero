import math
import random
import numpy as np

class PPOTrainingBuffer:
    def __init__(self, states, y_true_policies, returns):
        self.states = states
        self.y_true_policies = y_true_policies
        self.returns = returns

    def size(self):
        return len(self.states)
    
    def get_random_indexer(self):
        indices = np.arange(self.size())
        np.random.shuffle(indices)

        return indices

    def get_number_of_batches(self, batch_size):
        n_samples = self.size()

        return math.ceil(n_samples / batch_size)

    def get_batches(self, batch_size, indexer):
        """
        Generator that yields batches of experiences.

        Parameters:
        batch_size (int): The size of the batch to yield.
        """
        n_samples = self.size()

        for start_idx in range(0, n_samples, batch_size):
            batch_indices = indexer[start_idx:start_idx + batch_size]
            batch = (np.array(self.states)[batch_indices],
                     np.array(self.y_true_policies)[batch_indices],
                     np.array(self.returns)[batch_indices]
            )
            yield batch

    def add_step(self, state, y_true_policy, return_):
        self.states.append(state)
        self.y_true_policies.append(y_true_policy)
        self.returns.append(return_)

    def get_step(self, step):
        return (np.array(self.states)[step],
                     np.array(self.y_true_policies)[step],
                     np.array(self.returns)[step])

    def slice(self, start_idx, end_idx):
        return (np.array(self.states)[start_idx:end_idx],
                     np.array(self.y_true_policies)[start_idx:end_idx],
                     np.array(self.returns)[start_idx:end_idx])
    
    def add_slice(self, states, y_true_policies, returns):
        self.states.extend(states)
        self.y_true_policies.extend(y_true_policies)
        self.returns.extend(returns)

    def add_from_buffer(self, other_buffer):
        self.states.extend(other_buffer.states)
        self.y_true_policies.extend(other_buffer.y_true_policies)
        self.returns.extend(other_buffer.returns)
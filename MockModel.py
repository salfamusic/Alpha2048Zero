import numpy as np


class MockModel:
    def predict(self, inputs, **kwargs):
        batch_size = inputs[0].shape[0]

        # Generate random value predictions between -1 and 1
        value = 2 * np.random.rand(batch_size, 1) - 1

        # Generate random policy logits
        policy_logits = np.random.rand(batch_size, 4)

        # Convert logits to probabilities
        policy = np.exp(policy_logits) / np.sum(np.exp(policy_logits), axis=1, keepdims=True)

        return value, policy

    def fit(self, *args, **kwargs):
        pass

    def __call__(self, inputs, **kwargs):
        batch_size = inputs[0].shape[0]

        # Generate random value predictions between -1 and 1
        value = 2 * np.random.rand(batch_size, 1) - 1

        # Generate random policy logits
        policy_logits = np.random.rand(batch_size, 4)

        # Convert logits to probabilities
        policy = np.exp(policy_logits) / np.sum(np.exp(policy_logits), axis=1, keepdims=True)

        return value, policy

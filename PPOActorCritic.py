import numpy as np
import tensorflow as tf
from keras.src.layers import Conv2D
from tensorflow import keras
from keras.layers import Dense, Flatten, Input

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

class PPOActorCritic:
    def __init__(self, observation_shape, action_dim):
        self.observation_shape = observation_shape
        self.action_dim = action_dim

        self.actor = self.build_actor()
        self.critic = self.build_critic()

    def build_actor(self):
        observations = Input(shape=self.observation_shape)
        # Convolutional layers
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(observations)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)

        # Flatten
        x = Flatten()(x)

        # Dense layers
        x = Dense(128, activation='relu')(x)
        probs = Dense(self.action_dim, activation='softmax')(x)
        model = keras.Model(inputs=observations, outputs=probs)
        return model

    def build_critic(self):
        observations = Input(shape=self.observation_shape)
        x = Flatten()(observations)
        x = Dense(512, activation='relu')(x)
        x = Dense(256, activation='relu')(x)
        values = Dense(1)(x)
        model = keras.Model(inputs=observations, outputs=values)
        return model

    def predict(self, state, legal_actions_mask):
        """Return softmax probabilities for legal actions and the value estimate for the state."""
        logits = self.actor(state[np.newaxis, :])[0]
        masked_logits = logits * legal_actions_mask + (1 - legal_actions_mask) * -1e32  # Set illegal actions' logits to large negative values
        normalized_probs = softmax(masked_logits)
        value_estimate = self.critic(state[np.newaxis, :])[0][0]
        return value_estimate, normalized_probs

    def __call__(self, inputs, **kwargs):
        return self.predict(inputs[0], inputs[1])
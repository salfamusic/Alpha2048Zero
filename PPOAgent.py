from xml.sax.xmlreader import InputSource
import numpy as np
from keras import layers, Model, optimizers
import keras.backend as K
import tensorflow as tf


import numpy as np
from keras import layers, Model, optimizers
import keras.backend as K
import tensorflow as tf
from keras.src.regularizers import l2

# Define the L2 regularization factor
l2_reg = 0.001


class PPOAgent:
    def __init__(self, input_shape, action_space, gamma=0.99, lr=1e-3, clip_value=0.2, value_coeff=0.5, gae_lambda=0.95,
                 entropy_coeff=0.01, temperature=1.0, lr_final=1e-5, entropy_final=1e-4, decay_steps=10000, dropout_rate=0.5):
        self.input_shape = input_shape
        self.gamma = gamma
        self.clip_value = clip_value
        self.value_coeff = value_coeff
        self.gae_lambda = gae_lambda
        self.entropy_coeff = entropy_coeff
        self.action_space = action_space
        self.temperature = temperature
        self.dropout_rate = dropout_rate

        inputs = layers.Input(shape=input_shape, name='board_input')

        # Split the input into n separate inputs
        input_slices = [layers.Lambda(lambda x: x[:, i])(inputs) for i in range(input_shape[0])]

        # Define n separate CNNs
        cnn_outputs = []
        for i, input_slice in enumerate(input_slices):
            x = layers.Conv2D(16, (2, 2), activation='relu', padding='same', name=f'cnn{i+1}_1',
                              kernel_regularizer=l2(l2_reg))(input_slice)
            x = layers.Dropout(self.dropout_rate)(x)  # Add dropout here
            x = layers.Conv2D(16, (2, 2), activation='relu', padding='same', name=f'cnn{i+1}_2',
                              kernel_regularizer=l2(l2_reg))(x)
            x = layers.Dropout(self.dropout_rate)(x)  # Add dropout here
            x = layers.Conv2D(16, (2, 2), activation='relu', padding='same', name=f'cnn{i+1}_3',
                              kernel_regularizer=l2(l2_reg))(x)
            x = layers.Dropout(self.dropout_rate)(x)  # Add dropout here
            x = layers.Flatten(name=f'flatten{i+1}')(x)
            cnn_outputs.append(x)

        # If you want to merge the outputs
        merged_output = layers.Concatenate()(cnn_outputs) if len(cnn_outputs) > 1 else cnn_outputs[0]

        # Add dropout after concatenating the CNN outputs
        z = layers.Dropout(self.dropout_rate)(merged_output)

        # Dense layers with dropout and L2 regularization
        z = layers.Dense(1024, activation='relu', name='dense1', kernel_regularizer=l2(l2_reg))(z)
        z = layers.Dropout(self.dropout_rate)(z)  # Add dropout here
        z = layers.Dense(512, activation='relu', name='dense2', kernel_regularizer=l2(l2_reg))(z)
        z = layers.Dropout(self.dropout_rate)(z)  # Add dropout here

        logits = layers.Dense(action_space, activation=None, name='logits')(z)
        value = layers.Dense(1, activation=None, name='value')(z)

        # Create the model
        self.model = Model(inputs=inputs, outputs=[logits, value])
        self.model.compile(optimizer=optimizers.Adam(learning_rate=lr),
                           loss={'logits': self._ppo_loss, 'value': 'mean_squared_error'})


        # Learning rate and entropy decay
        self.initial_lr = lr
        self.lr = lr
        self.initial_entropy_coeff = entropy_coeff
        self.entropy_coeff = entropy_coeff
        self.lr_final = lr_final
        self.entropy_final = entropy_final
        self.decay_steps = decay_steps
        self.step = 0

        # Adjust the optimizer learning rate
        self.model.optimizer.lr.assign(self.lr)

    def _ppo_loss(self, y_true, y_pred):
        # Splitting the y_true tensor to get old probabilities, advantages, and actions taken
        old_prob, advantage, action = tf.split(y_true, [self.action_space, 1, 1], axis=-1)

        # Calculate the current policy probabilities using the current logits (y_pred)
        prob = K.softmax(y_pred)

        # Get the probabilities of the taken actions from prob and old_prob tensors
        prob_taken = tf.reduce_sum(prob * action, axis=-1)
        old_prob_taken = tf.reduce_sum(old_prob * action, axis=-1)

        # Compute the ratio of the new and old probabilities
        ratio = prob_taken / (old_prob_taken + 1e-8)
        clip_ratio = K.clip(ratio, 1.0 - self.clip_value, 1.0 + self.clip_value)

        # Compute the two surrogate functions
        surrogate1 = ratio * advantage
        surrogate2 = clip_ratio * advantage

        # Entropy regularization term
        entropy = -K.mean(prob * K.log(K.maximum(prob, 1e-8)))

        # Combine and return the PPO loss
        return -K.mean(K.minimum(surrogate1, surrogate2) - self.entropy_coeff * entropy)

    def predict(self, state, legal_moves_mask):
        logits, value = self.model(self.log2(state.reshape(1, *self.input_shape)))
        logits_array = logits.numpy()
        # Apply temperature scaling
        logits_array = logits_array / self.temperature  # Add this line

        # Apply mask to logits by setting illegal move logits to a large negative value
        logits_array[~legal_moves_mask.reshape(1,4)] = -1e10

        probs = tf.nn.softmax(logits_array).numpy()

        return value[0][0], probs[0]

    def select_action(self, state, legal_moves_mask):
        """
        Selects an action based on the network's output and a legal moves mask.

        Parameters:
            state (numpy array): The current board state.
            legal_moves_mask (numpy array): A boolean mask indicating legal moves.

        Returns:
            int: The selected action.
        """
        _, probs = self.predict(state, legal_moves_mask)

        # Sample an action from the probability distribution
        action = np.random.choice(self.action_space, p=probs)

        return action

    def _decay_entropy_coeff(self):
        # Linear decay towards the minimum
        decayed_entropy = (self.initial_entropy_coeff - self.entropy_final) * \
                          max(0, 1 - self.step / self.decay_steps) + self.entropy_final
        print(f'{decayed_entropy=}')
        return decayed_entropy

    def _decay_lr(self):
        # Linear decay towards the minimum
        decayed_lr = (self.initial_lr - self.lr_final) * \
                     max(0, 1 - self.step / self.decay_steps) + self.lr_final
        print(f'{decayed_lr=}')
        return decayed_lr

    def train(self, states, actions, rewards, next_states, old_logits, dones):
        # Increment the step counter
        self.step += 1

        # Update the entropy coefficient and learning rate
        self.entropy_coeff = self._decay_entropy_coeff()
        self.lr = self._decay_lr()
        K.set_value(self.model.optimizer.lr, self.lr)

        # Prepare the data
        states = self.log2(np.array(states).reshape((len(states), *self.input_shape)))
        actions = np.array(actions).reshape((len(actions), 1))
        rewards = np.array(rewards).reshape((len(rewards), 1))
        next_states = self.log2(np.array(next_states).reshape((len(next_states), *self.input_shape)))
        old_logits = np.array(old_logits).reshape((len(old_logits), self.action_space))
        dones = np.array(dones).reshape((len(dones), 1))

        # Predict the current values and the next values
        values = self.model.predict(states, verbose=0)[1]
        next_values = self.model.predict(next_states, verbose=0)[1]

        # Compute TD residuals
        deltas = rewards + self.gamma * next_values * (1. - dones) - values

        # Initialize the advantages and returns
        advantages = np.zeros_like(rewards)
        returns = np.zeros_like(rewards)

        # Compute GAE and returns in reverse order
        advantages[-1] = deltas[-1]  # For the last step, advantage is just the delta
        returns[-1] = rewards[-1] + self.gamma * (1.0 - dones[-1]) * next_values[-1]

        for t in reversed(range(len(rewards) - 1)):
            next_non_terminal = 1.0 - dones[t + 1]
            advantages[t] = deltas[t] + self.gamma * self.gae_lambda * next_non_terminal * advantages[t + 1]
            returns[t] = advantages[t] + values[t]



        # The true output for the policy network
        y_true_policy = np.concatenate([old_logits, advantages, actions], axis=-1)

        # Train the model and capture the losses/metrics
        loss_metrics = self.model.train_on_batch(states, [y_true_policy, returns])

        # Log the losses/metrics
        ppo_loss, value_loss, _ = loss_metrics
        print(f'PPO Loss: {ppo_loss:.4f}, Value Loss: {value_loss:.4f}')

    def get_unmasked_logits(self, state):
        logits, _ = self.model(self.log2(state.reshape((1,*self.input_shape))))
        return logits

    def log2(self, arr):
        # Sample matrix
        # Create a mask for entries that are 0
        mask = arr == 0

        # Compute the logarithm
        log_matrix = np.log2(arr, out=np.zeros_like(arr, dtype='float32'), where=(arr!=0))

        # Apply the mask
        log_matrix[mask] = 0
        return log_matrix

    def __call__(self, inputs, **kwargs):
        return self.predict(inputs[0], inputs[1])



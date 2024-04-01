from xml.sax.xmlreader import InputSource
import numpy as np
from keras import layers, Model, optimizers
import keras.backend as K
import tensorflow as tf
from PPOTrainingBuffer import PPOTrainingBuffer

import numpy as np
from keras import layers, Model, optimizers
import keras.backend as K
import tensorflow as tf
from keras.src.regularizers import l2
import keras.models
from tensorflow.keras.models import clone_model


# Define the L2 regularization factor
l2_reg = 0.01

# def __init__(self, input_shape, action_space, gamma=0.99, lr=1e-3, clip_value=0.2, value_coeff=1/1000, gae_lambda=0.9,
#                  entropy_coeff=0.01, temperature=1.0, lr_final=1e-5, entropy_final=1e-4, decay_steps=10000, dropout_rate=0.5):
class PPOAgent:
    def __init__(self, input_shape, action_space, gamma=0.99, lr=1e-4, clip_value=0.1, value_coeff=1, gae_lambda=0.9,
                entropy_coeff=0.01, temperature=1.0, lr_final=5e-6, entropy_final=1e-4, decay_steps=10000, dropout_rate=0.5):
        self.input_shape = input_shape
        self.gamma = gamma
        self.clip_value = clip_value
        self.value_coeff = value_coeff
        self.gae_lambda = gae_lambda
        self.entropy_coeff = entropy_coeff
        self.action_space = action_space
        self.temperature = temperature
        self.dropout_rate = dropout_rate

        self.clip_range = clip_value

        # Learning rate and entropy decay
        self.initial_lr = lr
        self.lr = lr
        self.initial_entropy_coeff = entropy_coeff
        self.entropy_coeff = entropy_coeff
        self.lr_final = lr_final
        self.entropy_final = entropy_final
        self.decay_steps = decay_steps
        self.step = 0

        
        
        # actor_inputs, actor_base_model = self._build_base_model(input_shape)
        # critic_inputs, critic_base_model = self._build_base_model(input_shape)

        # self.actor = self._build_actor(actor_inputs, actor_base_model, action_space, self.lr)
        # self.critic = self._build_critic(critic_inputs, critic_base_model, self.lr, self.value_coeff)

        self.actor, self.critic = self._build_models(input_shape, action_space, lr, clip_value, value_coeff, gae_lambda, entropy_coeff)

    def _build_models(self, input_shape, action_space, lr, clip_value, value_coeff, gae_lambda, entropy_coeff):
        inputs = layers.Input(shape=input_shape, name='board_input')

        # Convolutional layers
        x = layers.Conv2D(64, (2, 2), kernel_regularizer=l2(l2_reg), activation='relu', padding='same')(inputs)
        x = layers.Conv2D(128, (2, 2), kernel_regularizer=l2(l2_reg), activation='relu', padding='same')(x)

        # Flattening the convolutions
        x = layers.Flatten()(x)

        # Dense layer
        x = layers.Dense(256, kernel_regularizer=l2(l2_reg), activation='relu')(x)

        # Actor network
        logits = layers.Dense(action_space, activation=None)(x)
        actor = Model(inputs=inputs, outputs=logits)
        actor.compile(optimizer=optimizers.Adam(learning_rate=lr), loss=self._ppo_loss)

        # Critic network
        values = layers.Dense(1, activation=None)(x)
        critic = Model(inputs=inputs, outputs=values)
        critic.compile(optimizer=optimizers.Adam(learning_rate=lr), loss=self._value_loss(value_coeff))

        return actor, critic

    def _build_base_model(self, input_shape):
        inputs = layers.Input(shape=input_shape, name='board_input')
        x = layers.Conv3D(256, (1, 2, 2), activation='relu', padding='same', kernel_regularizer=l2(l2_reg), name=f'cnn_1')(inputs)
        x = layers.Conv3D(256, (input_shape[0], 3, 3), activation='relu', padding='same', kernel_regularizer=l2(l2_reg), name=f'cnn_2')(x)
        x = layers.Conv3D(256, (input_shape[0], 3, 3), activation='relu', padding='same', kernel_regularizer=l2(l2_reg), name=f'cnn_3')(x)
        x = layers.GlobalAveragePooling3D()(x)
        x = layers.Dense(1024, activation='relu', kernel_regularizer=l2(l2_reg), name='densex1')(x)
        x = layers.Dense(512, activation='relu', kernel_regularizer=l2(l2_reg), name='densex2')(x)

        return inputs, x


    def _build_actor(self, base_model_inputs, base_model, action_space, lr):
        logits = layers.Dense(action_space, activation=None, name='logits')(base_model)

        # Create the model
        actor = Model(inputs=base_model_inputs, outputs=[logits])
        actor.compile(optimizer=optimizers.Adam(learning_rate=lr),
                           loss={'logits': self._ppo_loss})
        actor.optimizer.lr.assign(lr)

        return actor


    def _build_critic(self, base_model_inputs, base_model, lr, value_coeff):
        value = layers.Dense(1, activation=None, name='value')(base_model)

        # Create the model
        critic = Model(inputs=base_model_inputs, outputs=[value])
        critic.compile(optimizer=optimizers.Adam(learning_rate=lr),
                           loss=self._value_loss(value_coeff))
        critic.optimizer.lr.assign(lr)

        return critic

    def _value_loss(self, value_coeff):
        def loss(y_true, y_pred):
            # Clipped value loss
            value_pred_clipped = y_true + K.clip(y_pred - y_true, -self.clip_range, self.clip_range)
            value_loss = K.square(y_true - y_pred)
            value_loss_clipped = K.square(value_pred_clipped - y_pred)
            return value_coeff * K.mean(K.maximum(value_loss, value_loss_clipped))
        return loss

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

    def predict_with_logits(self, state, legal_moves_mask):
        model_input = state.reshape(1, *self.input_shape)
        logits = self.actor(model_input)
        value = self.critic(model_input)
        logits_array = logits.numpy()
        # Apply temperature scaling
        unmasked_logits = logits_array.copy()

        # Apply mask to logits by setting illegal move logits to a large negative value
        logits_array[~legal_moves_mask.reshape(1,4)] = -1e10

        return value[0][0], tf.nn.softmax(logits_array).numpy()[0], logits_array, unmasked_logits

    def predict(self, state, legal_moves_mask):
        value, probs, _, _ = self.predict_with_logits(state, legal_moves_mask)

        return value, probs

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

    def select_action_and_logits(self, state, legal_moves_mask):
        """
        Selects an action based on the network's output and a legal moves mask.

        Parameters:
            state (numpy array): The current board state.
            legal_moves_mask (numpy array): A boolean mask indicating legal moves.

        Returns:
            int: The selected action.
        """
        _, probs, logits, _ = self.predict_with_logits(state, legal_moves_mask)

        # Sample an action from the probability distribution
        action = np.random.choice(self.action_space, p=probs)

        return action, logits
    
    def select_action_value_and_logits(self, state, legal_moves_mask):
        """
        Selects an action based on the network's output and a legal moves mask.

        Parameters:
            state (numpy array): The current board state.
            legal_moves_mask (numpy array): A boolean mask indicating legal moves.

        Returns:
            int: The selected action.
        """
        value, probs, logits, _ = self.predict_with_logits(state, legal_moves_mask)

        # Sample an action from the probability distribution
        action = np.random.choice(self.action_space, p=probs)

        return action, value, logits
    
    def select_action_value_and_unmasked_logits(self, state, legal_moves_mask):
        """
        Selects an action based on the network's output and a legal moves mask.

        Parameters:
            state (numpy array): The current board state.
            legal_moves_mask (numpy array): A boolean mask indicating legal moves.

        Returns:
            int: The selected action.
        """
        value, probs, _, unmasked_logits = self.predict_with_logits(state, legal_moves_mask)

        # Sample an action from the probability distribution
        action = np.random.choice(self.action_space, p=probs)

        return action, value, unmasked_logits, probs

    def _decay_entropy_coeff(self):
        # Linear decay towards the minimum
        decayed_entropy = (self.initial_entropy_coeff - self.entropy_final) * \
                          max(0, 1 - self.step / self.decay_steps) + self.entropy_final
        return decayed_entropy

    def _decay_lr(self):
        # Linear decay towards the minimum
        decayed_lr = (self.initial_lr - self.lr_final) * \
                     max(0, 1 - self.step / self.decay_steps) + self.lr_final
        return decayed_lr

    def make_training_data(self, states, actions, values, next_values, rewards, next_states, old_logits, dones):
        # Prepare the data
        states = np.array(states).reshape((len(states), *self.input_shape))
        actions = np.array(actions).reshape((len(actions), 1))
        values = np.array(values).reshape((len(values), 1))
        next_values = np.array(next_values).reshape((len(next_values), 1))
        rewards = np.array(rewards).reshape((len(rewards), 1))
        next_states = np.array(next_states).reshape((len(next_states), *self.input_shape))
        old_logits = np.array(old_logits).reshape((len(old_logits), self.action_space))
        dones = np.array(dones).reshape((len(dones), 1))

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


        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # The true output for the policy network
        y_true_policy = np.concatenate([old_logits, advantages, actions], axis=-1)

        return states, y_true_policy, returns

    def train(self, states, y_true_policy, returns):
        # Increment the step counter
        self.step += 1

        # Update the entropy coefficient and learning rate
        self.entropy_coeff = self._decay_entropy_coeff()
        self.lr = self._decay_lr()
        K.set_value(self.actor.optimizer.lr, self.lr)
        K.set_value(self.critic.optimizer.lr, self.lr)

        # Train the model and capture the losses/metrics
        actor_loss_metrics = self.actor.train_on_batch(states, y_true_policy)
        critic_loss_metrics = self.critic.train_on_batch(states, returns)

        # Log the losses/metrics
        ppo_loss = actor_loss_metrics
        value_loss = critic_loss_metrics
        return ppo_loss, value_loss, self.lr, self.entropy_coeff

    def get_unmasked_logits(self, state):
        logits = self.actor(state.reshape((1,*self.input_shape)))
        return logits
    
    def save_models(self, file_name):
        self.actor.save(f'{file_name}_actor_model.keras')
        self.critic.save(f'{file_name}_critic_model.keras')

    def load_models(self, file_name):
        self.actor = keras.models.load_model(f'{file_name}_actor_model.keras', compile=True)
        self.critic = keras.models.load_model(f'{file_name}_critic_model.keras', compile=True)

    def __call__(self, inputs, **kwargs):
        return self.predict(inputs[0], inputs[1])



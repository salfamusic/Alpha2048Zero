import numpy as np
import tensorflow as tf
from tqdm import trange

from Game2048Env import Game2048Env
from MCTS2048 import MCTS2048
from PPOActorCritic import PPOActorCritic
from webapp2048.WebApp import call_add_step

# Hyperparameters
GAMMA = 0.99
EPOCHS = 10
BATCH_SIZE = 64
EPSILON = 0.2
CLIP_VALUE = 0.2
ACTOR_LR = 1e-4
CRITIC_LR = 1e-4

# Using Adam optimizer for both actor and critic
actor_optimizer = tf.keras.optimizers.Adam(learning_rate=ACTOR_LR)
critic_optimizer = tf.keras.optimizers.Adam(learning_rate=CRITIC_LR)


# Helper functions
def compute_discounted_returns(rewards):
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + GAMMA * G
        returns.insert(0, G)
    return np.array(returns)


def do_epoch_actor_critic(batch_size, env, epoch, mcts, model):
    # Lists to store training data
    # 1. Data Collection
    states = []
    actions = []
    rewards = []
    dones = []
    next_states = []

    for game in range(batch_size):
        state = mcts.reset_env()

        while True:
            action = mcts.search(mcts.root)

            if not env.is_action_legal(env.board, action):
                print("illegal")

            next_state, reward, done, extra_info = env.step(action)
            next_random_action = extra_info['random_action']

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            next_states.append(next_state)

            mcts.move_to_child(next_state, action, next_random_action)

            state = next_state

            try:
                call_add_step({
                    'epoch_number': epoch,
                    'game_number': game,
                    'action': action,
                    'next_board': next_state.tolist()
                })
            except Exception as e:
                print(e)

            if done:
                break

    returns = compute_discounted_returns(rewards)

    # 3. Policy Update
    with tf.GradientTape() as actor_tape, tf.GradientTape() as critic_tape:
        # Old probabilities
        old_probs = model.actor(np.array(states))
        old_probs = old_probs[np.arange(BATCH_SIZE), actions]

        # Value estimates
        values = model.critic(np.array(states))

        # New probabilities
        new_probs = model.actor(np.array(states))
        new_probs = new_probs[np.arange(BATCH_SIZE), actions]

        # PPO Policy loss
        ratios = new_probs / (old_probs + 1e-5)
        advantages = returns - values[:, 0]
        unclipped_loss = ratios * advantages
        clipped_loss = tf.clip_by_value(ratios, 1 - EPSILON, 1 + EPSILON) * advantages
        actor_loss = -tf.reduce_mean(tf.minimum(unclipped_loss, clipped_loss))

        # Value loss
        critic_loss = tf.reduce_mean(tf.square(returns - values[:, 0]))

    # Calculate and apply gradients for actor and critic
    actor_grads = actor_tape.gradient(actor_loss, model.actor.trainable_variables)
    critic_grads = critic_tape.gradient(critic_loss, model.critic.trainable_variables)

    actor_optimizer.apply_gradients(zip(actor_grads, model.actor.trainable_variables))
    critic_optimizer.apply_gradients(zip(critic_grads, model.critic.trainable_variables))

    print(f"Epoch {epoch + 1}/{EPOCHS}, Actor Loss: {actor_loss.numpy()}, Critic Loss: {critic_loss.numpy()}")

def train_2048_ppo_actor_critic(model, env, mcts, epochs, batch_size):
    """
    Train the model for the 2048 game using the given environment and MCTS.

    Parameters:
    - model: Neural network model
    - env: 2048 game environment
    - mcts: MCTS object
    - epochs: Number of training epochs
    - games_per_epoch: Number of games to play per epoch
    - mcts_iterations: Number of MCTS iterations per move
    - batch_size: Batch size for training the model

    Returns:
    - model: Trained model
    """

    mcts.switch_to_primary_model()

    for epoch in trange(epochs):
        do_epoch_actor_critic(batch_size, env, epoch, mcts, model)

    return model

def train_ppo_actor_critic(*, epochs, mcts_iterations, batch_size):
    # Create the 2048 environment, model, and MCTS
    env = Game2048Env()
    model = PPOActorCritic(env.observation_space.shape, env.action_space.n)
    mcts = MCTS2048(model, mcts_iterations, env)

    # Train the model
    return train_2048_ppo_actor_critic(model, env, mcts, epochs, batch_size)

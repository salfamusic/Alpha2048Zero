import numpy as np
import tensorflow as tf
from tqdm.notebook import tqdm

from Game2048Env import Game2048Env
from MCTS2048 import MCTS2048
from PPOAgent import PPOAgent
from PPOExperienceBuffer import PPOExperienceBuffer
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



def do_epoch(batch_size, env, epoch, mcts, model):
    buffer = PPOExperienceBuffer()

    game_count = 0
    # pbar = tqdm(total=batch_size, desc="Buffer Filling", position=0, leave=True)
    while buffer.size() < batch_size:
        game_count += 1
        # state = mcts.reset_env()
        state = env.reset()

        while True:
            # action = mcts.search(mcts.root)
            action = model.select_action(state, env.legal_actions_mask_from_board(env.board))

            if not env.is_action_legal(env.board, action):
                print("illegal")

            logits = model.get_unmasked_logits(state)
            next_state, reward, done, extra_info = env.step(action)
            # next_random_action = extra_info['random_action']

            buffer.store(state, action, reward, next_state, logits, done)

            # mcts.move_to_child(next_state, action, next_random_action)
            # pbar.update(1)
            try:
                call_add_step(epoch, game_count, action, env.board.tolist())
            except Exception as e:
                print(e)

            state = next_state
            if done:
                break

    # pbar.close()
    model.train(buffer.states, buffer.actions, buffer.rewards, buffer.next_states, buffer.logits, buffer.dones)
    buffer.print_reward_mean()
    buffer.clear()



def train_2048_ppo(model, env, mcts, epochs, batch_size):
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

    for epoch in range(epochs):
        do_epoch(batch_size, env, epoch, mcts, model)

    return model

def train_ppo(*, epochs, mcts_iterations, batch_size, temperature, window_size):
    # Create the 2048 environment, model, and MCTS
    env = Game2048Env(window_size=window_size)
    model = PPOAgent(env.observation_space.shape, env.action_space.n, temperature=temperature)
    mcts = MCTS2048(model, mcts_iterations, env)

    # Train the model
    return train_2048_ppo(model, env, mcts, epochs, batch_size)

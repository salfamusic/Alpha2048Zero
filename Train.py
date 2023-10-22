from Game2048Env import Game2048Env
from MCTS2048 import MCTS2048
from Node import Node
from Policy2048 import create_2048_policy_model
from tqdm.notebook import tqdm, trange
import numpy as np

def train_2048(model, env, mcts, epochs, games_per_epoch, mcts_iterations, batch_size):
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

    for epoch in trange(epochs):
        # Lists to store training data
        states, action_masks, action_probs, values = [], [], [], []

        for game in range(games_per_epoch):
            state = mcts.reset_env()
            action_mask = env.legal_actions_mask(state)
            game_value = 0

            while True:
                action = mcts.search(mcts.root)

                if not env.is_action_legal(env.board, action):
                    print("illegal")

                next_state, reward, done, extra_info = env.step(action)
                next_random_action = extra_info['random_action']
                next_action_mask = env.legal_actions_mask(next_state)
                mcts.move_to_child(next_state, action, next_random_action)

                state_action_probs = np.zeros(4)
                state_action_probs[action] = 1

                states.append(state)
                values.append(reward)
                action_probs.append(state_action_probs)
                action_masks.append(action_mask)

                state = next_state
                action_mask = next_action_mask

                env.render()

                if done:
                    break
            # Assign final game rewards
            winner_value = 1 if game_value > 0 else -1
            values = [winner_value] * len(values)

        # Convert training data into arrays
        states = np.array(states).reshape(-1, 4, 4, 1)
        action_probs = np.array(action_probs)
        values = np.array(values).reshape(-1, 1)
        action_masks = np.array(action_masks).reshape(-1, 4)

        # Train the model
        model.fit([states, action_masks], [values, action_probs], batch_size=batch_size, epochs=1)

    return model

def train(*, epochs, games_per_epoch, mcts_iterations, batch_size):
    # Create the 2048 environment, model, and MCTS
    env = Game2048Env()
    model = create_2048_policy_model()
    mcts = MCTS2048(model, mcts_iterations, env)

    # Train the model
    return train_2048(model, env, mcts, epochs, games_per_epoch, mcts_iterations, batch_size)

def main():
    # Hyperparameters
    epochs = 1000
    games_per_epoch = 10
    mcts_iterations = 10
    batch_size = 32

    return train(epochs=epochs, games_per_epoch=games_per_epoch, mcts_iterations=mcts_iterations, batch_size=batch_size)


if __name__ == '__main__':
    main()
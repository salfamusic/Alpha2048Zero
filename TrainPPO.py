import math
from pickle import SHORT_BINUNICODE
from GameStepSampler import GameStepSampler, GameStepSampler2, GameStepSampler3, GameStepSampler4
from PPOLongestGameExperienceBuffer import PPOLongestGameExperienceBuffer
from PPOShortestGameBuffer import PPOShortestGameExperienceBuffer
from PPOTrainingBuffer import PPOTrainingBuffer
import numpy as np
import tensorflow as tf
from tqdm.notebook import tqdm, trange
from IPython.display import clear_output

from Game2048Env import Game2048Env
from PPOAgent import PPOAgent
from PPOExperienceBuffer import PPOExperienceBuffer
from webapp2048.WebApp2 import call_add_step, call_add_training_step, call_complete_epoch, call_refresh


def do_epoch(env, epoch, model, sampler, batch_size):
    game_count = 0

    #max_step = 10_000 if epoch == 0 else sampler.required_episode_length + 10
    max_step = 100_000

    playing_pbar = tqdm(leave=False, total=sampler.average_episodes_in_latter_half(max_step))

    while not sampler.is_done_collecting_episode_steps(max_step):
        # state = mcts.reset_env()
        state = env.reset()

        step = 0

        while True:
            # action = mcts.search(mcts.root)
            mask = env.legal_actions_mask_from_board(env.board)
            action, value, logits, probs = model.select_action_value_and_unmasked_logits(state, mask)

            if not env.is_action_legal(env.board, action):
                print(f'{state=}')
                print(f'{env.board=}')
                print(f'{action=}')
                print(f'{mask=}')
                print(f'{probs=}')
                print(f'{logits=}')
                print("illegal")

            next_state, reward, done, extra_info = env.step(action)
            # next_random_action = extra_info['random_action']

            sampler.add_step(state, action, value, reward, next_state, logits, done)

            # mcts.move_to_child(next_state, action, next_random_action)
            call_add_step(epoch, game_count, action, env.board.tolist(), reward)

            state = next_state

            step += 1
            
            if done or step >= max_step:
                mask = env.legal_actions_mask_from_board(env.board)
                _, done_state_value, _ = model.select_action_value_and_logits(state, mask)
                sampler.end_episode(done_state_value)
                
                call_refresh()
                game_count += 1
                playing_pbar.reset(total=sampler.average_episodes_in_latter_half(max_step))
                break


    clear_output(wait=True)

    def buffer_to_training_buffer(buffer):
        all_states, all_y_true_policy, all_returns = model.make_training_data(buffer.states, buffer.actions, buffer.values, buffer.next_values, buffer.rewards, buffer.next_states, buffer.logits, buffer.dones)
        return PPOTrainingBuffer(all_states, all_y_true_policy, all_returns)
    print("building training buffers")
    sampler.build_training_buffers(buffer_to_training_buffer)
    training_buffer = sampler.sample_training_steps(max_step)
    total = training_buffer.get_number_of_batches(batch_size)
    passes = 4
    training_pbar = tqdm(leave=False, total=total*passes)

    train_batches(epoch, passes, batch_size, training_buffer, model, training_pbar)
    
    sampler.clear()
    model.save_models("ppo")
    call_complete_epoch(epoch)
    clear_output(wait=True)

def train_batches(epoch, passes, batch_size, training_buffer, model, training_pbar):
    for _ in range(passes):
        batch_num = 0
        random_indexer = training_buffer.get_random_indexer()
        for states, y_true_policy, returns in training_buffer.get_batches(batch_size, random_indexer):
            # Perform training on the batch
            ppo_loss, value_loss, lr, entropy_coefficient = model.train(states, y_true_policy, returns)
            training_pbar.update(1)
            training_pbar.set_description_str(f"{ppo_loss=:.5f} {value_loss=:.5f} {lr=:.5f} {entropy_coefficient=:.5f}")
            call_add_training_step(epoch, batch_num, ppo_loss, value_loss/1000, lr, entropy_coefficient)
            batch_num += 1

def train_2048_ppo(model, env, epochs, sampler, batch_size):
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

    for epoch in range(epochs):
        do_epoch(env, epoch, model, sampler, batch_size)

    return model

def train_ppo(*, epochs, window_size, batch_size, required_number_of_latter_half_episodes_per_step):
    # Create the 2048 environment, model, and MCTS
    env = Game2048Env(window_size=window_size)
    model = PPOAgent(env.observation_space.shape, env.action_space.n)
    sampler = GameStepSampler4(required_number_of_latter_half_episodes_per_step)

    # model.load_models('ppo')

    # Train the model
    return train_2048_ppo(model, env, epochs, sampler, batch_size)
# ... [Your imports and hyperparameters here] ...
import asyncio

import numpy as np
from tqdm.notebook import tqdm

from Game2048Env import Game2048Env
from PPOExperienceBuffer import PPOExperienceBuffer
from VectorizedGame2048Env import VectorizedGame2048Env
from VectorizedPPOAgent import VectorizedPPOAgent
from webapp2048.WebApp import call_add_step, call_add_steps, async_call_add_steps


async def vectorized_do_epoch(batch_size, env, epoch, model, game_counts, num_envs=10):
    buffers = [PPOExperienceBuffer() for _ in range(num_envs)]

    states = env.observation()

    # Initialize a tqdm progress bar
    # pbar = tqdm(total=batch_size, desc="Buffer Filling", position=0, leave=True)
    while np.sum([buffer.size() for buffer in buffers]) < batch_size:
        actions = model.vectorized_select_action(states, [Game2048Env.legal_actions_mask(state) for state in states])
        next_states, rewards, dones, _ = env.step(actions)

        # Store the experiences in the respective buffers
        for i, (buf, state, action, reward, next_state, done) in enumerate(
                zip(buffers, states, actions, rewards, next_states, dones)):
            logits = model.get_unmasked_logits(state)
            buf.store(state, action, reward, next_state, logits, done)
            # pbar.update(1)

            # Handle game resets
            if done:
                game_counts[i] = max(game_counts) + 1
                next_states[i] = env.reset_single_game(i)  # Assuming you have a reset_single_game method

        try:
            await async_call_add_steps(0, game_counts, actions, [Game2048Env.board_from_state(next_state).tolist() for next_state in next_states])
        except Exception as e:
            print(e)

        states = next_states

    # pbar.close()

    # Train the model using data from all buffers
    all_states = np.concatenate([buf.states for buf in buffers], axis=0)
    all_actions = np.concatenate([buf.actions for buf in buffers], axis=0)
    all_rewards = np.concatenate([buf.rewards for buf in buffers], axis=0)
    all_next_states = np.concatenate([buf.next_states for buf in buffers], axis=0)
    all_logits = np.concatenate([buf.logits for buf in buffers], axis=0)
    all_dones = np.concatenate([buf.dones for buf in buffers], axis=0)

    print(f"{np.mean(all_rewards)=}")

    model.vectorized_train(all_states, all_actions, all_rewards, all_next_states, all_logits, all_dones)
    for buf in buffers:
        buf.clear()


async def train_vectorized_2048_ppo(model, env, epochs, batch_size, num_envs=10):
    env.reset()
    game_counts = [i for i in range(num_envs)]
    for epoch in range(epochs):
        await vectorized_do_epoch(batch_size, env, epoch, model, game_counts, num_envs)

    return model


async def train_vectorized_ppo(*, epochs, batch_size, temperature, num_envs=10):
    env = VectorizedGame2048Env(num_envs=num_envs)
    model = VectorizedPPOAgent(env.observation_space.shape, env.action_space.n, temperature=temperature)

    return await train_vectorized_2048_ppo(model, env, epochs, batch_size, num_envs)

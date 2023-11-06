import numpy as np


class OptimizedGame2048Env:
    def __init__(self, num_envs):
        self.num_envs = num_envs
        self.boards = np.zeros((num_envs, 4, 4), dtype=np.uint16)
        self.reset()

    def reset(self, **kwargs):
        self.boards[:] = 0
        for _ in range(2):
            self.add_tile()
        return self.observation()

    def observation(self):
        return self.make_observation(self.boards, self.is_game_over(self.boards))

    def step(self, actions):
        prev_boards = self.boards.copy()
        for action in range(4):
            mask = actions == action
            if np.any(mask):
                self.boards[mask] = self.half_step(self.boards[mask], action)
        self.add_tile()
        dones = self.is_game_over(self.boards)
        rewards = self.compute_reward(self.boards, prev_boards)
        rewards[dones] = self.terminal_reward(self.boards[dones])
        return self.observation(), rewards, dones, {}

    @classmethod
    def is_game_over(cls, boards):
        return ~np.any(cls.legal_actions_mask_from_board(boards), axis=1)

    def add_tile(self):
        empty_cells = np.transpose(np.where(self.boards == 0))
        for i in range(self.num_envs):
            env_empty_cells = empty_cells[empty_cells[:, 0] == i]
            if len(env_empty_cells) > 0:
                chosen_cell = env_empty_cells[np.random.choice(len(env_empty_cells))]
                self.boards[chosen_cell[0], chosen_cell[1], chosen_cell[2]] = 2 if np.random.random() < 0.9 else 4

    @classmethod
    def legal_actions_mask(cls, states):
        return cls.legal_actions_mask(states)

    @classmethod
    def legal_actions_mask_from_board(cls, boards):
        return np.array([
            [cls.is_action_legal(board, action) for action in range(4)]
            for board in boards
        ], dtype=bool)

    @classmethod
    def board_from_state(cls, states):
        states_reshaped = np.array(states).reshape((-1, 5, 4, 4))
        return states_reshaped[:, 0, :, :]

    @classmethod
    def is_action_legal(cls, board, action):
        return not np.array_equal(board, cls.half_step(np.array([board]), action)[0])

    @classmethod
    def half_step(cls, boards, action):
        """Performs a step on the provided boards without adding a new tile."""
        new_boards = boards.copy()
        if action == 0:  # Up
            new_boards = cls.move_up(new_boards)
        elif action == 1:  # Down
            new_boards = cls.move_down(new_boards)
        elif action == 2:  # Left
            new_boards = cls.move_left(new_boards)
        elif action == 3:  # Right
            new_boards = cls.move_right(new_boards)
        return new_boards

    @classmethod
    def move_left(cls, boards):
        new_boards = np.zeros_like(boards)
        for env_idx, board in enumerate(boards):
            for i in range(4):
                tiles = [val for val in board[i] if val != 0]
                merged = []
                while tiles:
                    if len(tiles) >= 2 and tiles[0] == tiles[1]:
                        merged.append(tiles[0] * 2)
                        tiles = tiles[2:]
                    else:
                        merged.append(tiles.pop(0))
                new_boards[env_idx, i, :len(merged)] = merged
        return new_boards

    @classmethod
    def move_right(cls, boards):
        boards = boards.reshape((-1,4,4))
        return np.array(
            [np.fliplr(cls.move_left(np.array([np.fliplr(board)]).reshape((1, 4, 4)))) for board in boards]).squeeze()

    @classmethod
    def move_up(cls, boards):
        boards = boards.reshape((-1,4,4))
        return np.array([np.transpose(cls.move_left(np.array([np.transpose(board)]).reshape((1, 4, 4)))) for board in
                         boards]).squeeze()

    @classmethod
    def move_down(cls, boards):
        boards = boards.reshape((-1,4,4))
        return np.array([np.transpose(cls.move_right(np.array([np.transpose(board)]).reshape((1, 4, 4)))) for board in
                         boards]).squeeze()

    @classmethod
    def compute_static_reward(cls, boards):
        rewards = np.zeros(boards.shape[0])
        for env_idx, board in enumerate(boards):
            tiles_adj_values = cls.get_adjacent_tile_values(board)
            for value, adj_tiles in tiles_adj_values:
                rewards[env_idx] += cls.compute_tile_score(value, adj_tiles)
        return rewards

    @staticmethod
    def get_adjacent_tile_values(board):
        adjacent_values = []

        for i in range(4):
            for j in range(4):
                current_tile_value = board[i, j]

                up = board[i - 1, j] if i - 1 >= 0 else 0
                down = board[i + 1, j] if i + 1 < 4 else 0
                left = board[i, j - 1] if j - 1 >= 0 else 0
                right = board[i, j + 1] if j + 1 < 4 else 0

                adj_values = [up, down, left, right]
                adjacent_values.append((current_tile_value, adj_values))

        return adjacent_values

    @classmethod
    def compute_reward(cls, boards, prev_boards):
        return np.tanh(cls.compute_static_reward(boards) - cls.compute_static_reward(prev_boards))

    @classmethod
    def make_observation(cls, boards, dones):
        # Compute half_steps for all actions in a batched manner
        half_steps = np.array([cls.half_step(boards, action) for action in range(4)]).transpose((1, 0, 2, 3))

        # Stack the original boards with their half_step results
        observations = np.concatenate([boards[:, np.newaxis, :, :], half_steps], axis=1)

        # For environments that are done, replace their observations with just the board
        done_indices = np.where(dones)[0]
        observations[done_indices] = np.array([boards[done_indices]] * 5).transpose((1, 0, 2, 3))

        return observations





from collections import Counter

import gym
from gym import spaces
import numpy as np

MAX_TILE = 65536
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

class Game2048Env(gym.Env):
    def __init__(self, window_size=1):
        super(Game2048Env, self).__init__()

        self.window_size = window_size

        # The action space is discrete with 4 possible moves: Up, Down, Left, Right
        self.action_space = spaces.Discrete(4)

        # The observation space will be a 4x4 matrix with the board state
        self.observation_space = spaces.Box(low=0, high=MAX_TILE, shape=self.observation_shape(self.window_size), dtype=np.uint16)

        self.reset()

    def reset(self, **kwargs):
        # Reset the board to the initial state
        self.board = np.zeros((4, 4), dtype=np.uint16)
        self.window = np.zeros((self.window_size, 4, 4), dtype=np.uint16)
        self.add_tile()
        self.add_tile()

        observation = self.make_observation(self.board, self.window)
        return observation

    @classmethod
    def observation_shape(cls, window_size):
        return cls.make_observation(np.zeros((4, 4), dtype=np.uint16), np.zeros((window_size, 4, 4), dtype=np.uint16)).shape

    @classmethod
    def legal_actions_mask_from_board(cls, board):
        return np.array([
            cls.is_action_legal(board, 0),
            cls.is_action_legal(board, 1),
            cls.is_action_legal(board, 2),
            cls.is_action_legal(board, 3),
        ], dtype=bool)

    @classmethod
    def is_action_legal(cls, board, action):
        return not (board == cls.half_step(board, action)).all()

    @classmethod
    def half_step(cls, board, action):
        """Performs a step on the provided board without adding a new tile, and without affecting the environment's state."""
        new_board = board.copy()
        if action == UP:  # Up
            new_board, _ = cls.move_up(new_board)
        elif action == DOWN:  # Down
            new_board, _ = cls.move_down(new_board)
        elif action == LEFT:  # Left
            new_board, _ = cls.move_left(new_board)
        elif action == RIGHT:  # Right
            new_board, _ = cls.move_right(new_board)

        return new_board

    @classmethod
    def merge_score(cls, board, action):
        """Performs a step on the provided board without adding a new tile, and without affecting the environment's state."""
        new_board, merge_score = board.copy(), 0
        if action == UP:  # Up
            _, merge_score = cls.move_up(new_board)
        elif action == DOWN:  # Down
            _, merge_score = cls.move_down(new_board)
        elif action == LEFT:  # Left
            _, merge_score = cls.move_left(new_board)
        elif action == RIGHT:  # Right
            _, merge_score = cls.move_right(new_board)

        return merge_score

    @classmethod
    def terminal_reward(cls, board):
        return -1
    
    def step(self, action):
        # Get the resulting board state after taking an action
        i = 0
        while not self.is_action_legal(self.board, action):
            action = (action + 1) % 4
            i += 1
            assert i < 5, "infinite loop"

        prev_board = self.board.copy()
        merge_score = self.merge_score(self.board, action)
        self.board = self.half_step(self.board, action)
        self.window = np.array(self.window[1:].tolist() + [prev_board.tolist()])
        # Add a new tile if the board changed

        random_action = self.add_tile()

        done = self.is_game_over(self.board)
        reward = self.compute_reward(self.board, prev_board, merge_score)


        if done:
            reward = self.terminal_reward(self.board)
        observation = self.make_observation(self.board, self.window)

        return observation, reward, done, {
            "random_action": random_action
        }

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    @classmethod
    def make_observation(cls, board, window):
        current_board = board.copy()
        up_board = cls.half_step(board, UP)
        down_board = cls.half_step(board, DOWN)
        left_board = cls.half_step(board, LEFT)
        right_board = cls.half_step(board, RIGHT)

        board_transpositions = cls.make_transpositions(current_board)
        up_transpositions = cls.make_transpositions(up_board)
        down_transpositions = cls.make_transpositions(down_board)
        left_transpositions = cls.make_transpositions(left_board)
        right_transpositions = cls.make_transpositions(right_board)

        windows_transpositions = [cls.make_transpositions(prev_board) for prev_board in window.tolist()]

        segmented_state = [board_transpositions, up_transpositions, down_transpositions, left_transpositions, right_transpositions, *windows_transpositions]
        state = np.concatenate(segmented_state)

        observation = np.array(state, dtype=np.uint32)
        return observation.reshape(len(segmented_state),4,4,board_transpositions.shape[-1])

    

    @classmethod
    def compute_reward(cls, board, prev_board, merge_score):
        return np.tanh((cls.compute_static_reward(board) - cls.compute_static_reward(prev_board)) / 100) + np.log2(merge_score + 1e-7)


    @classmethod
    def compute_static_reward(cls, board):
        tiles_adj_values = cls.get_adjacent_tile_values(board)
        reward = 0

        for value, adj_tiles in tiles_adj_values:
            reward += cls.compute_tile_score(value, adj_tiles)

        return reward / cls.number_of_non_zero_tiles(board)

    @classmethod
    def number_of_non_zero_tiles(cls, board):
        return len(board[board > 0])
    
    @staticmethod
    def compute_adj_score(tile, adj_tile):
        if adj_tile > tile:
            return 0.0

        return adj_tile / tile

    @staticmethod
    def tile_formula(tile, adj_score, tile_exp=2.0):
        bias = np.log2(tile)
        return (bias + adj_score) * tile ** tile_exp

    @classmethod
    def compute_tile_adj_score(cls, tile, adj_tiles):
        adj_score = np.max([cls.compute_adj_score(tile, adj_tile) for adj_tile in adj_tiles])
        adj_score /= 2

        return adj_score

    @classmethod
    def compute_tile_score(cls, tile, adj_tiles, tile_exp=2.0):
        if tile == 0:
            return 0.0
        adj_score = cls.compute_tile_adj_score(tile, adj_tiles)

        return cls.tile_formula(tile, adj_score, tile_exp)

    @staticmethod
    def get_adjacent_tile_values(board):
        # Check if the board is 4x4
        if board.shape != (4, 4):
            raise ValueError("The provided board is not a 4x4 matrix.")

        adjacent_values = []

        for i in range(4):
            for j in range(4):
                current_tile_value = board[i, j]

                # Determine adjacent tiles
                up = board[i - 1, j] if i - 1 >= 0 else 0
                down = board[i + 1, j] if i + 1 < 4 else 0
                left = board[i, j - 1] if j - 1 >= 0 else 0
                right = board[i, j + 1] if j + 1 < 4 else 0

                adj_values = [up, down, left, right]
                adjacent_values.append((current_tile_value, adj_values))

        return adjacent_values


    @staticmethod
    def move_left(board):
        new_board = np.zeros_like(board)
        merge_score = 0
        for i in range(4):
            # Pull non-zero tiles
            tiles = [val for val in board[i] if val != 0]
            merged = []
            while tiles:
                if len(tiles) >= 2 and tiles[0] == tiles[1]:
                    merged.append(tiles[0] * 2)
                    tiles = tiles[2:]
                else:
                    merged.append(tiles.pop(0))
            merge_score += sum(merged)
            new_board[i, :len(merged)] = merged
        return new_board, merge_score

    @classmethod
    def move_right(cls, board):
        new_board, merge_score = cls.move_left(np.fliplr(board))
        return np.fliplr(new_board), merge_score

    @classmethod
    def move_up(cls, board):
        new_board, merge_score = cls.move_left(np.transpose(board))
        return np.transpose(new_board), merge_score

    @classmethod
    def move_down(cls, board):
        new_board, merge_score = cls.move_right(np.transpose(board))
        return np.transpose(new_board), merge_score

    def add_tile(self):
        empty_cells = [(i, j) for i in range(4) for j in range(4) if self.board[i, j] == 0]
        if empty_cells:
            i, j = empty_cells[np.random.choice(len(empty_cells))]
            self.board[i, j] = 2 if np.random.random() < 0.9 else 4
            return i, j, self.board[i, j]
        return None, None, None

    @classmethod
    def is_game_over(cls, board):
        # If there are any empty cells, the game is not over
        return not np.any(cls.legal_actions_mask_from_board(board))

    def save_state(self):
        """Save the current game state."""
        return self.board.copy()

    def load_state(self, saved_state):
        """Load a saved game state."""
        self.board = saved_state.copy()
    
    @classmethod
    def make_transpositions(cls, board):
        max_board_tile = np.max(board)
        transpositions = []

        downward_board = np.array(board.copy())
        downward_max_board_tile = max_board_tile

        updward_board = np.array(board.copy())
        updward_max_board_tile = max(max_board_tile, 2)

        while downward_max_board_tile > 2:
            downward_board = cls.transpose_values_down(downward_board)
            transpositions.insert(0, downward_board)
            downward_max_board_tile /= 2

        transpositions.append(board.copy())

        while updward_max_board_tile < MAX_TILE:
            updward_board = cls.transpose_values_up(updward_board)
            transpositions.append(updward_board)
            updward_max_board_tile *= 2
        
        return np.array(transpositions, dtype=np.uint32).T.reshape((4,4, len(transpositions)))



    @staticmethod
    def transpose_values_up(board):
        board *= 2
        return board

    @staticmethod
    def transpose_values_down(board):
        divided_board = board / 2
        divided_board[divided_board == 1] = 0
        return divided_board


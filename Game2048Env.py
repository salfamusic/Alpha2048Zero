import gym
from gym import spaces
import numpy as np

class Game2048Env(gym.Env):
    def __init__(self, enable_pygame=False):
        super(Game2048Env, self).__init__()

        # The action space is discrete with 4 possible moves: Up, Down, Left, Right
        self.action_space = spaces.Discrete(4)

        # The observation space will be a 4x4 matrix with the board state
        self.observation_space = spaces.Box(low=0, high=2048, shape=(4, 4), dtype=np.uint16)

        self.reset()

    def reset(self, **kwargs):
        # Reset the board to the initial state
        self.board = np.zeros((4, 4), dtype=np.uint16)
        self.add_tile()
        self.add_tile()
        return self.board

    def legal_actions_mask(self, board):
        return np.array([
            self.is_action_legal(board, 0),
            self.is_action_legal(board, 1),
            self.is_action_legal(board, 2),
            self.is_action_legal(board, 3),
        ], dtype=bool)

    def is_action_legal(self, board, action):
        return not (board == self.half_step(board, action)).all()

    def half_step(self, board, action):
        """Performs a step on the provided board without adding a new tile, and without affecting the environment's state."""
        new_board = board.copy()
        if action == 0:  # Up
            new_board = self.move_up(new_board)
        elif action == 1:  # Down
            new_board = self.move_down(new_board)
        elif action == 2:  # Left
            new_board = self.move_left(new_board)
        elif action == 3:  # Right
            new_board = self.move_right(new_board)

        return new_board

    def step(self, action):
        # Get the resulting board state after taking an action
        prev_board = self.board.copy()
        self.board = self.half_step(self.board, action)
        # Add a new tile if the board changed
        assert not np.array_equal(prev_board, self.board), "Illegal action"
        random_action = self.add_tile()

        done = self.is_game_over()
        previous_reward = self.compute_reward(prev_board)
        current_reward = self.compute_reward(self.board)
        reward = max(current_reward - previous_reward, 0.0)

        if done:
            reward = -(1/current_reward)

        return self.board, reward, done, {
            "random_action": random_action
        }

    def render(self, mode='human'):
        pass

    def close(self):
        pass
    def compute_reward(self, board):
        non_zero_tiles = board[board > 0]
        if non_zero_tiles.size == 0:
            return 0
        return np.mean(non_zero_tiles)

    # Additional helper methods for game logic
    def move_left(self, board):
        new_board = np.zeros_like(board)
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
            new_board[i, :len(merged)] = merged
        return new_board

    def move_right(self, board):
        return np.fliplr(self.move_left(np.fliplr(board)))

    def move_up(self, board):
        return np.transpose(self.move_left(np.transpose(board)))

    def move_down(self, board):
        return np.transpose(self.move_right(np.transpose(board)))

    def add_tile(self):
        empty_cells = [(i, j) for i in range(4) for j in range(4) if self.board[i, j] == 0]
        if empty_cells:
            i, j = empty_cells[np.random.choice(len(empty_cells))]
            self.board[i, j] = 2 if np.random.random() < 0.9 else 4
            return i, j, self.board[i, j]
        return None, None, None

    def is_game_over(self):
        # If there are any empty cells, the game is not over
        if np.any(self.board == 0):
            return False

        # Check for possible moves in all directions
        for move in [self.move_left, self.move_right, self.move_up, self.move_down]:
            if not np.array_equal(self.board, move(self.board)):
                return False

        # If no moves are possible, then the game is over
        return True

    def save_state(self):
        """Save the current game state."""
        return self.board.copy()

    def load_state(self, saved_state):
        """Load a saved game state."""
        self.board = saved_state.copy()


import numpy as np

from Node import Node


class MCTS2048:
    def __init__(self, model, iterations, env):
        self.model = model
        self.iterations = iterations
        self.env = env
        initial_state = env.board  # Assuming you have this method
        self.root = Node(initial_state)

    def reset_env(self):
        board = self.env.reset()
        self.root = Node(board)

        return board

    def search(self, root):
        self.root = root
        for _ in range(self.iterations):
            node = root
            search_path = [node]

            # Save the initial state of the environment
            saved_state = self.env.save_state()

            # Selection
            while not node.is_leaf():
                action, node = self.select_child(node)
                search_path.append(node)

            # Expansion
            value = self.expand_node(node).value  # Getting the value prediction directly from the expansion

            # Backpropagation
            self.backpropagate(search_path, value)

            # Restore the initial state of the environment
            self.env.load_state(saved_state)

        return self.best_action(root)

    def select_child(self, node):
        """Select the child with the highest UCB value."""
        ucb_values = [
            child.value / (child.visits + 1e-10) + c * np.sqrt(node.visits) * child.priors[child.action] / (
                        1 + child.visits)
            for child in node.children
        ]
        action = np.argmax(ucb_values)
        found_child = node.children[action]
        return found_child.action, found_child

    def expand_node(self, node):
        """Expand the current node using the neural network."""
        board = node.state
        board_input = board.reshape(1, 4, 4, 1)
        mask_input = self.env.legal_actions_mask(board).reshape(1, 4)
        value, policy = self.model.predict([board_input, mask_input], verbose=0)
        value = value[0]
        policy = policy[0]

        for action in range(4):
            if not self.env.is_action_legal(board, action):
                continue
            next_state = self.take_half_step_action(board, action)

            child = Node(next_state, parent=node)
            child.value = value  # Setting the child's value to the neural network's value prediction
            child.action = action
            child.detailed_action = action
            child.priors = policy
            self.expand_random_node(child)  # Get the priors from expand_random_node
            node.children.append(child)

        return node

    def expand_random_node(self, node):
        """Expand the current node using the neural network."""
        board = node.state
        empty_spots = self.find_empty_spots(board)
        log_probs = []

        for spot in empty_spots:
            for tile_value, probability in [(2, 0.9), (4, 0.1)]:
                child_state = self.place_tile(board, spot, tile_value)
                child = Node(child_state, parent=node)
                child.value = 0.0
                child.detailed_action = tuple([spot[0], spot[1], tile_value])
                node.children.append(child)
                log_probs.append(np.log(probability))
        priors = self.softmax(np.array(log_probs))
        for action, child in enumerate(node.children):
            child.priors = priors
            child.action = action

    def softmax(self, x):
        """Compute the softmax of vector x."""
        exp_x = np.exp(x - np.max(x))  # subtracting max for numerical stability
        return exp_x / exp_x.sum()

    def take_half_step_action(self, board, action):
        next_state = self.env.half_step(board, action)
        return next_state

    def find_empty_spots(self, board):
        return [(i, j) for i in range(4) for j in range(4) if board[i, j] == 0]

    def place_tile(self, board, spot, tile_value):
        board_copy = board.copy()
        i, j = spot
        board_copy[i, j] = tile_value
        return board_copy

    def backpropagate(self, search_path, value):
        """Backpropagate the value up to the root."""
        for node in search_path:
            node.visits += 1
            node.value += value

    # def best_action(self, root):
    #     """Return the action of the most visited child of the root."""
    #     visits = [child.visits for child in root.children]
    #     return np.argmax(visits)

    def best_action(self, root):
        """Return the action of the best child of the root."""
        values = [child.value / (child.visits + 1e-10) for child in root.children]
        visits = [child.visits for child in root.children]

        # Considering both value and visits for selecting the best action
        best_value_index = np.argmax(values)
        best_visits_index = np.argmax(visits)

        # For simplicity, we can give preference to the node with the most visits if there's a tie
        # but you can introduce other criteria based on the specific needs of your application.
        return root.children[
            best_visits_index if visits[best_visits_index] >= visits[best_value_index] else best_value_index].action

    def move_to_child(self, new_state, move_action, random_action):
        """Move to the child node corresponding to the chosen action and new_state."""
        for action_child in self.root.children:
            if action_child.detailed_action == move_action:
                for random_child in action_child.children:
                    if random_child.detailed_action == random_action:
                        self.root = random_child
                        self.root.parent = None
                        assert np.array_equal(self.root.state, new_state), "Moving to child with incorrect state"
                        return

        # If the resulting state's child is not found, create a new root with the new_state
        self.root = Node(new_state)

# Constants
c = 1.0  # exploration constant
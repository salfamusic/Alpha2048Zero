import numpy as np


class Node:
    def __init__(self, state, parent=None):
        self.state = state  # board state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.action = -1
        self.detailed_action = None
        self.value = 0.0
        self.priors = np.zeros(4)  # prior probabilities for 4 actions

    def is_leaf(self):
        return len(self.children) == 0

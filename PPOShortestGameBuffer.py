import math
from PPOExperienceBuffer import PPOExperienceBuffer
import numpy as np
import random


class PPOShortestGameExperienceBuffer:
    def __init__(self):
        self.full_experience_buffer = PPOExperienceBuffer()        
        self.shortest_game_experience_buffer = PPOExperienceBuffer()
        self.current_game_experience_buffer = PPOExperienceBuffer()

    def store(self, state, action, value, reward, next_state, logits, done):
        self.full_experience_buffer.store(state, action, value, reward, next_state, logits, done)
        self.current_game_experience_buffer.store(state, action, value, reward, next_state, logits, done)

    def clear(self):
        self.full_experience_buffer = PPOExperienceBuffer()        
        self.shortest_game_experience_buffer = PPOExperienceBuffer()
        self.current_game_experience_buffer = PPOExperienceBuffer()

    def size(self):
        return self.full_experience_buffer.size()
    
    def shortest_size(self):
        return self.shortest_game_experience_buffer.size()

    def finalize_episode_with_next_values(self, next_value):
        self.full_experience_buffer.finalize_episode_with_next_values(next_value)
        self.current_game_experience_buffer.finalize_episode_with_next_values(next_value)
        if self.current_game_experience_buffer.size() < self.shortest_game_experience_buffer.size() or self.shortest_game_experience_buffer.size() == 0:
            self.shortest_game_experience_buffer = self.current_game_experience_buffer.copy()

        self.current_game_experience_buffer.clear()

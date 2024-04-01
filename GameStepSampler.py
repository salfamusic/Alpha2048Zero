import math
from random import choice
from PPOExperienceBuffer import PPOExperienceBuffer
from PPOTrainingBuffer import PPOTrainingBuffer


class GameStepSampler:
    def __init__(self, samples_per_episode, required_average_samples_per_episode):
        self.samples_per_episode = samples_per_episode
        self.required_average_samples_per_episode = required_average_samples_per_episode
        self.episodes = [PPOExperienceBuffer()]
        self.training_buffers = []

    def clear(self):
        self.episodes = [PPOExperienceBuffer()]
        self.training_buffers = []

    def add_step(self, *args):
        self.episodes[-1].store(*args)

    def end_episode(self, *args):
        self.episodes[-1].finalize_episode_with_next_values(*args)
        self.episodes.append(PPOExperienceBuffer())
    
    def sample_training_steps(self):
        sample_buffer = PPOTrainingBuffer([],[],[])
        sorted_training_buffers = sorted(self.training_buffers, key=lambda b: b.size())

        for training_buffer in sorted_training_buffers:
            steps_to_take = min(training_buffer.size() - sample_buffer.size(), self.samples_per_episode)
            if steps_to_take < 1:
                continue
            start_position = sample_buffer.size()
            end_position = start_position + steps_to_take
            sample_buffer.add_slice(*training_buffer.slice(start_position,end_position))

        sample_buffer.add_slice(*sorted_training_buffers[-1].slice(sample_buffer.size(), None))

        # sample_buffer.add_slice(*sorted_training_buffers[0].slice(None, None))
        # sample_buffer.add_slice(*sorted_training_buffers[-1].slice(None, None))

        return sample_buffer

    def average_samples_per_episode(self):
        sorted_episodes = sorted(self.episodes, key=lambda b: b.size())
        sampled_steps = 0
        steps_sampled_per_episode = []

        if len(self.episodes) == 1 and self.episodes[0].size() == 0:
            return 0

        for idx, episode in enumerate(sorted_episodes):
            steps_to_take = min(episode.size() - sampled_steps, self.samples_per_episode)
            if steps_to_take < 1:
                continue
            sampled_steps += steps_to_take
            steps_sampled_per_episode.append(max(steps_to_take, self.samples_per_episode))

        remaining_steps = sorted_episodes[-1].size() - sampled_steps
        steps_sampled_per_episode.append(remaining_steps)

        if not steps_sampled_per_episode:
            return 0
        return sum(steps_sampled_per_episode) / len(steps_sampled_per_episode)

    def is_done_collecting_episode_steps(self):
        average = self.average_samples_per_episode()

        return average > 0 and average <= self.required_average_samples_per_episode
    
    def build_training_buffers(self, convert_fn):
        episode_training_buffers = []
        for episode_buffer in self.episodes:
            if episode_buffer.size() > 0:
                episode_training_buffers.append(convert_fn(episode_buffer))
        self.training_buffers = episode_training_buffers


class GameStepSampler2:
    def __init__(self, batch_size, required_number_of_latter_half_episodes_per_step):
        self.batch_size = batch_size
        self.required_number_of_latter_half_episodes_per_step = required_number_of_latter_half_episodes_per_step
        self.episodes = [PPOExperienceBuffer()]
        self.training_buffers = []

    def clear(self):
        self.episodes = [PPOExperienceBuffer()]
        self.training_buffers = []

    def add_step(self, *args):
        self.episodes[-1].store(*args)

    def end_episode(self, *args):
        self.episodes[-1].finalize_episode_with_next_values(*args)
        self.episodes.append(PPOExperienceBuffer())
    
    def is_done_collecting_episode_steps(self, max_step):
        average = self.average_episodes_in_latter_half(max_step)

        return average > 0 and average >= self.required_number_of_latter_half_episodes_per_step and self.total_size() > self.batch_size
    
    def total_size(self):
        return sum([buffer.size() for buffer in self.episodes])

    def build_training_buffers(self, convert_fn):
        episode_training_buffers = []
        for episode_buffer in self.episodes:
            if episode_buffer.size() > 0:
                episode_training_buffers.append(convert_fn(episode_buffer))
        self.training_buffers = episode_training_buffers

    def average_episodes_in_latter_half(self, max_step):
        if not self.episodes:
            return 0

        # Calculate the average episode length
        max_length = min(max_step, max(buffer.size() for buffer in self.episodes))
        latter_half_start = max_length / 2

        # Total steps and episode counts for each step in the latter half
        total_steps = 0
        total_episodes_per_step = 0

        # Find the maximum size among episodes
        max_size = max_length

        # Iterate through each step in the latter half
        for step in range(int(latter_half_start), max_size):
            # Count how many episodes include this step
            episodes_with_step = sum(1 for buffer in self.episodes if buffer.size() > step)
            total_episodes_per_step += episodes_with_step
            total_steps += 1

        # Calculate and return the average number of episodes per step in the latter half
        return total_episodes_per_step / total_steps if total_steps > 0 else 0
        

    def sample_training_steps(self, max_step):
        sample_buffer = PPOTrainingBuffer([],[],[])
        amount = self.batch_size
        # Collect the sizes of all episodes
        episode_sizes = [buffer.size() for buffer in self.training_buffers]
        if not episode_sizes:
            return []

        # Find the maximum size among episodes
        max_size = min(max_step, max(episode_sizes))

        # Calculate the step size needed to collect the desired number of samples
        step_size = max(1, math.ceil(max_size / amount))

        # Iterate to collect samples
        current_step = max_size - 1
        while sample_buffer.size() < amount:
            # Collect all buffers that have the current step
            available_buffers = [buffer for buffer in self.training_buffers if buffer.size() > current_step]

            # Randomly select a buffer and collect the step
            if available_buffers:
                selected_buffer = choice(available_buffers)
                sample_buffer.add_step(*selected_buffer.get_step(current_step))

            # Move to the next step, loop back if necessary
            current_step -= step_size
            if current_step < 0:
                current_step = max_size - 1

        return sample_buffer
    
class GameStepSampler3:
    def __init__(self, required_number_of_latter_half_episodes_per_step):
        self.required_number_of_latter_half_episodes_per_step = required_number_of_latter_half_episodes_per_step
        self.episodes = [PPOExperienceBuffer()]
        self.training_buffers = []

    def clear(self):
        self.episodes = [PPOExperienceBuffer()]
        self.training_buffers = []

    def add_step(self, *args):
        self.episodes[-1].store(*args)

    def end_episode(self, *args):
        self.episodes[-1].finalize_episode_with_next_values(*args)
        self.episodes.append(PPOExperienceBuffer())
    
    def is_done_collecting_episode_steps(self, max_step):
        average = self.average_episodes_in_latter_half(max_step)

        return average > 0 and average >= self.required_number_of_latter_half_episodes_per_step
    
    def total_size(self):
        return sum([buffer.size() for buffer in self.episodes])

    def build_training_buffers(self, convert_fn):
        episode_training_buffers = []
        for episode_buffer in self.episodes:
            if episode_buffer.size() > 0:
                episode_training_buffers.append(convert_fn(episode_buffer))
        self.training_buffers = episode_training_buffers

    def average_episodes_in_latter_half(self, max_step):
        if not self.episodes:
            return 0

        # Calculate the average episode length
        max_length = min(max_step, max(buffer.size() for buffer in self.episodes))
        latter_half_start = max_length / 2

        # Total steps and episode counts for each step in the latter half
        total_steps = 0
        total_episodes_per_step = 0

        # Find the maximum size among episodes
        max_size = max_length

        # Iterate through each step in the latter half
        for step in range(int(latter_half_start), max_size):
            # Count how many episodes include this step
            episodes_with_step = sum(1 for buffer in self.episodes if buffer.size() > step)
            total_episodes_per_step += episodes_with_step
            total_steps += 1

        # Calculate and return the average number of episodes per step in the latter half
        return total_episodes_per_step / total_steps if total_steps > 0 else 0
        

    def sample_training_steps(self, max_step):
        sample_buffer = PPOTrainingBuffer([],[],[])
        print("sampling training buffer")
        for training_buffer in self.training_buffers:
            sample_buffer.add_from_buffer(training_buffer)

        return sample_buffer
    
class GameStepSampler4:
    def __init__(self, required_number_of_latter_half_episodes_per_step):
        self.required_number_of_latter_half_episodes_per_step = required_number_of_latter_half_episodes_per_step
        self.required_episode_length = -1
        self.episodes = [PPOExperienceBuffer()]
        self.training_buffers = []

    def clear(self):
        self.episodes = [PPOExperienceBuffer()]
        self.training_buffers = []

    def add_step(self, *args):
        self.episodes[-1].store(*args)

    def end_episode(self, *args):
        self.episodes[-1].finalize_episode_with_next_values(*args)
        episode_len = self.episodes[-1].size()

        if episode_len > self.required_episode_length:
            self.required_episode_length = episode_len
        self.episodes.append(PPOExperienceBuffer())
    
    def is_done_collecting_episode_steps(self, max_step):
        average = self.average_episodes_in_latter_half(max_step)

        return average > 0 and average >= self.required_number_of_latter_half_episodes_per_step
    
    def total_size(self):
        return sum([buffer.size() for buffer in self.episodes])

    def build_training_buffers(self, convert_fn):
        episode_training_buffers = []
        for episode_buffer in self.episodes:
            if episode_buffer.size() > 0:
                episode_training_buffers.append(convert_fn(episode_buffer))
        self.training_buffers = episode_training_buffers

    def average_episodes_in_latter_half(self, max_step):
        if not self.episodes or self.required_episode_length < 0:
            return 0

        # Calculate the average episode length
        max_length = self.required_episode_length
        latter_half_start = max_length / 2

        # Total steps and episode counts for each step in the latter half
        total_steps = 0
        total_episodes_per_step = 0

        # Find the maximum size among episodes
        max_size = max_length

        # Iterate through each step in the latter half
        for step in range(int(latter_half_start), max_size):
            # Count how many episodes include this step
            episodes_with_step = sum(1 for buffer in self.episodes if buffer.size() > step)
            total_episodes_per_step += episodes_with_step
            total_steps += 1

        # Calculate and return the average number of episodes per step in the latter half
        return total_episodes_per_step / total_steps if total_steps > 0 else 0
        

    def sample_training_steps(self, max_step):
        sample_buffer = PPOTrainingBuffer([],[],[])
        print("sampling training buffer")
        for training_buffer in self.training_buffers:
            sample_buffer.add_from_buffer(training_buffer)

        return sample_buffer

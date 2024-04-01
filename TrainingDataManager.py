from dataclasses import dataclass
import sys
from typing import Optional
from collections import defaultdict

Epoch = int
Game = int
Step = int
Batch = int
Board = list

@dataclass
class GameStep:
    epoch: Epoch
    game: Game
    action: int
    next_board: Board
    reward: float

@dataclass
class TrainingStep:
    epoch: Epoch
    batch: Batch
    ppo_loss: float
    value_loss: float
    lr: float
    entropy_coefficient: float

@dataclass
class GameData:
    epoch: Epoch
    game: Game
    steps: list[Board]


@dataclass
class ChartData:
    name: Epoch = 0
    average_game_steps: float = 0.0
    max_game_steps: int = 0
    min_game_steps: int = sys.maxsize
    average_game_reward: float = 0.0
    max_game_reward: float = float('-inf')
    min_game_reward: float = float('inf')
    average_batch_ppo_loss: float = 0.0
    average_batch_value_loss: float = 0.0
    average_batch_lr: float = 0.0
    average_batch_entropy_coefficient: float = 0.0



@dataclass
class LatestStep:
    epoch: Epoch
    game: Game
    step: Step
    board: Board

@dataclass
class GameStats:
    total_steps: int = 0
    total_reward: float = 0.0
    max_steps: int = 0
    min_steps: int = sys.maxsize
    max_reward: float = float('-inf')
    min_reward: float = float('inf')

@dataclass
class TrainingStats:
    total_ppo_loss: float = 0.0
    total_value_loss: float = 0.0
    total_lr: float = 0.0
    total_entropy_coefficient: float = 0.0
    batches: int = 0

class TrainingDataManager:
    def __init__(self):
        self.game_steps = defaultdict(lambda: defaultdict(list))
        self.training_steps = defaultdict(list)
        self.latest_game_step = None
        self.latest_training_step = None
        self.best_game_data = None
        self.game_stats = defaultdict(lambda: defaultdict(GameStats))
        self.training_stats = defaultdict(TrainingStats)
        self.chart_data_data = defaultdict(ChartData)

    @property
    def best_game(self) -> Optional[GameData]:
        return self.best_game_data
    
    @property
    def latest_step(self) -> Optional[LatestStep]:
        return self.latest_game_step

    @property
    def chart_data(self) -> list[ChartData]:
        return list(self.chart_data_data.values())

    def add_step(self, step: GameStep):
        # Update game steps
        self.game_steps[step.epoch][step.game].append(step)
        self.latest_game_step = LatestStep(epoch=step.epoch, game=step.game, step=len(self.game_steps[step.epoch][step.game]), board=step.next_board)

        # Update game stats
        stats = self.game_stats[step.epoch][step.game]
        stats.total_steps += 1
        stats.total_reward += step.reward
        stats.max_steps = max(stats.max_steps, stats.total_steps)
        stats.min_steps = min(stats.min_steps, stats.total_steps)
        stats.max_reward = max(stats.max_reward, step.reward)
        stats.min_reward = min(stats.min_reward, step.reward)

        # Check for best game
        if self.best_game_data is None or stats.total_steps > len(self.best_game_data.steps):
            self.best_game_data = GameData(epoch=step.epoch, game=step.game, steps=[s.next_board for s in self.game_steps[step.epoch][step.game]])

        # Update chart data
        self.update_chart_data(step.epoch)

    def add_training_step(self, step: TrainingStep):
        # Update training steps
        self.training_steps[step.epoch].append(step)
        self.latest_training_step = step

        # Update training stats
        stats = self.training_stats[step.epoch]
        stats.total_ppo_loss += step.ppo_loss
        stats.total_value_loss += step.value_loss
        stats.total_lr += step.lr
        stats.total_entropy_coefficient += step.entropy_coefficient
        stats.batches += 1

        # Update chart data
        self.update_chart_data(step.epoch)

    def update_chart_data(self, epoch):
        game_stats = self.game_stats[epoch]
        training_stats = self.training_stats[epoch]

        total_games = len(game_stats)
        if total_games > 0:
            game_steps, game_rewards = zip(*[(stats.total_steps, stats.total_reward) for stats in game_stats.values()])
            self.chart_data_data[epoch] = ChartData(
                name=epoch,
                average_game_steps=sum(game_steps) / total_games,
                max_game_steps=max(game_steps),
                min_game_steps=min(game_steps),
                average_game_reward=sum(game_rewards) / sum(game_steps),
                max_game_reward=max(game_rewards),
                min_game_reward=min(game_rewards),
                average_batch_ppo_loss=training_stats.total_ppo_loss / training_stats.batches if training_stats.batches > 0 else 0,
                average_batch_value_loss=training_stats.total_value_loss / training_stats.batches if training_stats.batches > 0 else 0,
                average_batch_lr=training_stats.total_lr / training_stats.batches if training_stats.batches > 0 else 0,
                average_batch_entropy_coefficient=training_stats.total_entropy_coefficient / training_stats.batches if training_stats.batches > 0 else 0,
            )

    def complete_epoch(self, epoch: Epoch):
        """Mark an epoch as completed and clear its raw data."""
        if epoch in self.game_steps:
            del self.game_steps[epoch]

        if epoch in self.training_steps:
            del self.training_steps[epoch]
# agents/random_agent.py

from agents.abstract_agent import AbstractAgent
import gymnasium as gym


class RandomAgent(AbstractAgent):
    def __init__(self, state_space: gym.Space, action_space: gym.Space, discount_factor=0.9):
        super().__init__(state_space, action_space, discount_factor)
        self.name = "RandomAgent"

    def add_trajectory(self, trajectory):
        """
        RandomAgent does not utilize trajectories for learning.
        """
        pass

    def update(self, terminated: bool):
        """
        RandomAgent does not learn; no update is performed.
        """
        pass

    def policy(self, state):
        """
        Selects a random action.
        """
        return self.action_space.sample()

    def save(self, file_path: str = './') -> None:
        """
        RandomAgent does not have a model to save.
        """
        pass

    def load(self, file_path: str = './') -> None:
        """
        RandomAgent does not have a model to load.
        """
        pass

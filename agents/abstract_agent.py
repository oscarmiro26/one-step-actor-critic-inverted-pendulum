# agents/abstract_agent.py

from abc import ABC, abstractmethod
import gymnasium as gym
from torchrl.data import ReplayBuffer, LazyTensorStorage

from util.device import fetch_device


class AbstractAgent(ABC):
    """
    Agent abstract base class.
    """

    def __init__(self, state_space: gym.Space, action_space: gym.Space, discount_factor=0.9):
        """
        Agent Base Class constructor.

        :param state_space: State space of the gym environment.
        :param action_space: Action space of the gym environment.
        :param discount_factor: Discount factor (`gamma`).
        """
        self.state_space = state_space
        self.action_space = action_space
        self.discount_factor = discount_factor

        # By default, One-Step Actor Critic does not use a replay buffer. In this case, the replay buffer is simply used
        # to store the latest trajectory. Algorithms like DQN do use > 1 size replay buffers.
        self._replay_buffer = ReplayBuffer(
            storage=LazyTensorStorage(
                max_size=1,
                device=fetch_device()
            )
        )

    @abstractmethod
    def add_trajectory(self, trajectory):
        """
        Abstract method to add a trajectory to the agent's replay buffer.

        :param trajectory: Trajectory to add to the replay buffer.
        """
        pass

    @abstractmethod
    def update(self, terminated: bool):
        """
        Abstract method where the update rule is applied.

        :param terminated: Boolean indicating if the episode has terminated.
        """
        pass

    @abstractmethod
    def policy(self, state):
        """
        Abstract method to define the agent's policy.
        For actor-critic algorithms, the output of the actor would be a probability distribution over actions.
        For discrete actions, this is simply a discrete probability distribution, describing a probability
        for each action.
        For continuous actions, you can have some kind of continuous distribution you sample actions from.

        :param state: The current state of the environment.
        """
        pass

    @abstractmethod
    def save(self, file_path: str = './') -> None:
        """
        Abstract method to save the agent's model.

        :param file_path: The path to save the model.
        """
        pass

    @abstractmethod
    def load(self, file_path: str = './') -> None:
        """
        Abstract method to load the agent's model.

        :param file_path: The path to load the model from.
        """
        pass

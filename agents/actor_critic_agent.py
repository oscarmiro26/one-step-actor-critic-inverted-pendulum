# agents/actor_critic_agent.py

import gymnasium as gym
import torch
import numpy as np

from agents.abstract_agent import AbstractAgent
from models.mlp import MLP
from models.mlpmultivariategaussian import MLPMultivariateGaussian
from models.sampling import sample_two_headed_gaussian_model
from trainers.actrainer import ACTrainer
from util.device import fetch_device


class ActorCriticAgent(AbstractAgent):
    def __init__(self,
                 state_space: gym.Space,
                 action_space: gym.Space,
                 discount_factor: float = 0.9,
                 learning_rate_actor: float = 0.01,
                 learning_rate_critic: float = 0.05):
        super().__init__(state_space, action_space, discount_factor)
        """
        Initialize the Actor-Critic Agent.
        
        NOTE: One rule of thumb for the learning rates is that the learning rate of the actor should be lower
        than the critic. Intuitively because the estimated values of the critic are based on past policies,
        so the actor cannot "get ahead" of the critic.

        :param state_space: The state space of the environment.
        :param action_space: The action space of the environment.
        :param discount_factor: Discount factor for future rewards.
        :param learning_rate_actor: Learning rate for the actor model.
        :param learning_rate_critic: Learning rate for the critic model.
        """
        # Initialize Actor and Critic models
        if isinstance(action_space, gym.spaces.Discrete):
            self._actor_model = MLPMultivariateGaussian(input_size=state_space.shape[0],
                                                        output_size=action_space.n).to(device=fetch_device())
        elif isinstance(action_space, gym.spaces.Box):
            self._actor_model = MLPMultivariateGaussian(input_size=state_space.shape[0],
                                                        output_size=action_space.shape[0],
                                                        action_space=action_space).to(device=fetch_device())
        else:
            raise NotImplementedError("Unsupported action space type.")

        # Critic outputs a single scalar value
        self._critic_model = MLP(input_size=state_space.shape[0], output_size=1).to(device=fetch_device())

        # Initialize the Trainer
        self._trainer = ACTrainer(self._replay_buffer,
                                   self._actor_model,
                                   self._critic_model,
                                   learning_rate_actor,
                                   learning_rate_critic,
                                   discount_factor)

        self.device = fetch_device()

    def add_trajectory(self, trajectory: tuple) -> None:
        """
        Add a trajectory to the replay buffer.

        NOTE: One-step Actor Critic does not by default use a replay buffer.
        Therefore, the replay buffer is assumed to have a size of 1 which means
        it will only store the latest trajectory.
        The trainer will later sample from the buffer to retrieve the trajectory
        to apply the update rule.

        :param trajectory: The trajectory to add to the replay buffer.
                           Expected format: (states, actions, rewards, next_states)
        """
        states, actions, rewards, next_states, dones = trajectory

        # Convert to tensors with float32
        states_t = torch.tensor(states, device=self.device, dtype=torch.float32)
        actions_t = torch.tensor(actions, device=self.device, dtype=torch.float32) if isinstance(self.action_space, gym.spaces.Box) else torch.tensor(actions, device=self.device, dtype=torch.int64)
        rewards_t = torch.tensor(rewards, device=self.device, dtype=torch.float32)
        next_states_t = torch.tensor(next_states, device=self.device, dtype=torch.float32)
        dones_t = torch.tensor(dones, device=self.device, dtype=torch.float32)

        self._replay_buffer.add((states_t, actions_t, rewards_t, next_states_t, dones_t))

    def update(self, terminated: bool):
        """
        Perform a gradient descent step on both actor (policy) and critic (value function).
        """
        loss = self._trainer.train(terminated)
        return loss

    def policy(self, state) -> np.array:
        """
        Get the action to take based on the current state.

        :param state: The current state of the environment.
        :return: The action to take.
        """
        state = torch.from_numpy(state).to(device=self.device, dtype=torch.float32)

        action, _ = sample_two_headed_gaussian_model(self._actor_model, state)

        return action.cpu().numpy()

    def save(self, file_path='./') -> None:
        """
        Save the actor and critic models.

        :param file_path: The directory path to save the models.
        """
        torch.save(self._actor_model.state_dict(), f"{file_path}/actor_model.pth")
        torch.save(self._critic_model.state_dict(), f"{file_path}/critic_model.pth")

    def load(self, file_path='./') -> None:
        """
        Load the actor and critic models.

        :param file_path: The directory path to load the models from.
        """
        self._actor_model.load_state_dict(torch.load(f"{file_path}/actor_model.pth"))
        self._critic_model.load_state_dict(torch.load(f"{file_path}/critic_model.pth"))

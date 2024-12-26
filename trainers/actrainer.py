import torch
from torch import nn
from torchrl.data import ReplayBuffer

from models.twoheadedmlp import TwoHeadedMLP
from models.sampling import log_prob_policy
from util.device import fetch_device

""""
IMPORTANT: in pseudocode gradient ascent is performed. But PyTorch automatic differentiation
facilities perform gradient descent by default. Therefore, you should reverse the signs to turn gradient ascent
in the pseudocode to gradient descent.
"""


class ACTrainer:
    """
    One-step Actor-Critic Trainer based on Sutton and Barto's algorithm.
    """

    def __init__(self,
                 buf: ReplayBuffer,
                 actor_model: TwoHeadedMLP,
                 critic_model: nn.Module,
                 learning_rate_actor: float,
                 learning_rate_critic: float,
                 discount_factor: float):
        """
        Initialize the Actor-Critic Trainer.

        :param buf: ReplayBuffer for storing experiences.
        :param actor_model: The actor model (policy).
        :param critic_model: The critic model (value function).
        :param learning_rate_actor: Learning rate for the actor.
        :param learning_rate_critic: Learning rate for the critic.
        :param discount_factor: Discount factor for future rewards.
        """

        self.buf = buf
        self.device = fetch_device()
        self.discount_factor = discount_factor

        self.actor_model = actor_model
        self.critic_model = critic_model
        # Optimizes policy parameters
        self.actor_optimizer = torch.optim.Adam(self.actor_model.parameters(), lr=learning_rate_actor)
        # Optimizes critic parameters
        self.critic_optimizer = torch.optim.Adam(self.critic_model.parameters(), lr=learning_rate_critic)

    def _trajectory(self) -> tuple:
        """
        Sample the latest trajectory from the replay buffer.

        :return: A tuple containing the states, actions, rewards, and next states.
        """
        trajectories = self.buf.sample(batch_size=1)
        return trajectories[0], trajectories[1], trajectories[2], trajectories[3]

    def train(self, done):
        state, action, reward, next_state = self._trajectory()

        value = self.critic_model(state).squeeze()
        next_value = self.critic_model(next_state).squeeze() if not done else 0

        td_error = reward + self.discount_factor * next_value - value

        # Update Critic
        critic_loss = td_error.pow(2)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update Actor
        log_prob = log_prob_policy(self.actor_model, state, action)

        actor_loss = -log_prob * td_error.detach()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return actor_loss.item(), critic_loss.item()

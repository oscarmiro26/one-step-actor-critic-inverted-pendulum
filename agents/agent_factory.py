import gymnasium as gym

from agents.abstract_agent import AbstractAgent
from agents.actor_critic_agent import ActorCriticAgent
from agents.random_agent import RandomAgent


class AgentFactory:
    """
    Naive factory method implementation for
    RL agent creation.
    """

    @staticmethod
    def create_agent(
            agent_str: str,
            env: gym.Env,
            gamma: float = 0.99,
            actor_lr: float = 0.001,
            critic_lr: float = 0.001
    ) -> AbstractAgent:
        """
        Factory method for Agent creation.
        :param env: gymnasium environment.
        :param agent_type: a string key corresponding to the agent.
        :return: an object of type Agent.
        """
        obs_space = env.observation_space
        action_space = env.action_space

        if agent_str == "ACTOR-CRITIC-AGENT":
            return ActorCriticAgent(
                obs_space,
                action_space,
                discount_factor=gamma,
                learning_rate_actor=actor_lr,
                learning_rate_critic=critic_lr
            )
        elif agent_str == "RANDOM":
            return RandomAgent(obs_space, action_space)

        raise ValueError("Invalid agent type")

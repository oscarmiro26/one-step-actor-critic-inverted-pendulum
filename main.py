# main.py
import gymnasium as gym
from agents.agent_factory import AgentFactory


def run_episodes(
        env_str: str,
        agent_str: str,
        num_episodes: int,
        gamma: float,
        actor_lr: float,
        critic_lr: float
):
    env = gym.make(env_str, render_mode='rgb_array')
    obs, info = env.reset()

    agent = AgentFactory.create_agent(
        agent_str=agent_str,
        env=env,
        gamma=gamma,
        actor_lr=actor_lr,
        critic_lr=critic_lr
    )

    episode_return_list = []
    episode_actor_loss = []
    episode_critic_loss = []

    episode_return = 0
    while num_episodes > 0:
        old_obs = obs
        action = agent.policy(obs)

        obs, reward, terminated, truncated, info = env.step(action)
        episode_return += reward

        agent.add_trajectory((old_obs, action, reward, obs))
        loss = agent.update(terminated or truncated)

        if loss:
            actor_loss, critic_loss = loss
            episode_actor_loss.append(actor_loss)
            episode_critic_loss.append(critic_loss)

        if terminated or truncated:
            num_episodes -= 1
            episode_return_list.append(episode_return)
            episode_return = 0

            obs, info = env.reset()

    env.close()
    return episode_return_list, episode_actor_loss, episode_critic_loss


def compare_agents(
        env_str: str,
        agent_str: str,
        num_episodes: int,
        gamma: float,
        actor_lr: float,
        critic_lr: float
):
    
    num_runs = 1

    returns_lst = []
    actor_loss = []
    critic_loss = []

    for run in range(num_runs):
        print(f'Run {run + 1} of {num_runs}.')
        returns, run_actor_loss, run_critic_loss = run_episodes(
            env_str,
            agent_str,
            num_episodes,
            gamma,
            actor_lr,
            critic_lr
        )

        returns_lst.append(returns)
        if run_actor_loss and run_critic_loss:
            actor_loss.append(run_actor_loss)
            critic_loss.append(run_critic_loss)

    return returns_lst, actor_loss, critic_loss

def run_default(env_str: str, agent_str: str, num_episodes: int):
    gamma, actor_lr, critic_lr = 0.9, 0.01, 0.05


    returns, run_actor_loss, run_critic_loss = compare_agents(
        env_str=env_str, 
        agent_str=agent_str, 
        num_episodes=num_episodes,
        gamma=gamma,
        actor_lr=actor_lr,
        critic_lr=critic_lr
    )

    print(returns)


def main():
    env_str = 'CartPole-v1'
    agent_list = ["RANDOM", "ACTOR-CRITIC"]
    num_episodes = 1000
    num_runs = 500

    run_default(env_str, agent_str=agent_list[0], num_episodes=num_episodes)


if __name__ == "__main__":
    main()

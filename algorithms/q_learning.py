import numpy as np
import random
import gym
from typing import Union
from abc import ABC, abstractmethod
from stable_baselines3.common.vec_env import VecEnv

Env = Union[gym.Env, VecEnv]
State = np.ndarray
QTable = np.ndarray


class EpsilonStrategy(ABC):
    @abstractmethod
    def get_epsilon(self) -> float:
        pass

    @abstractmethod
    def update(self, episode_num: int):
        pass


class ConstantEpsilon(EpsilonStrategy):
    def __init__(self, epsilon: float):
        self.epsilon = epsilon

    def get_epsilon(self) -> float:
        return self.epsilon

    def update(self, episode_num: int):
        pass


class LinearEpsilon(EpsilonStrategy):
    def __init__(self, start: float = 1, min: float = 0.01, decay_rate: float = 0.001):
        self.min = min
        self.decay_rate = decay_rate
        self.current = start

    def get_epsilon(self) -> float:
        return self.current

    def update(self, episode_num: int):
        self.current = max(self.current - self.decay_rate, self.min)


class ExponentialEpsilon(EpsilonStrategy):
    def __init__(self, start: float = 1, min: float = 0.01, decay_rate: float = 0.001):
        self.min = min
        self.decay_rate = decay_rate
        self.current = start
        self.range = start - min

    def get_epsilon(self) -> float:
        return self.current

    def update(self, episode_num: int):
        self.current = self.min + (self.range) * np.exp(-self.decay_rate * episode_num)


class ExplorationStrategy(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def sample_action(self, env: Env, state: State, q_table: QTable):
        pass

    @abstractmethod
    def update(self, episode_num: int):
        pass


class GreedyStrategy(ExplorationStrategy):
    """Exploit-only strategy. Takes the action with the highest reward"""

    def __init__(self):
        pass

    def sample_action(self, env: Env, state: State, q_table: QTable):
        # Choose the action with the highest value in the current state
        if np.max(q_table[state]) > 0:
            return np.argmax(q_table[state])
        # If there's no best action (only zeros), take a random one
        else:
            return env.action_space.sample()

    def update(self, episode: int):
        pass


class EpsilonGreedyStrategy(ExplorationStrategy):
    def __init__(
        self,
        epsilon_strategy: EpsilonStrategy = ExponentialEpsilon(),
    ):
        self.epsilon_strategy = epsilon_strategy

    def sample_action(self, env: Env, state: State, q_table: QTable):
        exploration_rate_threshold = random.uniform(0, 1)
        if exploration_rate_threshold > self.epsilon_strategy.get_epsilon():
            # Choose action with highest q value for the state
            return np.argmax(q_table[state])
        else:
            # Choose random action
            return env.action_space.sample()

    def update(self, episode_num: int):
        self.epsilon_strategy.update(episode_num)


class QLearning:
    """Q Learning algorithm"""

    def __init__(
        self,
        env: Env,
        learning_rate: int = 0.1,
        discount_factor: int = 0.99,
        num_episodes: int = 1000,
        max_steps_per_episode: int = 1000,
        exploration_strategy: ExplorationStrategy = EpsilonGreedyStrategy(),
    ):
        """Initialize the QLearning model with

        Args:
            env (Env): The OpenAI gym environment to learn
            learning_rate (int, optional): Defines how much should the values in the Q Table change with each update. Lower values give more importance to past knowledge while higher values prioritize new knowledge. Defaults to 0.1.
            discount_factor (int, optional): Defines how much weight to give to the following states. High values mean the agent values long-term action, while lower values mean that the agent focuses on the immediate action. For sparse environments, it is recommended to set this to a high value. Defaults to 0.99.
            num_episodes (int, optional): The number of episodes to learn from. The end of any episode is determined by the environment. Defaults to 1000.
            max_steps_per_episode (int, optional): The maximum steps before an episode gets terminated regardless of whether it has finished or not. Helps guard against the agent getting stuck. Defaults to 1000.
            exploration_strategy (ExplorationStrategy, optional): The exploration strategy to follow. Determines how the agent chooses actions. Defaults to EpsilonGreedyStrategy().
        """
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_strategy = exploration_strategy
        self.num_episodes = num_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.q_table = np.zeros((self.env.observation_space.n, self.env.action_space.n))
        self.rewards_all_episodes = []  # contains sum of rewards for each episode

        print(f"Q-Learning with {self.exploration_strategy.__class__.__name__} exploration strategy")

    def update_qtable(self, state: State, new_state: State, action, reward):
        self.q_table[state, action] = (
            self.q_table[state, action] * (1 - self.learning_rate)
            + (reward + self.discount_factor * np.max(self.q_table[new_state])) * self.learning_rate
        )

    def reset(self):
        self.q_table = np.zeros((self.env.observation_space.n, self.env.action_space.n))
        self.rewards_all_episodes = []

    def learn(self):
        """Run the algorithm to learn and update the Q Table"""
        self.reset()
        for episode in range(self.num_episodes):
            state = self.env.reset()

            done = False
            rewards_current_episode = 0

            for step in range(self.max_steps_per_episode):

                # select an action based on the exploration strategy
                action = self.exploration_strategy.sample_action(self.env, state, self.q_table)

                new_state, reward, done, info = self.env.step(action)

                self.update_qtable(state, new_state, action, reward)

                state = new_state
                rewards_current_episode += reward

                if done:
                    break

            self.exploration_strategy.update(episode)

            self.rewards_all_episodes.append(rewards_current_episode)

    def evaluate(self, num_episodes=1000):
        """Evaluate the performance of the models

        Args:
            num_episodes (int, optional): The number of episodes to evaluate. The end of any episode is determined by the environment. Defaults to 1000.
        """
        total_reward = 0

        for episode in range(num_episodes):
            state = self.env.reset()
            done = False

            while not done:
                action = self.predict(state)

                new_state, reward, done, info = self.env.step(action)

                state = new_state
                total_reward += reward

        print(f"Average reward: {total_reward/num_episodes}\nTotal reward: {total_reward}")

    def predict(self, observation: State):
        return np.argmax(self.q_table[observation, :])

    def print_info(self, divisions: int = 10, print_qtable: bool = False):
        divisions = 10
        episodes_per_division = self.num_episodes / divisions
        rewards_per_thousand_episodes = np.split(np.array(self.rewards_all_episodes), divisions)
        count = episodes_per_division
        print("--------Average reward per thousand episodes--------")
        for r in rewards_per_thousand_episodes:
            print(count, ": ", str(sum(r / episodes_per_division)))
            count += episodes_per_division

        if print_qtable:
            print("\n---------------------Q Table------------------------")
            print(self.q_table)

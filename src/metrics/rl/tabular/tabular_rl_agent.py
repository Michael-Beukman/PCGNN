from typing import List, Tuple
import gym
import numpy as np
from gym import spaces

class TabularRLAgent:
    """
        This is a simple RL agent to perform tabular Q-learning on discrete environments.
        methods:
        
        
        train(self, env: gym.Env, episodes: int = 100, verbose: bool=False) -> List[float]
            Train the agent for some number of episodes on the environment. Return list of rewards per episode
        
        eval(self, env: gym.Env, eps: int) -> List[float]:        
            Evaluate the agent by setting alpha = epsilon = 0. Returns a list of rewards for each episode

        get_trajectory(self, env: gym.Env) -> List[int]
            Returns a list of states visited during one evaluation run.

        get_action(self, state: int) -> int
            Returns an action in an epsilon greedy fashion for a single state.

    """
    def __init__(self, num_observations: int, num_actions: int, alpha: float = 0.1, epsilon: float = 0.05, gamma: float = 0.99):
        """
        Args:
            num_observations (int): The number of discrete states. For a 14x14 maze this is 196.
            num_actions (int): The number of actions. For a maze this is 4
            alpha (float, optional): The learning rate. Defaults to 0.1.
            epsilon (float, optional): Fraction of actions that are chosen randomly. Defaults to 0.05.
            gamma (float, optional): The reward discount factor. Defaults to 0.99.
        """
        self.table = np.zeros((num_observations, num_actions))
        self.alpha = alpha
        self.epsilon = epsilon
        self.num_observations = num_observations
        self.num_actions = num_actions
        self.gamma = gamma
    

    def train(self, env: gym.Env, episodes: int = 100, verbose: bool=False) -> List[float]:
        assert isinstance(env.observation_space, spaces.Discrete)
        rewards = []
        for episode in range(1, episodes + 1):
            done = False
            state = env.reset()
            assert type(state) == int or type(state) == np.int32 or type(state) == np.int64, type(state)
            tot_r = 0
            while not done:
                action = self.get_action(state)
                next_state, reward, done, info = env.step(action)
                Qsa = self.table[state, action]
                best_next = self.table[next_state].max() 
                # Since this is tabular, we always have that table[terminal, :] = 0, because:
                # - We initialised that to 0 initially
                # - we never update table[terminal, :], because if next_state is terminal, the loop ends,
                # and we don't update it.
                # Q learning update rule, from Sutton And Barto 2018, p131
                self.table[state, action] = Qsa + self.alpha * \
                                                (reward + self.gamma * best_next - Qsa)
                state = next_state
                tot_r += reward
            rewards.append(tot_r)
            if verbose and episode % 100 == 0:
                print(f"Average over 100 at ep {episode} = {np.mean(rewards[-100:])}")
        return rewards
    
    def eval(self, env: gym.Env, eps: int) -> List[float]:
        self.epsilon = 0
        self.alpha  = 0
        return self.train(env, eps)
    
    def eval_difficulty(self, env: gym.Env, eps) -> List[List[float]]:
        rewards = []
        self.epsilon = 0
        for e in range(eps):
            done = False
            state = env.reset()
            rs =[]
            while not done:
                action = self.get_action(state)
                next_state, reward, done, info = env.step(action)
                state = next_state
                rs.append(reward)
            rewards.append(rs)
        return rewards


    


    def get_trajectory(self, env: gym.Env, return_actions: bool = False) -> List[int]:
        states = []
        actions = []
        self.epsilon = 0
        done = False
        state = env.reset()
        states.append(state)
        assert type(state) == int or type(state) == np.int32 or type(state) == np.int64, type(state)
        while not done:
            action = self.get_action(state)
            actions.append(action)
            next_state, reward, done, info = env.step(action)
            state = next_state
            states.append(state)
        if return_actions:
            return states, actions
        return states
    
    def get_action(self, state: int) -> int:
        if np.random.rand() < self.epsilon: return np.random.randint(0, self.num_actions)
        action_row = self.table[state]    
        bests = np.where(action_row == action_row.max())[0]
        if len(bests) == 1: return bests[0]
        return np.random.choice(bests)



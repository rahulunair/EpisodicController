"""An  implementation of model free episodic controller
    Arxiv paper: https://arxiv.org/pdf/1606.04460.pdf
    Other implementations:
    http://www.gitxiv.com/posts/HQJ3F9YzsQZ3eJjpZ/model-free-episodic-control"""

import operator
import random

import numpy as np

from episodic_memory import Memory

np.random.seed(42)


class HPS(object):
    epsilon = 1.0
    epsilon_decay = 0.98
    epsilon_min = 0
    neighbours = 50
    memory_size = 50000
    gamma = 0.95


class EPCAgent(object):
    """An RL agent that works using nearest neighbour principle."""

    def __init__(self, state_size, action_size):
        self.action_size = action_size
        self.state_size = state_size
        self.eps = HPS.epsilon
        self.eps_decay = HPS.epsilon_decay
        self.eps_min = HPS.epsilon_min
        self.gamma = HPS.gamma
        self.neighbours = HPS.neighbours
        self.memory_size = HPS.memory_size
        self.memory = Memory(self.memory_size, self.state_size)
        # save hyper parameters

    def _action(self, state):
        if np.random.rand() <= self.eps:
            return random.randrange(self.action_size)
        else:
            return self._predict(state)

    def _predict(self, state):
        """predict action values based on states"""
        action_val = {}
        action_count = {}
        indices = self.memory.k_nearest(state, self.neighbours)
        for i in indices:
            value = self.memory.values[i]
            action = self.memory.actions[i]
            new_value = action_val.get(action, 0) + value
            action_val[action] = new_value
            action_count[action] = action_count.get(action, 0) + 1
        for n in action_count:  # may be better to vectorize this
            action_val[n] = action_val[n] / action_count[n]
        sorted_act = sorted(action_val.items(), key=operator.itemgetter(1))
        return sorted_act[-1][0]

    def _remember(self, state, action, reward):
        if self.memory.index < self.memory.size:
            self.memory.states[self.memory.index] = state
            self.memory.actions[self.memory.index] = action
            self.memory.rewards[self.memory.index - 1] = reward
            self.memory.values[self.memory.index - 1] = 0
            self.memory.index += 1
        else:
            self.memory.index = 0

    def play(self, state, reward, done):
        act = self._action(state)
        self._remember(state, act, reward)
        if done:
            value = 0
            for i in reversed(range(self.memory.episode_pointer, self.memory.index)):
                value = self.gamma * value + self.memory.rewards.get(i, 0)
                self.memory.values[i] = value

            self.memory.episode_pointer = self.memory.index
            self.memory.end = min(max(self.memory.end, self.memory.index),
                                  self.memory.size)

            self.eps *= self.eps_decay
            self.eps = max(self.eps, self.eps_min)
            print("Size of experience: ", self.memory.index)
        return act

    def save(self):
        pass

    def load(self):
        pass

import numpy as np
import random

class ArmedBandit:
    def __init__(self, k=10):
        self.action_values = [np.random.normal(0, 1) for k in range(10)]
    
    def get_reward(self, action):
        return np.random.normal(self.action_values[action], 1)

class ArmedBanditRunner:
    def __init__(self, steps, runs, bandits=10, episolon=0.1):
        self.STEPS = steps
        self.RUNS = runs
        self.episolon = episolon
        self.bandits = bandits

    def run(self):
        means = np.zeros(self.STEPS)
        for run in range(self.RUNS):
            Q = np.zeros(self.bandits)
            reward_sums = np.zeros(self.bandits)
            taken = np.zeros(self.bandits)
            ab = ArmedBandit(self.bandits)
            for step in range(self.STEPS):
                action = np.argmax(Q) if random.random() < 1 - self.episolon else random.randint(0,self.bandits-1)
                r = ab.get_reward(action)
                taken[action] += 1
                reward_sums[action] += r
                Q[action] = reward_sums[action]/taken[action]
                means[step] += r
        means = means/self.RUNS

        return means
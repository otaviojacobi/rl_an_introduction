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
        optimal = np.zeros(self.STEPS)

        # Suggested algorithm on section 2.4
        for run in range(self.RUNS):
            Q = np.zeros(self.bandits)
            taken = np.zeros(self.bandits)
            ab = ArmedBandit(self.bandits)
            for step in range(self.STEPS):
                action = np.argmax(Q) if random.random() < 1 - self.episolon else random.randint(0,self.bandits-1)
                r = ab.get_reward(action)
                taken[action] += 1
                Q[action] = Q[action] + (1.0/taken[action]) * (r - Q[action])

                #this is for plotting
                means[step] += r
                optimal[step] += 1 if np.argmax(ab.action_values) == action else 0

        means = means/self.RUNS
        optimal = optimal/self.RUNS

        return means, optimal
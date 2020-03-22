import numpy as np
import random

class ArmedBandit:
    def __init__(self, k=10):
        self.action_values = [np.random.normal(0, 1) for _ in range(k)]
    
    def get_reward(self, action):
        return np.random.normal(self.action_values[action], 1)

class ArmedBanditRunner:
    def __init__(self, steps, runs, bandits=10, episolon=0.1):
        self.STEPS = steps
        self.RUNS = runs
        self.episolon = episolon
        self.bandits = bandits

    def run(self, step_size=0.1, init_q=0):
        means = np.zeros(self.STEPS)
        optimal = np.zeros(self.STEPS)

        # Suggested algorithm on section 2.4
        for run in range(self.RUNS):
            Q = np.zeros(self.bandits) if init_q == 0 else np.ones(self.bandits) * init_q 
            taken = np.zeros(self.bandits)
            ab = ArmedBandit(self.bandits)
            for step in range(self.STEPS):
                action = np.argmax(Q) if random.random() < 1 - self.episolon else random.randint(0,self.bandits-1)
                r = ab.get_reward(action)
                taken[action] += 1
                Q[action] = Q[action] + step_size * (r - Q[action])

                #this is for plotting
                means[step] += r
                optimal[step] += 1 if np.argmax(ab.action_values) == action else 0

        means = means/self.RUNS
        optimal = optimal/self.RUNS

        return means, optimal
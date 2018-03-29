import numpy as np
import random

class ArmedBanditNonstationary:
    def __init__(self, k=10):
        action_value = np.random.normal(0, 1)
        self.action_values = [action_value for _ in range(k)]
    
    def get_reward(self, action):
        return np.random.normal(self.action_values[action], 1)

    def update_action_values(self, mean, sd):
        self.action_values = [self.action_values[i]*np.random.normal(mean, sd) for i in range(len(self.action_values))] 

class ArmedBanditNonstationaryRunner:
    def __init__(self, steps, runs, bandits=10, episolon=0.1):
        self.STEPS = steps
        self.RUNS = runs
        self.episolon = episolon
        self.bandits = bandits

    def run(self):

        alpha = 0.1
        action_values_update_mean = 0
        action_values_update_sd = 0.01

        means = np.zeros(self.STEPS)
        optimal = np.zeros(self.STEPS)

        # Suggested algorithm on section 2.4
        for run in range(self.RUNS):
            if run%100 == 0:
                print(run)
            Q = np.zeros(self.bandits)
            taken = np.zeros(self.bandits)
            ab = ArmedBanditNonstationary(self.bandits)
            for step in range(self.STEPS):
                action = np.argmax(Q) if random.random() < 1 - self.episolon else random.randint(0,self.bandits-1)
                r = ab.get_reward(action)
                taken[action] += 1
                Q[action] = Q[action] + alpha * (r - Q[action])

                #Non Stationary now
                ab.update_action_values(action_values_update_mean, action_values_update_sd)

                #this is for plotting
                means[step] += r
                optimal[step] += 1 if np.argmax(ab.action_values) == action else 0

        means = means/self.RUNS
        optimal = optimal/self.RUNS

        return means, optimal
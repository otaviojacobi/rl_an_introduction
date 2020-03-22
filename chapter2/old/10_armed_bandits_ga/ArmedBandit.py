import numpy as np
import random

class ArmedBandit:
    def __init__(self, k=10, mean=0):
        self.action_values = [np.random.normal(mean, 1) for _ in range(k)]
    
    def get_reward(self, action):
        return np.random.normal(self.action_values[action], 1)

class ArmedBanditRunner:
    def __init__(self, steps, runs, bandits=10, episolon=0.1, mean=0):
        self.STEPS = steps
        self.RUNS = runs
        self.episolon = episolon
        self.bandits = bandits
        self.mean = mean

    def softmax(self, H):
        den = np.sum(np.exp(H))
        out = np.zeros(len(H))
        for k in range(len(H)):
            out[k] = np.exp(H[k])
        return out/den

    def run(self, step_size=0.1, init_q=0, baseline=False):
        means = np.zeros(self.STEPS)
        optimal = np.zeros(self.STEPS)

        # Suggested algorithm on section 2.4
        for run in range(self.RUNS):
            H = np.zeros(self.bandits)
            R = 0
            taken = np.zeros(self.bandits)
            ab = ArmedBandit(self.bandits, mean = self.mean)
            for step in range(self.STEPS):
                probs = self.softmax(H)
                action = np.random.choice(np.arange(0,10), p=probs)
                r = ab.get_reward(action)
                taken[action] += 1
                R += r
                Rt_mean = R/np.sum(taken) if baseline else 0
                for act in range(len(H)):
                    H[act] = H[act] + step_size*(r - Rt_mean)*(1 - probs[act]) if act == action else H[act] - step_size*(r - Rt_mean)*(probs[act])
                #this is for plotting
                means[step] += r
                optimal[step] += 1 if np.argmax(ab.action_values) == action else 0

        means = means/self.RUNS
        optimal = optimal/self.RUNS

        return means, optimal
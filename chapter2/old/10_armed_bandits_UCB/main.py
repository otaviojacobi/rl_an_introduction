
import matplotlib.pyplot as plt
from ArmedBandit import *

def main():
    '''
    - ε-greedy policy implementation for a stateless problem
    '''
    STEPS = 1000
    RUNS = 2000

    reward_regular, _ = ArmedBanditRunner(STEPS, RUNS, bandits=10, episolon=0.1).run()
    reward_UCB, _ = ArmedBanditRunner(STEPS, RUNS, bandits=10, episolon=0.1).run(policy='UCB', c=2)

    plt.plot([k+1 for k in range(STEPS)], reward_regular, color='green', label='ε-greedy, ε=0.1')
    plt.plot([k+1 for k in range(STEPS)], reward_UCB, color='black', label='UCB, c=2')
    plt.legend(loc=4, prop={'size': 17})
    plt.xlabel('Steps')
    plt.ylabel('Average reward')
    plt.show()

if __name__ == '__main__':
    main()
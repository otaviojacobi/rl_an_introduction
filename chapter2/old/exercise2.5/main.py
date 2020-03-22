
import matplotlib.pyplot as plt
from ArmedBandit import *

def main():
    '''
    - ε-greedy policy implementation for a stateless problem
    '''
    STEPS = 10000
    RUNS = 2000

    means01, optimal01 = ArmedBanditNonstationaryRunner(STEPS, RUNS, bandits=10, episolon=0.1).run()

    plt.plot([k+1 for k in range(STEPS)], means01, color='black', label='ε=0.1')
    plt.legend(loc=4, prop={'size': 17})
    plt.xlabel('Steps')
    plt.ylabel('Average reward')
    plt.show()

    plt.plot([k+1 for k in range(STEPS)], optimal01, color='black', label='ε=0.1')
    plt.legend(loc=4, prop={'size': 17})
    plt.xlabel('Steps')
    plt.ylabel('% Optimal action')
    plt.show()


if __name__ == '__main__':
    main()
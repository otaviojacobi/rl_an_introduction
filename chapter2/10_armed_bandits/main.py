
import matplotlib.pyplot as plt
from ArmedBandit import *

def main():
    '''
    - ε-greedy policy implementation for a stateless problem
    '''
    STEPS = 1000
    RUNS = 2000

    means0, optimal0 = ArmedBanditRunner(STEPS, RUNS, bandits=10, episolon=0).run()
    means01, optimal01 = ArmedBanditRunner(STEPS, RUNS, bandits=10, episolon=0.1).run()
    means001, optimal001 = ArmedBanditRunner(STEPS, RUNS, bandits=10, episolon=0.01).run()

    plt.plot([k+1 for k in range(STEPS)], means0, color='green', label='ε=0')
    plt.plot([k+1 for k in range(STEPS)], means01, color='black', label='ε=0.1')
    plt.plot([k+1 for k in range(STEPS)], means001, color='red', label='ε=0.01')
    plt.legend(loc=4, prop={'size': 17})
    plt.xlabel('Steps')
    plt.ylabel('Average reward')
    plt.show()

    plt.plot([k+1 for k in range(STEPS)], optimal0, color='green', label='ε=0')
    plt.plot([k+1 for k in range(STEPS)], optimal01, color='black', label='ε=0.1')
    plt.plot([k+1 for k in range(STEPS)], optimal001, color='red', label='ε=0.01')
    plt.legend(loc=4, prop={'size': 17})
    plt.xlabel('Steps')
    plt.ylabel('% Optimal action')
    plt.show()


if __name__ == '__main__':
    main()
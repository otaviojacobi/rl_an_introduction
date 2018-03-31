
import matplotlib.pyplot as plt
from ArmedBandit import *

def main():
    '''
    - ε-greedy policy implementation for a stateless problem
    '''
    STEPS = 1000
    RUNS = 2000

    _, optimal0 = ArmedBanditRunner(STEPS, RUNS, bandits=10, episolon=0).run(init_q = 5)
    _, optimal01 = ArmedBanditRunner(STEPS, RUNS, bandits=10, episolon=0.1).run()

    plt.plot([k+1 for k in range(STEPS)], optimal0, color='black', label='Q1=5, ε=0')
    plt.plot([k+1 for k in range(STEPS)], optimal01, color='green', label='Q1=0, ε=0.1')
    plt.legend(loc=4, prop={'size': 17})
    plt.xlabel('Steps')
    plt.ylabel('% Optimal action')
    plt.show()

if __name__ == '__main__':
    main()
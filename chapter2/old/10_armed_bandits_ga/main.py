
import matplotlib.pyplot as plt
from ArmedBandit import *

def main():
    '''
    - ε-greedy policy implementation for a stateless problem
    '''
    STEPS = 1000
    RUNS = 2000

    _, with_baseline_01 = ArmedBanditRunner(STEPS, RUNS, bandits=10, mean=4).run(step_size=0.1, baseline=True)
    _, with_baseline_04 = ArmedBanditRunner(STEPS, RUNS, bandits=10, mean=4).run(step_size=0.4, baseline=True)
    _, without_baseline_01 = ArmedBanditRunner(STEPS, RUNS, bandits=10, mean=4).run(step_size=0.1, baseline=False)
    _, without_baseline_04 = ArmedBanditRunner(STEPS, RUNS, bandits=10, mean=4).run(step_size=0.4, baseline=False)


    plt.plot([k+1 for k in range(STEPS)], with_baseline_01, color='green', label='with baseline a = 0.1')
    plt.plot([k+1 for k in range(STEPS)], with_baseline_04, color='black', label='with baseline a = 0.4')
    plt.plot([k+1 for k in range(STEPS)], without_baseline_01, color='red', label='without baseline a = 0.1')
    plt.plot([k+1 for k in range(STEPS)], without_baseline_04, color='blue', label='without baseline a = 0.4')
    plt.legend(loc=4, prop={'size': 15})
    plt.xlabel('Steps')
    plt.ylabel('% Optimal action')
    plt.show()

if __name__ == '__main__':
    main()
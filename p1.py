# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import math
import statistics as stats

# Define Action class
class Actions:
    def __init__(self, m, id):
        self.m = m
        self.mean = 0
        self.N = 0
        self.id = id
        self.vals = []
    # Choose a random action
    def choose(self):
        return np.random.randn() + self.m
    # Update the action-value estimate
    def inc_update(self, x):
        self.N += 1
        self.mean = (1 - 1.0 / self.N)*self.mean + 1.0 / self.N * x
    def greedy(self):
        self.mean = self.m
    def update(self,x):
        self.N += 1
        self.vals.append(x)
        self.mean = stats.mean(self.vals)
    def print(self):
        print("\tArm", self.id,":\tM = ", self.m, "\tMean  = ", self.mean, "\tN = ", self.N)

"""# Greedy Approach"""

def greedy_experiment(m1, m2, m3, N):
    actions = [Actions(m1,1), Actions(m2,2), Actions(m3,3)]
    for a in actions:
        a.greedy()
    data = np.empty(N)  
    for i in range(N):
    # epsilon greedy
        p = np.random.random()
        j = np.argmax([a.mean for a in actions])
        x = actions[j].choose()
        actions[j].inc_update(x)
        # for the plot
        data[i] = x
    cumulative_average = np.cumsum(data) / (np.arange(N) + 1)
    # plot moving average ctr
    plt.plot(cumulative_average, label='Average Reward')
    plt.plot(np.ones(N)*m1, label='Reward for Arm 1')
    plt.plot(np.ones(N)*m2, label='Reward for Arm 2')
    plt.plot(np.ones(N)*m3, label='Reward for Arm 3')
    plt.xscale('log')
    plt.xlabel('Number of Trials')
    plt.ylabel('Reward')
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.show()    
    for a in actions:
        a.print()
    return cumulative_average

exp1 = greedy_experiment(2.0, 2.0, 3.0, 100000)
print(exp1)

"""# Epsilon Greedy Approach"""

def eps_greedy_experiment(m1, m2, m3, eps, N):
    actions = [Actions(m1,1), Actions(m2,2), Actions(m3,3)]
    data = np.empty(N)  
    for i in range(N):
    # epsilon greedy
        p = np.random.random()
        if p < eps:
            j = np.random.choice(3)
        else:
            j = np.argmax([a.mean for a in actions])
        x = actions[j].choose()
        actions[j].inc_update(x)
        # for the plot
        data[i] = x
    cumulative_average = np.cumsum(data) / (np.arange(N) + 1)
    # plot moving average ctr
    plt.plot(cumulative_average,label='Average Reward')
    plt.plot(np.ones(N)*m1, label='Reward for Arm 1')
    plt.plot(np.ones(N)*m2, label='Reward for Arm 2')
    plt.plot(np.ones(N)*m3, label='Reward for Arm 3')
    plt.xscale('log')
    plt.xlabel('Number of Trials')
    plt.ylabel('Reward')
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.show()
    for a in actions:
        a.print()   
    return cumulative_average

exp2 = eps_greedy_experiment(2.0, 2.0, 3.0, 0.05, 100000)
print(exp2)
exp3 = eps_greedy_experiment(1.0, 2.0, 3.0, 0.01, 100000)
print(exp3)
exp4 = eps_greedy_experiment(1.0, 2.0, 3.0, 0, 100000)
print(exp4)

"""# Incremental Approach"""

def inc_experiment(m1, m2, m3, N):
    actions = [Actions(m1,1), Actions(m2,2), Actions(m3,3)]
    data = np.empty(N)  
    for i in range(N):
    # epsilon greedy
        p = np.random.random()
        j = np.argmax([a.mean for a in actions])
        x = actions[j].choose()
        actions[j].inc_update(x)
        # for the plot
        data[i] = x
    cumulative_average = np.cumsum(data) / (np.arange(N) + 1)
    # plot moving average ctr
    plt.plot(cumulative_average,label='Average Reward')
    plt.plot(np.ones(N)*m1, label='Reward for Arm 1')
    plt.plot(np.ones(N)*m2, label='Reward for Arm 2')
    plt.plot(np.ones(N)*m3, label='Reward for Arm 3')
    plt.xscale('log')
    plt.xlabel('Number of Trials')
    plt.ylabel('Reward')
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.show()
    for a in actions:
        a.print()   
    return cumulative_average

exp5 = greedy_experiment(2.0, 2.0, 3.0, 100000)
print(exp5)

"""# UCB Approach"""

def ucb_experiment(m1, m2, m3, c, N):
    actions = [Actions(m1,1), Actions(m2,2), Actions(m3,3)]
    data = np.empty(N)  
    for i in range(N):
    # UCB
        p = np.random.random()
        j = np.argmax([a.mean+ c*math.sqrt(math.log(i+1)/(a.N+1))  for a in actions])
        x = actions[j].choose()
        actions[j].inc_update(x)
        # for the plot
        data[i] = x
    cumulative_average = np.cumsum(data) / (np.arange(N) + 1)
    # plot moving average ctr
    plt.plot(cumulative_average,label='Average Reward')
    plt.plot(np.ones(N)*m1, label='Reward for Arm 1')
    plt.plot(np.ones(N)*m2, label='Reward for Arm 2')
    plt.plot(np.ones(N)*m3, label='Reward for Arm 3')
    plt.xscale('log')
    plt.xlabel('Number of Trials')
    plt.ylabel('Reward')
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.show()
    for a in actions:
        a.print()   
    return cumulative_average

exp_6 = ucb_experiment(1.0, 2.0, 3.0, 0.9, 100000)
print(exp_6[-1])
exp_6 = ucb_experiment(1.0, 2.0, 3.0, 0.95, 100000)
print(exp_6[-1])

# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import math
import statistics as stats

class Arm:
    def __init__(self, m, sd, id):
        self.loc = m
        self.scale = sd
        self.id = id
        self.mean = m
        self.N  = 0
    def reward(self):
        return np.random.normal(loc=self.loc, scale=self.scale)
    def inc_update(self, x):
        self.N += 1
        self.mean = (1 - 1.0 / self.N) * self.mean + 1.0 / self.mean * x
    def set_mean_reward(self, mean):
        self.mean = mean
    def print(self):
        print("\nArm ",self.id, ": \n\tArgs of normal distribution for reward = ", self.loc, "and", self.scale, "\t Mean Reward = ", self.mean)

def naive_pac(actions, eps, delta):
    k = len(actions)
    l = int(round((4 / eps ** 2) * np.log((2 * k) / delta),0))
    print("No of samples per arm: ",l)
    avg = []
    tot_rewards = []
    for a in actions:
        rewards = []
        for i in range(l):
            rewards.append(a.reward())
        Pa = stats.mean(rewards)
        a.set_mean_reward(Pa)
        avg.append(Pa)
        tot_rewards.append(rewards)
    arm = np.argmax(avg) + 1
    for i in range(len(tot_rewards)):
        plt.plot(tot_rewards[i], label='reward')
        plt.plot(np.ones(l) * actions[i].mean, label='mean reward')
        plt.plot(np.ones(l) * actions[i].loc, label='dist mean reward')
        plt.xlabel("No of Trials")
        plt.ylabel("Reward")
        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        plt.title(f'Rewards for Arm {i + 1}')
        plt.show()

def median_pac(actions, eps, delta):
    S = [actions]
    while len(S[-1]) > 1:
        l = (4 / (eps ** 2)) * math.log(3 / delta)
        for a in S[-1]:
            rewards = []
            for i in range(l):
                rewards.append(a.reward())
            Pa = stats.mean(rewards)
            a.set_mean_reward(Pa)
        S[-1].sort(key=lambda x: x.Mean)
        S.append(S[len(S)//2:]) 
    arm = S[-1][0]

actions = [Arm(2,1,1), Arm(3,1,2), Arm(4,1,3)]
naive_pac(actions, 1, 0.1)


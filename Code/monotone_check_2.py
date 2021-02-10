import numpy as np
import matplotlib.pyplot as plt

def generateRewards(l,b,alpha,delta):
    isSatisfied = False
    rewards = []
    
    while not isSatisfied:
        r = np.random.gamma(alpha,delta)
        if sum(rewards)+r >= (b**len(rewards))*l :
            isSatisfied = True
        else:
            rewards.append(r)
    
    return rewards


alpha = 2
delta = 1/3
rewards = generateRewards(100,0.97,alpha,delta)
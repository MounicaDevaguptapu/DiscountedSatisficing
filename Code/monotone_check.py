import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy import stats

lambdas = np.linspace(1,500,5)
# print(lambdas)
betas = np.arange(0.1,0.97,0.02)
alpha = 3
delta = 1/9
e = 2.71828

def fact(n):
    if n==0 or n==1:
        return 1
    else:
        pro = 1
        for i in range(1,n+1):
            pro *= i
    return pro

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


        
def cdf(alpha,delta,x):
    # print(stats.gamma.pdf(x,alpha,scale=delta))
    return stats.gamma.cdf(x,alpha,scale=delta)

def F(l,b,rewards):
    t = len(rewards)
    z = (b**(t-1))*l
    y = (b**(t-2))*l
            
    cdf1 = cdf((t-1)*alpha,delta,y)
    cdf2 = cdf((t-1)*alpha,delta,z)
    cdf3 = cdf(alpha,delta,z)
    cdf4 = cdf(t*alpha,delta,z)

    val = (cdf1-(cdf2*cdf3))*(1-cdf4)
    # val = 1 - cdf4

    return val



def main():
    realLambda = 100
    realBeta = 0.87
    x,y = np.meshgrid(lambdas,betas)
    rewards = {}
    z = [0]*100
    for i in range(100):
        rewards[i] = generateRewards(realLambda,realBeta,alpha,delta)
    
        z[i] = F(x,y,rewards[i])
    
    

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.contour3D(x,y,sum(z)/len(z),cmap='viridis')
    
    plt.title(r'$\lambda=$'+str(realLambda)+r'  $\beta=$'+str(realBeta))
    plt.grid()
    plt.show()
   




if __name__ == "__main__":
    main()
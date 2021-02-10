#! /usr/bin/python3
#import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import cvxpy as cp
import math
import matplotlib.pyplot as plt
from numpy.linalg import norm
from mpl_toolkits import mplot3d
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import matplotlib
from random import shuffle

class Algorithm:
    def __init__(self):
        self.count=0
        self.weightvector = []
        self.limits = {}
        self.lam = 0
        self.bet = 0
        self.Min=0
        self.Max=99999


    def readCSV(self,filename):
        f = open(filename)
        content = f.read()
        dataset = [0]*len(content.split("\n"))
        # dataset = [0]*10
        i=0
        for line in content.split("\n"):
            line_split = line.split("],")
            lis = []
            if(line == ''):
                continue
            for rw in line_split:
                rw = rw.replace('[','')
                rw = rw.replace(']','')
                lis.append([float(x) for x in rw.split(",")])
                #print(lis)
            dataset[i] = lis
            i+=1
        return dataset

    def preparedata(self,data,weightVector):
        # print(data)
        taskList = []
        for task in data:
            utility = 0
            for attrNum in range(len(task)):
                utility += float(weightVector[attrNum] * task[attrNum])
            taskList.append(utility)
        
        return taskList

    def calculatestoppingtime(self,test,lam,bet,initial_weights):
        # print("test",test,initial_weights)
        isSatisfied = False
        dummyrewards = [[0]*2]*100
        for i in range(100):
            for j in range(2):
                dummyrewards[i][j] = np.random.gamma(2,scale=4)
        
        test = test+dummyrewards
        # print("length of test",len(test))
        # print('dummyrewards ',test[5][1])
        st = 0
        sum=0
        while not isSatisfied:
            # print(st)
            sum += ((initial_weights[0]*test[st][0])+(initial_weights[1]*test[st][1])) 
            if sum >= (bet**st) * lam or st >= len(test)-1:
                isSatisfied = True
            else:
                st+=1
            
        return st 


    
    def predictLambdaBeta(self,data):
        Lambda = 0
        Beta = 0
        # print("****",data)
        for i in range(len(data)):
            if i == len(data) - 1:
                #max limits
                if i in list(self.limits.keys()):
                    if self.limits[i][1] > sum(data[:i+1]):
                        self.limits[i] = (self.limits[i][0],sum(data[:i+1]))
                else:
                    self.limits[i] = (self.Min, sum(data[:i+1]))
            else:
                if i in list(self.limits.keys()):
                    if self.limits[i][0] < sum(data[:i+1]):
                        self.limits[i] = (sum(data[:i+1]),self.limits[i][1])              
                else:
                    self.limits[i] = (sum(data[:i+1]),self.Max)
        
        # print(self.limits)
        difference = {}
        average = {}
        for i in range(len(self.limits)):
            difference[i] = self.limits[i][1] - self.limits[i][0]
            average[i] = (self.limits[i][0]  + self.limits[i][1])/2
        # print(difference)
        # print("\n")
        # print(average)

        sorted_difference_Keys = sorted(difference, key=(lambda key:difference[key]))
        # print(sorted_difference_Keys)

        low_key = sorted_difference_Keys[0]
        high_key = sorted_difference_Keys[1]
        # print(low_key,high_key,average[low_key],average[high_key])

        if low_key > high_key:
            Beta = (average[high_key]/average[low_key])**(1/(high_key - low_key))
        else:
            Beta = (average[low_key]/average[high_key])**(1/(low_key - high_key))
        
        Beta = 1 if Beta > 1 else Beta
        
        Lambda = average[low_key]/(Beta ** low_key)

        return Lambda,Beta



    def predictWeightMatrix(self,d,lam,bet,initial_weights):
        sum = [0]*2
        A = []
        B = []
        # print(d)
        for i in range(len(d)):
            dummA = []
            for j in range(2):
                sum[j] +=  d[i][j]
                dummA.append(sum[j])
            if len(d)-1 == i:
                dummA = dummA + [-1 if k==i else 0 for k in range(len(d))]
            else:
                dummA = dummA + [1 if k==i else 0 for k in range(len(d))]
            A.append(dummA)
            B.append([(bet**i)*lam])

        A = np.array(A)
        B = np.array(B)
        # print(A)
        B = np.reshape(B,(B.shape[0],))
        # print(A.shape,B.shape)
        X = cp.Variable(2+len(d))

        weight_contraint_matrix = [1,1]+[0 for i in range(len(d))]
        constraints = [0 <= X , 1 == weight_contraint_matrix@X]
        objective = cp.Minimize(cp.norm((A@X)-B,2))
        prob = cp.Problem(objective,constraints)
        try:
            prob.solve()
            alpha = 0.7
            weight_1 = X.value[0]
            weight_2 = X.value[1]
            # weight_1 = alpha*initial_weights[0] + (1-alpha)*weight_1
            # weight_2 = alpha*initial_weights[1] + (1-alpha)*weight_2
            return [weight_1,weight_2]
        except:
            return initial_weights


    def predModelParameters(self,dataset,test,true_lambda,true_beta,true_weights,initial_weights):
        st_error = {}
        lambda_error = {}
        beta_error = {}
        weights_error = {}
        lam = 0
        bet = 0
        prev_lam = 0
        prev_bet = 0
        i=0
        prev_weight = initial_weights
        # print(len(dataset))
        for d in dataset:
            # print(d)
            epsilon = 0.5
            regret = 1
            cnt = 0
            while regret > epsilon and cnt <= 30:
                prep_data = self.preparedata(d,initial_weights)
                lam,bet = self.predictLambdaBeta(prep_data)
                print("iter-{},lam-{},bet-{}".format(cnt,lam,bet))
                initial_weights = self.predictWeightMatrix(d,lam,bet,initial_weights)

                lam_regret = (true_lambda-lam)**2/true_lambda**2 
                bet_regret = (true_beta-bet)**2/true_beta**2
                weight_regret = norm(np.array(true_weights)-np.array(initial_weights),2)/norm(np.array(true_weights),2)#/math.sqrt(true_weights[0]**2 + true_weights[1]**2)
                # weight_regret = ((true_weights[0]-initial_weights[0]/true_weights[0])**2)+((true_weights[1]-initial_weights[1]/true_weights[1])**2)
                regret = lam_regret+bet_regret+weight_regret
                print("regret:",regret)
                cnt += 1
            alpha = 0.1
            alpha1 = 0.3
            alpha2=0.7
            # lam = prev_lam + alpha*(lam-prev_lam)
            # bet = prev_bet + (alpha1)*(bet-prev_bet)
            # initial_weights = [alpha2*prev_weight[0]+(1-alpha2)*initial_weights[0], alpha2*prev_weight[1]+(1-alpha2)*prev_weight[1]]

            prev_lam = lam
            prev_bet = bet
            prev_weight = initial_weights

            # print("weights-{}".format(initial_weights))
            # print("-------------------------------------------------------")

            st = []
            for j in range(len(test)):
                s = self.calculatestoppingtime(test[j],lam,bet,initial_weights)
                error  = ((len(test[j]) - s )**2)/(len(test[j])**2)
                st.append(error)
            stime = sum(st)/len(st)

            lambda_error[i] = (true_lambda-lam)**2/(true_lambda)**2
            beta_error[i] = (true_beta-bet)**2/(true_beta)**2
            st_error[i] = stime
            weights_error[i] = norm(np.array(true_weights)-np.array(initial_weights),2)/norm(np.array(true_weights),2)#/math.sqrt(true_weights[0]**2 + true_weights[1]**2)
            # weights_error[i]= (((true_weights[0]-initial_weights[0])/true_weights[0])**2)+(((true_weights[1]-initial_weights[1])/true_weights[1])**2)
            i+=1

            print("lam-{},true_lambda-{},error-{}".format(lam,true_lambda,lambda_error))

        fig,ax = plt.subplots()
        # axins = inset_axes(ax, 2,2, loc=1)
        ax.plot(list(st_error.keys()),list(st_error.values()),'b--*',fillstyle='none',label=r'$E((t^*-\hat{t^*})^2/(t^*)^2)$')
        ax.plot(list(lambda_error.keys()),list(lambda_error.values()),'g-^',fillstyle='none',label=r'$(\lambda-\hat{\lambda})^2/\lambda^2$')
        ax.plot(list(beta_error.keys()),list(beta_error.values()),'r-.o',fillstyle='none',label=r'$(\beta-\hat{\beta})^2/\beta^2$')
        ax.plot(list(weights_error.keys()),list(weights_error.values()),'k:d',fillstyle='none',label=r'$\vert|{\alpha}-\hat{{\alpha}}\vert|_2^2$')
        # axins.plot(list(st_error.keys()),list(st_error.values()), 'b--*',fillstyle='none')
        # axins.plot(list(lambda_error.keys()),list(lambda_error.values()), 'g-^',fillstyle='none')
        # axins.plot(list(beta_error.keys()),list(beta_error.values()),'r-.o',fillstyle='none')
        # axins.plot(list(weights_error.keys()),list(weights_error.values()),'y:p',fillstyle='none')

        # x1, x2, y1, y2 = 0, 30, 0,1 # specify the limits
        # axins.set_xlim(x1, x2) # apply the x-limits
        # axins.set_ylim(y1, y2) # apply the y-limits
        # mark_inset(ax, axins, loc1=3, loc2=4, fc="none", ec="0.5") 

        ax.set_ylim(0,1)
        # axins.grid()
        # ax.set_ylim(0,1)
        ax.legend(frameon=True, loc='upper right',ncol=1)
        ax.set_xlabel('Number of Iterations')
        ax.set_ylabel('Error')

        ax.set(title=r'$\lambda=$'+str(true_lambda)+r' $\beta=$'+str(true_beta)+r" $\alpha_1=$"+str(true_weights[0])+r' $\alpha_2= $'+str(true_weights[1]))
        ax.grid()
        plt.show()

        
        return lam,bet,initial_weights
    

def main():
    dataset = 'gammadataset_100_0_85_0_15_0_85.csv'
    true_lambda = 100
    true_beta  = 0.85
    true_weights = [0.15,0.85]


    initial_weights = [0.5,0.5]

    algoObj = Algorithm()
    dataset = algoObj.readCSV(dataset)
    shuffle(dataset)
    # print(dataset)
    train = dataset[:50]
    test = dataset[990:]

    pred_lambda,pred_beta,pred_weight = algoObj.predModelParameters(train,test,true_lambda,true_beta,true_weights,initial_weights)
    print(pred_lambda,pred_beta,pred_weight)



if __name__ == '__main__':
    main()
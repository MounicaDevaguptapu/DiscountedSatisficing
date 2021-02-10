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
from predictStoppingTime import predictStoppingTime


class Algorithm:
    def __init__(self):
        self.limits = {}
        self.count = 0

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

    def preparedata(self,dataset,weightMatrix):
        #print("length {}".format(len(dataset)))
        final_ds = []*len(dataset)
        for dp in dataset:
            lis = []*len(dp)
            for task in dp:
                s=0
                for i in range(len(task)):
                    s += weightMatrix[i]*task[i]
                self.count+= 1
                lis.append(s)
            final_ds.append(lis)
        return final_ds
   
    def BLLOW(self,data,lamb,beta,weightMatrix):

        A_lis = []
        b_lis = []
        count = 0
        for dp in data:
            sum = {}
            for i in range(len(dp)):
                noise = [0]*(self.count)
                noise[count] = 1 if(i!= len(dp)-1) else -1
                for j in range(len(dp[i])):
                    sum[j] = sum[j]+dp[i][j] if(j in sum.keys()) else dp[i][j]
                reward = [float(x) for x in sum.values()]
                modelparam = (beta**i)*lamb
                A_lis.append(reward + noise)
                b_lis.append(modelparam)
                count+=1
        
        A = np.array(A_lis)
        b = np.array(b_lis)
        X = cp.Variable(len(weightMatrix)+self.count)
        
        constraint_dummy_matrix = [0 for k in range(self.count)]
        weightsConstraintMatrix = [1,1]+constraint_dummy_matrix

        objective = cp.Minimize(cp.norm((A@X)-b,2))
        constraints = [0 <= X , 1 == weightsConstraintMatrix @ X]
        prob = cp.Problem(objective,constraints)
        try:
            prob.solve()
            return [X.value[0],X.value[1]]
        except Exception as e:
            print("while weights.."+str(e))
            return weightMatrix
    
    
    def generateAmatrix(self,data,num_of_noise_variables):
        mat  = []
        count = 0
        for i in range(len(data)):
            for j in range(len(data[i])):
                noise = [0]*num_of_noise_variables
                if j == len(data[i])-1:
                    noise[count+j] = 1
                else:
                    noise[count+j] = -1
                row = [1,j]+noise
                mat.append(row)
            count += len(data[i])
        return mat

    def generatebVector(self,data):
        vec = []
        for i in range(len(data)):
            # print("****data***",data[i])
            for j in range(len(data[i])):
                vec.append([math.log(sum(data[i][:j+1]),2) ])
        return vec

    def BLLO(self,train):
        num_of_noise_variables = 0
        for i in range(len(train)):
            num_of_noise_variables += len(train[i])
        
        A = self.generateAmatrix(train,num_of_noise_variables)
        A = np.array(A)
        # print("A:",A)
        X = cp.Variable(2+num_of_noise_variables)
        b = self.generatebVector(train)
        b = np.array(b)
        b = np.reshape(b,(b.shape[0],))
        # print("b:",b)
        # print("shapes: ",A.shape,X.shape,b.shape)

        beta_limit_matrix = [0,1]+[0 for i in range(num_of_noise_variables)]
        lambda_limit_matrix = [1,0]+[0 for i in range(num_of_noise_variables)]
        noise_limit_matrix = [[1 if i==j and i not in [0,1] else 0 for j in range(num_of_noise_variables+2)] for i in range(num_of_noise_variables+2)]
        constraints = [0 >= beta_limit_matrix@X, 0 <= lambda_limit_matrix@X, [0 for i in range(num_of_noise_variables+2)] <= noise_limit_matrix@X]
        objective = cp.Minimize(cp.norm(((A@X)-b),2))
        prob = cp.Problem(objective,constraints)
        try:
            prob.solve()
            pred_lam = 2**X.value[0]
            pred_bet = 2**X.value[1]
            print(pred_lam,pred_bet)
            pred_bet = 1 if pred_bet>1 else pred_bet
        except Exception as e:
            print(str(e))

        return pred_lam,pred_bet
    
    def calculatestoppingtime(self,test,lam,bet,initial_weights):
        isSatisfied = False
        dummyrewards = [[0]*2]*100
        for i in range(100):
            for j in range(2):
                dummyrewards[i][j] = np.random.gamma(2,scale=4)
        
        test = test+dummyrewards
        st = 0
        sum=0
        while not isSatisfied:
            sum += ((initial_weights[0]*test[st][0])+(initial_weights[1]*test[st][1])) 
            if sum >= (bet**st) * lam or st >= len(test)-1:
                isSatisfied = True
            else:
                st+=1    
        return st 
    
    def error(self,data,iter_lambda,iter_beta,iter_weight):
        st = []
        stn=[]
        for j in range(len(data)):
            s = self.calculatestoppingtime(data[j],iter_lambda,iter_beta,iter_weight)
            error  = ((len(data[j]) - s )**2)
            error1 = ((len(data[j]) - s )**2)/(len(data[j])**2)
            st.append(error)
            stn.append(error1)
        stime = sum(st)/len(st)
        sntime = sum(stn)/len(stn)
        return stime**(1/2),sntime

    def BLAT(self,train,true_lambda,true_beta,true_weights,initial_weights):
        # pst = predictStoppingTime()
        allowed_error = 0.005
        iter_error = 1
        iter_num = 0
        iter_weight = initial_weights
        iter_lambda = 0
        iter_beta = 0
        st_error,beta_error,lambda_error,weight_error = {},{},{},{}

        while(iter_error > allowed_error and iter_num <= 30):
            prep_data = self.preparedata(train,iter_weight)
            iter_lambda,iter_beta = self.BLLO(prep_data)
            iter_weight = self.BLLOW(train,iter_lambda,iter_beta,iter_weight)
            iter_error,itern_error = self.error(train,iter_lambda,iter_beta,iter_weight)
            print("iter error = ",iter_error)

            st_error[iter_num] = itern_error
            lambda_error[iter_num] = (1 - ((1*iter_lambda)/true_lambda))**2
            beta_error[iter_num] = (1 - ((1*iter_beta)/true_beta))**2
            weight_error[iter_num] = norm(np.array(true_weights)-np.array(iter_weight),2)

            iter_num += 1
        
        fig,ax = plt.subplots()
        axins = inset_axes(ax, 2,2, loc=1)
        ax.plot(list(st_error.keys()),list(st_error.values()),'b--*',fillstyle='none',label=r'$E((t^*-\hat{t^*})^2/(t^*)^2)$')
        ax.plot(list(lambda_error.keys()),list(lambda_error.values()),'g-^',fillstyle='none',label=r'$(\lambda-\hat{\lambda})^2$')
        ax.plot(list(beta_error.keys()),list(beta_error.values()),'r-.o',fillstyle='none',label=r'$(\beta-\hat{\beta})^2$')
        ax.plot(list(weight_error.keys()),list(weight_error.values()),'k:d',fillstyle='none',label=r'$\vert|{\alpha}-\hat{{\alpha}}\vert|_2^2$')
        axins.plot(list(st_error.keys()),list(st_error.values()), 'b--*',fillstyle='none')
        axins.plot(list(lambda_error.keys()),list(lambda_error.values()), 'g-^',fillstyle='none')
        axins.plot(list(beta_error.keys()),list(beta_error.values()),'r-.o',fillstyle='none')
        axins.plot(list(weight_error.keys()),list(weight_error.values()),'k:d',fillstyle='none')
        x1, x2, y1, y2 = 0, 30, 0,0.05 # specify the limits
        axins.set_xlim(x1, x2) # apply the x-limits
        axins.set_ylim(y1, y2) # apply the y-limits
        mark_inset(ax, axins, loc1=3, loc2=4, fc="none", ec="0.5") 
        axins.grid()


        ax.title.set_text(r'$\lambda=$'+str(true_lambda)+r'$ ;\beta=$'+str(true_beta)+r'$ ;\alpha_1=$'+str(true_weights[0])+r'$ ;\alpha_2=$'+str(true_weights[1]))


        ax.grid()
        # plt.ylim(0,2)
        ax.legend(frameon=True, loc='center left',ncol=1)
        ax.set_xlabel('Number of Iterations')
        ax.set_ylabel('Error')
        plt.show()

        return iter_lambda,iter_beta,iter_weight

def main():
    alg = Algorithm()
    dataset = 'gammadataset_100_0_85_0_15_0_85.csv'

    true_lambda = 100
    true_beta = 0.85
    true_weights = [0.15,0.85]
    initial_weights = [0.5,0.5]

    data = alg.readCSV(dataset)
    shuffle(data)
    train = data[:200]
    test = data[950:]

    p_lambda,p_beta,p_weights = alg.BLAT(train,true_lambda,true_beta,true_weights,initial_weights)
    print("Final Prediction-Lambda-{},beta-{},weights-{}".format(p_lambda,p_beta,p_weights))

    test_rmse = alg.error(test,p_lambda,p_beta,p_weights)
    print("Test RMSE - ",test_rmse)

if __name__=='__main__':
    main()
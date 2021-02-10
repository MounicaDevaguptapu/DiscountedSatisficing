import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import math
from predictStoppingTime import predictStoppingTime
from random import shuffle

class BLKWLO():
    def __init__(self):
        pass

    def readDataset(self,fileName):
        fileContent = []
        with open(fileName,'r') as f:
            Content = f.read()
            Content = Content.split("\n")
            for day in Content:
                day = day.split("],")
                dayList = []
                for task in day:
                    task = task.replace("[","").replace("]","")
                    attr = [float(x) for x in task.split(",")]
                    dayList.append(attr)
                fileContent.append(dayList)

        return fileContent


    def main(self,train_df,test,true_weights,true_lambda,true_beta):
        dataset = 'gammadataset_100_0_35_0_15_0_85.csv'
        # true_weights = [0.15,0.85]
        # true_beta = 0.85
        # true_lambda = 100

        # fileContent = self.readDataset(dataset)
        lambda_error = {}
        beta_error = {}

        pst = predictStoppingTime()
        # preparedData = pst.prepareData(fileContent,true_weights)
        # # preparedData = preparedData[:3]
        # shuffle(preparedData)
        # train_df = preparedData[:30]
        # # print(train)
        # test = preparedData[993:]
        error = {}
        for i in range(len(train_df)):
            train = train_df[:i+1]
            num_of_noise_variables = 0
            for i in range(len(train)):
                # print(len(train[i]))
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
                prob.solve(solver=cp.ECOS)
                pred_lam = 2**X.value[0]
                pred_bet = 2**X.value[1]
                # print(pred_lam,pred_bet)
                pred_bet = 1 if pred_bet>1 else pred_bet
            except Exception as e:
                print(str(e))
            
            error[i] = pst.calculateError(train,pred_lam,pred_bet)
            lambda_error[i] = ((true_lambda - pred_lam)**2)/(true_lambda**2)
            beta_error[i] = ((true_beta - pred_bet)**2)/(true_beta**2)

        # marker_style = dict(color='tab:blue', linestyle='--', marker='o',
        #                 markersize=8, markerfacecoloralt='tab:red')   
        # plt.plot(list(error.keys()),list(error.values()),**marker_style,fillstyle='none',label=r'$E((t^*-\hat{t^*})^2/(t^*)^2)$')
        # # plt.plot(list(lambda_error.keys()),list(lambda_error.values()),'r-*',label=r'$(\lambda - \hat{\lambda})^2/\lambda^2$')
        # # plt.plot(list(beta_error.keys()),list(beta_error.values()),'b-o',label=r'$(\beta - \hat{\beta})^2/\beta^2$')
        # plt.legend()
        # plt.xlabel("Number of Iterations")
        # plt.ylabel("WST Normalized Average Error")
        # plt.xticks(np.arange(1,len(train)+1,2))
        # plt.xlim(1,len(train))
        # plt.grid()
        # plt.show()
        test_rmse = pst.calculateError(test,pred_lam,pred_bet)
        return error,lambda_error,beta_error,test_rmse


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


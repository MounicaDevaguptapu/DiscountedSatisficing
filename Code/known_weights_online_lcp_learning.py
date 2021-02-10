import numpy as np
import cvxpy as cp
import math
import matplotlib.pyplot as plt
from predictStoppingTime import predictStoppingTime

class KWOLLO():
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



    def main(self,train,test,true_weights,true_lambda,true_beta):
        # dataset = 'dataset_100_0_95_0_15_0_85.csv'
        dataset = 'gammadataset_100_0_35_0_15_0_85.csv'
        # true_weights = [0.15,0.85]
        # true_beta = 0.85
        # true_lambda = 100
        pred_lam = 1
        pred_bet = 1

        lam_error = {}
        beta_error = {}
        fileContent = self.readDataset(dataset)

        pst = predictStoppingTime()
        # preparedData = pst.prepareData(fileContent,true_weights)
        # # preparedData = preparedData[:3]
        # train = preparedData[:100]
        # # print("train",train)
        # test = preparedData[900:]
        error = {}
        for i in range(len(train)):
            # train = train_df[:i+1]
            num_of_noise_variables = len(train[i])
            # for i in range(len(train[i])):
            #     # print(len(train[i]))
            #     num_of_noise_variables += len(train[i])
            
            A = self.generateAmatrix(train[i],num_of_noise_variables)
            A = np.array(A)
            # print("A:",A)
            X = cp.Variable(2+num_of_noise_variables)
            b = self.generatebVector(train[i])
            b = np.array(b)
            b = np.reshape(b,(b.shape[0],))
            # print("b:",b)
            # print("sahpes: ",A.shape,X.shape,b.shape)

            beta_limit_matrix = [0,1]+[0 for i in range(num_of_noise_variables)]
            lambda_limit_matrix = [1,0]+[0 for i in range(num_of_noise_variables)]
            constraint_matrix = [[1 if i==j and i!=1 else 0 for j in range(num_of_noise_variables+2)] for i in range(num_of_noise_variables+2)]
            #print(constraint_matrix)
            constraints = [0 >= beta_limit_matrix@X, [0 for i in range(num_of_noise_variables+2)] <= constraint_matrix@X]
            if i != 0:
                objective = cp.Minimize(cp.norm(((A@X)-b),2)+1000*(cp.square(beta_limit_matrix@X - pred_bet)))
            else:
                objective = cp.Minimize(cp.norm(((A@X)-b),2))
            prob = cp.Problem(objective,constraints)
            try:
                prob.solve(solver=cp.ECOS)
                alpha = 0.7
                pred_lam_step = 2**X.value[0]
                pred_bet_step = 2**X.value[1]
                pred_lam = alpha*pred_lam + (1-alpha)*pred_lam_step if(i!=0) else pred_lam_step
                pred_bet = alpha*pred_bet + (1-alpha)*pred_bet_step if(i!=0) else pred_bet_step
                pred_bet = 1 if pred_bet > 1 else pred_bet
            except Exception as e:
                print(str(e))
            
            lam_error[i] = (true_lambda - pred_lam)**2/true_lambda**2
            beta_error[i] = (true_beta - pred_bet)**2/true_beta**2
            error[i] = pst.calculateError([train[i]],pred_lam,pred_bet)

        # print(error)    
        # plt.plot(list(error.keys()),list(error.values()),label=r'$E((T-\hat{T})^2/T^2)$')
        # plt.plot(list(lam_error.keys()),list(lam_error.values()),'r-*',label='Lambda Error')
        # plt.plot(list(beta_error.keys()),list(beta_error.values()),'g--^',label='Beta Error')
        # plt.legend()
        # #plt.ylim(0,1)
        # plt.xlabel("Number of Simulations")
        # plt.ylabel("WST Normalized Average Error")
        # plt.grid()
        # plt.show()
        # print(pred_lam,pred_bet)
        test_rmse = pst.calculateError(test,pred_lam,pred_bet)
        return error,lam_error,beta_error,test_rmse


    def generateAmatrix(self,data,num_of_noise_variables):
        mat  = []
        count = 0
        for i in range(len(data)):
            # for j in range(len(data[i])):
            noise = [0]*num_of_noise_variables
            if i == len(data)-1:
                noise[i] = 1
            else:
                noise[i] = -1
            row = [1,i]+noise
            mat.append(row)
            # count += len(data)
        return mat

    def generatebVector(self,data):
        vec = []
        for i in range(len(data)):
            # print("****data***",data[i])
            # for j in range(len(data[i])):
            vec.append([math.log(sum(data[:i+1]),2) ])
        return vec


if __name__ == "__main__":
    main()
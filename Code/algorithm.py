#! /usr/bin/python3
#import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from numpy.linalg import norm
from mpl_toolkits import mplot3d
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

#log flag
LOG_FLAG = True
GREEN_FLAG = False
log_file = open("log_file.txt","w")
epsilon = 0.15
regret = 1

class Algorithm:
    def __init__(self):
        self.count = 0

    def readCSV(self,dataFile):
        #dataFile = 'dataset_100_0_95_0_63_0_37.csv'
        #dataFile = 'dataset_100_0_95_0_15_0_85.csv'
        f = open(dataFile)
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
        return dataset[:300]
        #return [[[1,2],[2,3],[3,4]],[[4,5],[5,6],[6,7],[8,9]]]

    #function to prepare multi attribute dataset into dataset with weighted utility values
    def prepareData(self,dataset,weightMatrix):
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

    def predictLambdaBeta(self,train_data):
        #print(train_data)
        LAMBDA_MAP = {}
        BETA_MAP = {}
        limits_map = {}
        count=0
        differences_limits_map = {}
        averages_limits_map = {}
        LAMBDA,BETA = 0,0
        MAX = 999999
        MIN = 0
        for line in train_data:
            #print("line",line)
            if(line==''): continue
            rewards = [line[i-1] for i in range(1,len(line)+1)]
            log_file.write("---------------------------------\n") if(LOG_FLAG) else print("---------------------------------")
            log_file.write("DataPoint:\n") if(LOG_FLAG) else print("DataPoint:")
            log_file.write("{}\n".format(rewards)) if(LOG_FLAG) else print("{}".format(rewards))
            for i in range(1,len(rewards)+1):
                if(i!= len(rewards)):
                    if(i in limits_map.keys()):
                        if(limits_map[i][0] < sum(rewards[:i])):
                            limits_map[i][0] = sum(rewards[:i])
                    else:
                        limits_map[i] = [sum(rewards[:i]),MAX]
                else:
                    if(i in limits_map.keys()):
                        if(limits_map[i][1] > sum(rewards[:i])):
                            limits_map[i][1] = sum(rewards[:i])
                    else:
                        limits_map[i] = [0,sum(rewards[:i])]
        
            log_file.write("After {}th datapoint \n".format(count)) if(LOG_FLAG) else print("After {}th datapoint ".format(count))
            log_file.write("{}\n".format(limits_map)) if(LOG_FLAG) else print("{}".format(limits_map))
            count += 1

            log_file.write("---------------------------------\n") if(LOG_FLAG) else print("---------------------------------")
            log_file.write("{}\n".format(limits_map)) if(LOG_FLAG) else print("{}".format(limits_map))


            for i in range(1,len(limits_map)+1):
                differences_limits_map[i] = abs(limits_map[i][1] - limits_map[i][0])
                averages_limits_map[i] = (limits_map[i][1] + limits_map[i][0])/2

            log_file.write("---------------------------------\n") if(LOG_FLAG) else print("---------------------------------")
            log_file.write("Differenece array {}\n".format(differences_limits_map)) if(LOG_FLAG) else print("Differenece array {}\n".format(differences_limits_map))
            
            sorted_differences = sorted(differences_limits_map, key=(lambda key:differences_limits_map[key]))
            log_file.write("---------------------------------\n") if(LOG_FLAG) else print("---------------------------------")
            log_file.write("Sorted Array of index with minimum difference in the limits {}\n".format(sorted_differences)) if(LOG_FLAG) else print("Sorted Array of index with minimum difference in the limits {}".format(sorted_differences))

            log_file.write("---------------------------------\n") if(LOG_FLAG) else print("---------------------------------")
            log_file.write("Average Limits HashMap {}\n".format(averages_limits_map)) if(LOG_FLAG) else print("Average Limits HashMap {}".format(averages_limits_map))

            BETA = (averages_limits_map[sorted_differences[1]]/averages_limits_map[sorted_differences[0]])**(1/(sorted_differences[1]-sorted_differences[0])) if(sorted_differences[1]>sorted_differences[0]) else (averages_limits_map[sorted_differences[0]]/averages_limits_map[sorted_differences[1]])**(1/(sorted_differences[0]-sorted_differences[1]))
            log_file.write("---------------------------------\n") if(LOG_FLAG) else print("---------------------------------")
            log_file.write("BETA: {}\n".format(BETA)) if(LOG_FLAG) else print("BETA: {}".format(BETA))
            BETA = 1 if(BETA > 1) else BETA
            
            if sorted_differences[0] == 1:
                LAMBDA = averages_limits_map[sorted_differences[0]]
            else:
                LAMBDA = averages_limits_map[sorted_differences[0]] / (BETA ** (sorted_differences[0] -1))
            LAMBDA_MAP[count] = LAMBDA
            BETA_MAP[count] = BETA


        # plt.plot(LAMBDA_MAP.keys(),LAMBDA_MAP.values(),BETA_MAP.keys(),BETA_MAP.values())
        # plt.grid()
        # plt.show()
        
        return LAMBDA,BETA

    #function to predict weights using LCP method
    def predictWeightMatrix(self,data,lamb,beta,weightMatrix):
        #data = [[[1,2],[2,3],[3,4]],[[4,5],[5,6],[6,7],[8,9]]]
        # lamb = 100
        # beta = 0.99
        log_file.write("Predicting Weights using LCP Method\n") if(LOG_FLAG) else print("Predicting Weights using LCP Method")
        A_lis = []
        b_lis = []
        count = 0
        for dp in data:
            sum = {}
            for i in range(len(dp)):
                noise = [0]*(self.count)
                #noise = [0 if(j!= i) else 1 if(i!=len(dp)-1) else -1 for j in range(len(dp))]
                noise[count] = 1 if(i!= len(dp)-1) else -1
                for j in range(len(dp[i])):
                    sum[j] = sum[j]+dp[i][j] if(j in sum.keys()) else dp[i][j]
                reward = [float(x) for x in sum.values()]
                modelparam = (beta**i)*lamb
                A_lis.append(reward + noise)
                b_lis.append(modelparam)
                count+=1
        
        #matrix A in Ax-b
        A = np.array(A_lis)
        log_file.write("Shape of matrix A in A@X-b is - {}".format(A.shape)) if(LOG_FLAG) else print("Shape of matrix A in A@X-b is - {}".format(A.shape))
        #matrix b in Ax-b
        b = np.array(b_lis)
        log_file.write("Shape of matrix b in A@X-b is - {}".format(b.shape)) if(LOG_FLAG) else print("Shape of matrix B in A@X-b is - {}".format(b.shape))
        #print(b)
        #print(A)
        #matrix x in Ax-b
        X = cp.Variable(len(weightMatrix)+self.count)
        log_file.write("Shape of matrix X in A@X-b is - {}".format(X.shape)) if(LOG_FLAG) else print("Shape of matrix X in A@X-b is - {}".format(X.shape))
        # print("length of x --",len(X))
        
        constraint_dummy_matrix = [0 for k in range(self.count)]
        weightsConstraintMatrix = [1,1]+constraint_dummy_matrix
        
        
        objective = cp.Minimize(cp.norm((A@X)-b,2))
        constraints = [0 <= X , 1 == weightsConstraintMatrix @ X]
        prob = cp.Problem(objective,constraints)
        try:
            prob.solve()
            return [X.value[0],X.value[1]]
        except:
            return weightMatrix
        #solver = cp.ECOS

    def calculatestoppingtime(self,test,lam,bet,initial_weights):
        # print("test",test,initial_weights)
        isSatisfied = False
        dummyrewards = [[0]*2]*100
        for i in range(100):
            for j in range(2):
                dummyrewards[i][j] = np.random.gamma(2,scale=4)
        
        test = test+dummyrewards
        # print('dummyrewards ',test[5][1])
        st = 0
        sum=0
        while not isSatisfied:
            sum += ((initial_weights[0]*test[st][0])+(initial_weights[1]*test[st][1])) 
            if sum >= (bet**st) * lam:
                isSatisfied = True
            else:
                st+=1
            
        return st


    def stoppingtime(self,test,lamda,beta,weightMatrix):
        st = []
        for i in range(len(test)):
            s = self.calculatestoppingtime(test[i],lamda,beta,weightMatrix)
            error  = ((len(test[i]) - s )**2)/(len(test[i])**2)
            st.append(error)

        return sum(st)/len(st)



    #funciton to predict lambda and beta using limits algorithm
    def predModelParameters(self,dataset,true_lambda,true_beta,true_weight,weightMatrix):
        global regret
        ##lamb,beta = 0,0
        lambda_error = {}
        beta_error = {}
        weight_error = {}
        st_error = {}
        # regret = {}
        flag = True
        i=0
        true_weightVector = np.array(true_weight)
        test = dataset[290:]
        train=dataset[:200]
        while regret > epsilon:
            
            # train = dataset[:i+1]
            # for j in range(5):
            data = []
            #dt = dataset[:i+1]
            #print("-----------------")
            lamb,beta = 0,0
            log_file.write("Trial Number - {}".format(i)) if(GREEN_FLAG) else print("Trial Number - {}".format(i))
            data = self.prepareData(train,weightMatrix)
            lamb,beta = self.predictLambdaBeta(data)
            log_file.write("Predicted Lambda and Beta using Limits method - {} and {}".format(lamb,beta)) if(GREEN_FLAG) else print("Predicted Lambda and Beta using Limits method - {} and {}".format(lamb,beta))
            
            
            
            weightMatrix = self.predictWeightMatrix(train,lamb,beta,weightMatrix)
            log_file.write("Predicted Weights using CVXPY -- {}".format(weightMatrix)) if(GREEN_FLAG) else print("Predicted Weights using LCP method -- {}".format(weightMatrix))
            lambda_error[i] = (true_lambda-lamb)**2/true_lambda**2
            beta_error[i] = (true_beta-beta)**2
            weightVector = np.array(weightMatrix)
            weight_error[i] = norm((weightVector - true_weightVector),2)/norm(true_weightVector,2)
            print(lambda_error[i],beta_error[i],weight_error[i])
            st_error[i] = self.stoppingtime(test,lamb,beta,weightMatrix)
            regret = 0.33*weight_error[i] + 0.33*beta_error[i] + 0.33*lambda_error[i]
            print(regret)
            # flag = False if(regret[i] <= epsilon) else True
            i+=1
        
        fig,ax = plt.subplots()
        axins = inset_axes(ax, 2,2, loc=1)
        ax.plot(list(lambda_error.keys()),list(lambda_error.values()),'g-*',fillstyle='none',label=r'$(\lambda-\hat{\lambda})^2$')
        ax.plot(list(beta_error.keys()),list(beta_error.values()),'r-.^',fillstyle='none',label=r'$(\beta-\hat{\beta})^2$')
        ax.plot(list(weight_error.keys()),list(weight_error.values()),'c:p',fillstyle='none',label=r'$\vert|\alpha-\hat{\alpha}\vert|_2^2$')
        ax.plot(list(st_error.keys()),list(st_error.values()),'b--o',fillstyle='none',label=r'$E((t^*-\hat{t^*})^2/(t^*)2)$')

        axins.plot(list(st_error.keys()),list(st_error.values()), 'b--o',fillstyle='none')
        axins.plot(list(lambda_error.keys()),list(lambda_error.values()), 'g-*',fillstyle='none')
        x1, x2, y1, y2 = 0, 30, 0,1 # specify the limits
        axins.set_xlim(x1, x2) # apply the x-limits
        axins.set_ylim(y1, y2) # apply the y-limits
        mark_inset(ax, axins, loc1=3, loc2=4, fc="none", ec="0.5") 

        # plt.ylim(0,10)
        ax.legend(frameon=True, loc='upper left',ncol=1)
        ax.set_xlabel('Number of Iterations')
        ax.set_ylabel('Error')
        plt.show()
        
        return lamb,beta,weightMatrix
            
        

def main():
    dataFiles = ['gammadataset_100_0_35_0_15_0_85.csv']#,'dataset_100_0_95_0_63_0_37.csv','dataset_100_0_95_0_5_0_5.csv']
    lambda_error = []
    beta_error = []
    weight_error = []
    for dataFile in dataFiles:
        filename = dataFile.replace('.csv','')
        filename_split = filename.split("_")
        TRUE_LAMBDA = float(filename_split[1])
        TRUE_BETA = float(filename_split[2]+"."+filename_split[3])
        TRUE_WEIGHT = [float(filename_split[4]+"."+filename_split[5]),float(filename_split[6]+"."+filename_split[7])]
        
        algoObj = Algorithm()
        dataset = algoObj.readCSV(dataFile)
        log_file.write("Dataset---{}".format(str(dataset))) if(LOG_FLAG) else print("Dataset---{}".format(str(dataset)))
        intial_weights = [1/len(dataset[0][0]) for i in range(len(dataset[0][0]))]
        pred_lambda,pred_beta,pred_weight = algoObj.predModelParameters(dataset,TRUE_LAMBDA,TRUE_BETA,TRUE_WEIGHT,intial_weights)
        
        print(" True Lambda - {} \n True Beta - {} \n True weights - {} \n Predicted lambda - {} \n Predicted Beta - {} \n Predicted Weights -- {} \n".format(TRUE_LAMBDA,TRUE_BETA,TRUE_WEIGHT,pred_lambda,pred_beta,pred_weight))
        
if __name__ == "__main__":
    main()

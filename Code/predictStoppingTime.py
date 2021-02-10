#################################################################################
# Alternative Minimization Algorithm to predict stopping time of crowd worker
# Heuristic: Discounted Satisficing
# Predict model parameters using limits approach
# Predict weights of the arrtibutes using linear optimization(Online Algorithm)
#
# Author: Mounica Devaguptapu(md2rt@mst.edu)
#################################################################################
import cvxpy as cp
import numpy as np


class predictStoppingTime:

    def __init__(self):
        self.limits = {}
        self.Max = 99999
        self.Min = 0
        

    def predictModelParameters(self,data):
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
    
    def generateNoiseVector(self,totalTasks,taskNum):
        if totalTasks == taskNum + 1:
            return [0 for i in range(totalTasks-1)]+[-1]
        else:
            return [0 if i == taskNum else 1 for i in range(totalTasks)]

    def predictWeightVector(self,data,Lambda,Beta,numofAttributes,weightVector):
        A = []
        b = []
        sum = [0 for i in range(len(data))]
        for taskNum in range(len(data)):
            noiseVector = self.generateNoiseVector(len(data),taskNum)
            row = []
            for attrNum in range(len(data[taskNum])):
                sum[attrNum]+= data[taskNum][attrNum]
                row.append(sum[attrNum])
            A.append(row+noiseVector)
            b.append(Lambda * (Beta ** 2))

        A = np.array(A)
        b = np.array(b)
        X = cp.Variable(numofAttributes + len(noiseVector))

        objective = cp.Minimize(cp.norm(((A@X)-b),2))

        constraintMatrix = [1 for i in range(numofAttributes)]+[0 for i in range(len(noiseVector))]

        constraints = [0 <= X , 1 == constraintMatrix @ X]
        prob = cp.Problem(objective,constraints)
        try:
            prob.solve()
            weightVector = X.value[:numofAttributes]
        except Exception as e:
            print(str(e))
        return weightVector

    def prepareData(self,data,weightVector):
        preparedData = []

        for day in data:
            taskList = []
            for task in day:
                utility = 0
                for attrNum in range(len(task)):
                    utility += float(weightVector[attrNum] * task[attrNum])
                taskList.append(utility)
            preparedData.append(taskList)
        
        return preparedData

    def calculateStoppingTime(self,datapoint,lam,bet):
        # print(lam,bet)
        dummyRewards = [np.random.uniform(0,35) for i in range(100)]
        isSatisfied = True
        datapoint = datapoint+dummyRewards
        i=0
        while isSatisfied:
            if sum(datapoint[:i+1]) >= (bet**i)*lam:
                return i
            i+=1
        return i

    def calculateError(self,testdata,lam,bet):
        error = []
        for i in range(len(testdata)):
            T = len(testdata[i])
            T_hat = self.calculateStoppingTime(testdata[i],lam,bet)
            error.append(((T-T_hat)**2)/(T**2))
        return (sum(error)/len(error))

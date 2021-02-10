# known weights without slack variables
import numpy as np
import cvxpy as cp
import math
import matplotlib.pyplot as plt
from predictStoppingTime import predictStoppingTime

def readDataset(fileName):
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

def main():
    dataset = 'dataset_40_0_85_0_15_0_85.csv'
    true_weights = [0.15,0.85]
    true_beta = 0.85
    true_lambda = 40
    pred_lam = 1
    pred_bet = 1

    fileContent = readDataset(dataset)
    pst = predictStoppingTime()
    preparedData = pst.prepareData(fileContent,true_weights)
    train = preparedData[:10]
    test = preparedData[900:]

    for i in range(len(train)):
        T = len(train[i])
        A = generateAmatrix(train[i])
        X = cp.Variable(2)
        # Y = cp.Variable()
        b = generatebVector(train[i])
        b = np.reshape(b,(b.shape[0],))

        print("Shapes: ",A.shape,b.shape)

        lambda_v = [1,0]
        beta_v = [0,1]
       
        objective = cp.Minimize(cp.sum_squares(T- pst.calculateStoppingTime(train[i],lambda_v@X, beta_v@X) ))
        constraints = [b >= A@X, 0 >= lambda_v@X, 0 <= beta_v@X]
        prob = cp.Problem(objective,constraints)
        try:
            prob.solve()
            pred_lam = 2**X.value[0]
            pred_bet = 2**X.value[1]
        except Exception as e:
            print(str(e))

        print("pred_lam-{},pred_bet-{}".format(pred_lam,pred_bet))    

def generateAmatrix(record):
    A = []
    for i in range(len(record)):
        if i != len(record):
            A.append([-1,-i])
        else:
            A.append([1,i])

    return np.array(A)

def generatebVector(record):
    b = []
    for i in range(len(record)):
        b.append([math.log(sum(record[:i+1]),2)])
    
    return np.array(b)


if __name__ == "__main__":
    main()

import os
import numpy as np
from numpy.linalg import norm
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

    datasetFiles = ['dataset_100_0_75_0_15_0_45_0_40.csv']
    # datasetFiles = ['dataset_40_0_85_0_15_0_85.csv']

    pst = predictStoppingTime()

    for dataFile in datasetFiles:
        # get data
        fileContent = readDataset(dataFile)
        dataFile = dataFile.replace(".csv","")
        numofAttributes = len(fileContent[0][0])

        # Initializations  
        LambdaError,BetaError,weightError = {},{},{}
        trueLambda = float(dataFile.split("_")[1])
        trueBeta = float(dataFile.split("_")[2]+"."+dataFile.split("_")[3])
        trueweightVector = []
        for i in range(4,4+(numofAttributes*2 - 1),2):
            print(i)
            weight = float(dataFile.split("_")[i] + "."+dataFile.split("_")[i+1])
            trueweightVector.append(weight)
        print(trueweightVector)
        trueweightVector = np.array(trueweightVector)
        weightVector = np.array([float(1/numofAttributes) for i in range(numofAttributes)])
        Lambda,Beta = 0,0
        #prepare data for limits method
        preparedData = pst.prepareData(fileContent,weightVector)
        # print(preparedData)

        preparedData = preparedData
        # print(preparedData)

        for dataPoints in range(len(preparedData)):
            print("----------------------\nDatapoint-{}\n".format(dataPoints))
            partofPreparedData = preparedData[dataPoints]
            partofActualData = fileContent[dataPoints]

            # Prediction
            Lambda,Beta = pst.predictModelParameters(partofPreparedData)
            print("Lambda,Beta - {} and {}".format(Lambda,Beta))
            weightVector = pst.predictWeightVector(partofActualData,Lambda,Beta,numofAttributes,weightVector)
            print("Weight Matrix - {} ".format(weightVector))
            print("----------------------")

            LambdaError[dataPoints] = (trueLambda - Lambda)**2
            BetaError[dataPoints] = (trueBeta - Beta)**2
            weightError[dataPoints] = norm((trueweightVector - weightVector),2)

        
        print("True Lambda - {} and True Beta - {} and True Weights - {}".format(trueLambda,trueBeta,trueweightVector))
        print("Predicted Lambda - {} and predicted Beta - {} and predicted weights - {}".format(Lambda,Beta,weightVector))

        plt.plot(list(LambdaError.keys()),list(LambdaError.values()),'r-*',label='Lambda')
        plt.plot(list(BetaError.keys()),list(BetaError.values()),'b--^',label='Beta')
        plt.plot(list(weightError.keys()),list(weightError.values()),'g-.',label='Weight')
        plt.ylim(0,1)

        plt.legend()
        plt.grid()
        plt.show()


if __name__ == "__main__":
    main()      
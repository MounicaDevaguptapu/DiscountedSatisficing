from predictStoppingTime import predictStoppingTime
import numpy as np
import matplotlib.pyplot as plt
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
    dataset = 'dataset_100_0_95_0_15_0_85.csv'
    true_weights = [0.15,0.85]
    true_beta = 0.95
    true_lambda = 100

    fileContent = readDataset(dataset)

    pst = predictStoppingTime()
    preparedData = pst.prepareData(fileContent,true_weights)
    preparedData = preparedData[:300]
    train = preparedData[250:]
    test = preparedData[251:]
    error = {}
    for i in range(len(train)):
        data = train[:i+1]
        lam = 0
        beta = 0
        for j in range(len(data)):
            lam,beta = pst.predictModelParameters(data[j])
        error[i] = pst.calculateError(test,lam,beta)
    
    plt.plot(list(error.keys()),list(error.values()),label=r'$E((T-\hat{T})^2/T^2)$')
    plt.legend()
    plt.xlabel("Number of Simulations")
    plt.ylabel("WST Normalized Average Error")
    plt.grid()
    plt.show()
        

if __name__ == '__main__':
    main()
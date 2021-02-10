from predictStoppingTime import predictStoppingTime
import matplotlib.pyplot as plt

def readDataset(fileName):
    fileContent = []
    with open(fileName,'r') as f:
        Content = f.read()
        Content = Content.split("\n")
        for day in Content:
            print(day.split(","))
            attr = [float(x) for x in day.split(",")]
            fileContent.append(attr)

    return fileContent


def main():
    dataset = 'dataset_100_0_35.csv'
    true_weights = [1]
    true_beta = 0.35
    true_lambda = 100

    fileContent = readDataset(dataset)
    print(fileContent)
    pst = predictStoppingTime()
    preparedData = fileContent
    #preparedData = pst.prepareData(fileContent,true_weights)
    # preparedData = preparedData[:500]
    train = preparedData[:100]
    # print(train)
    test = preparedData[980:]
    error = {}
    lambda_error = {}
    beta_error = {}
    for i in range(len(train)):
        lam,bet = pst.predictModelParameters(train[i])
        print("Datapoint - {},lam-{},beta-{}".format(i,lam,bet))
        error[i] = pst.calculateError(test,lam,bet)
        lambda_error[i] = ((true_lambda-lam)**2)#/(true_lambda**2)
        beta_error[i] = ((true_beta-bet)**2)#/(true_beta**2)

    
    plt.plot(list(error.keys()),list(error.values()),'r--o',label=r'$E((T-\hat{T})^2/T^2)$')
    plt.plot(list(lambda_error.keys()),list(lambda_error.values()),'b-*',label=r'$(\lambda-\hat{\lambda})^2/\hat{\lambda}^2$')
    plt.plot(list(beta_error.keys()),list(beta_error.values()),'g-.+',label=r'$(\beta-\hat{\beta})^2/\hat{\beta}^2$')
    plt.legend()
    plt.xlabel("Number of Simulations")
    plt.ylabel("WST Normalized Average Error")
    plt.ylim(0,10)
    plt.grid()
    plt.show()

if __name__ == '__main__':
    main()
from predictStoppingTime import predictStoppingTime
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from random import shuffle
from known_weights_batch_lcp_learning import BLKWLO
from known_weights_online_lcp_learning import KWOLLO

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
    dataset = 'gammadataset_100_0_85_0_15_0_85.csv'
    true_weights = [0.15,0.85]
    true_beta = 0.85
    true_lambda = 100

    fileContent = readDataset(dataset)

    pst = predictStoppingTime()
    preparedData = pst.prepareData(fileContent,true_weights)
    # preparedData = preparedData[:500]
    shuffle(preparedData)
    train = preparedData[:30]
    # print(train)
    test = preparedData[995:]
    error = {}
    lambda_error = {}
    beta_error = {}
    for i in range(len(train)):
        l = [0]*100
        b = [0]*100
        for j in range(100):
            l[j],b[j] = pst.predictModelParameters(train[i])
        # print("Datapoint - {},lam-{},beta-{}".format(i,lam,bet))
        lam = sum(l)/len(l)
        bet = sum(b)/len(b)
        error[i] = pst.calculateError(train,lam,bet)
        lambda_error[i] = ((true_lambda-lam)**2)/(true_lambda**2)
        beta_error[i] = ((true_beta-bet)**2)/(true_beta**2)

    b = BLKWLO()
    blkwlo_error,blkwlo_lambda_error,blkwlo_beta_error,blkwlo_test_rmse = b.main(train,test,true_weights,true_lambda,true_beta)

    o = KWOLLO()
    kwollo_error,kwollo_lambda_error,kwollo_beta_error,kwollo_test_rmse = o.main(train,test,true_weights,true_lambda,true_beta)


    fig, ax = plt.subplots()
    # plt.subplot(2,2,1)
    ax.plot(list(error.keys()),list(error.values()),'b--*',fillstyle='none',label='OL-BB')#label=r'$E((t^*-\hat{t^*})^2/(t^*)^2)$'
    ax.plot(list(blkwlo_error.keys()),list(blkwlo_error.values()),'g-.o',fillstyle='none',label='BL-QP')
    ax.plot(list(kwollo_error.keys()),list(kwollo_error.values()),'r-^',fillstyle='none',label='OL-QP')
    ax.legend()
    ax.set_xlabel("Number of Iterations")
    ax.set_ylabel(r'$E((t^*-\hat{t^*})^2/(t^*)^2)$')
    ax.grid()
    ax.set_xlim(1,len(train))
    ax.title.set_text(r'$\lambda=$'+str(true_lambda)+r" ;$\beta=$"+str(true_beta)+r";$\alpha_1=$"+str(true_weights[0])+r";$\alpha_2=$"+str(true_weights[1]))

    fig,ax1 = plt.subplots()
    # plt.subplot(2,2,3)
    ax1.plot(list(lambda_error.keys()),list(lambda_error.values()),'b--*',fillstyle='none',label='OL-BB')
    ax1.plot(list(blkwlo_lambda_error.keys()),list(blkwlo_lambda_error.values()),'g-.o',fillstyle='none',label='BL-QP')
    ax1.plot(list(kwollo_lambda_error.keys()),list(kwollo_beta_error.values()),'r-^',fillstyle='none',label='OL-QP')
    ax1.set_ylim(0,0.5)
    ax1.legend()
    ax1.set_xlabel("Number of Iterations")
    ax1.set_ylabel(r'$(\lambda - \hat{\lambda})^2/\lambda^2$')
    ax1.set_xlim(1,len(train))
    ax1.title.set_text(r'$\lambda=$'+str(true_lambda)+r" ;$\beta=$"+str(true_beta)+r";$\alpha_1=$"+str(true_weights[0])+r";$\alpha_2=$"+str(true_weights[1]))
    ax1.grid()

    fig,ax2 = plt.subplots()
    # plt.subplot(2,2,4)
    ax2.plot(list(beta_error.keys()),list(beta_error.values()),'b--*',fillstyle='none',label='OL-BB')
    ax2.plot(list(blkwlo_beta_error.keys()),list(blkwlo_beta_error.values()),'g-.o',fillstyle='none',label='BL-QO')
    ax2.plot(list(kwollo_beta_error.keys()),list(blkwlo_beta_error.values()),'r-^',fillstyle='none',label='OL-QO')
    ax2.set_ylim(0,0.5)
    ax2.legend()
    ax2.set_xlim(1,len(train))
    ax2.set_xlabel("Number of Iterations")
    ax2.set_ylabel(r'$(\beta - \hat{\beta})^2/\beta^2$')
    ax2.grid()
    ax2.title.set_text(r'$\lambda=$'+str(true_lambda)+r" ;$\beta=$"+str(true_beta)+r";$\alpha_1=$"+str(true_weights[0])+r";$\alpha_2=$"+str(true_weights[1]))
    # plt.plot(list(lambda_error.keys()),list(lambda_error.values()),'b-*',label=r'$(\lambda-\hat{\lambda})^2/\hat{\lambda}^2$')
    # plt.plot(list(beta_error.keys()),list(beta_error.values()),'g-.+',label=r'$(\beta-\hat{\beta})^2/\hat{\beta}^2$')
    
    # plt.ylim(0,4)
    # plt.xticks(np.arange(1,len(train)+1,2))
    # plt.xlim(1,len(train))
    # plt.title(r'$\lambda=$'+str(true_lambda)+r" ;$\beta=$"+str(true_beta)+r";$\alpha_1=$"+str(true_weights[0])+r";$\alpha_2=$"+str(true_weights[1]))
    # plt.grid()
    plt.show()

    # print("RMSE on test(OLBB)-",pst.calculateError(test,lam,bet))
    # print("RMSE on test(BLLO)-",blkwlo_test_rmse)
    # print("RMSE on test(OLLO)-",kwollo_test_rmse)
if __name__ == '__main__':
    main()
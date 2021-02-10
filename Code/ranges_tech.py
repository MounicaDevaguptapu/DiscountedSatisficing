
import pandas as pd
#import matplotlib.pyplot as plt
import numpy as np




MAX = 999999
MIN = 0
#TRUE_BETA = 0.855
#TRUE_LAMBDA = 500
LOG_FILE = 'range_technique_log.txt'
log_file = open(LOG_FILE,'w')

#if flag is Off, log will be printed in console
LOG_FLAG = False


# FILE_LIST = ['dataset_75_0_65.csv','dataset_100_0_95.csv','dataset_500_0_855.csv','dataset_100_0_35.csv']
FILE_LIST = ['dataset_100_0_95_0_63_0_37.csv']
#FILE_LIST = ['rewards_modelTest.csv']

def main():

    for i in range(len(FILE_LIST)):
        LAMBDA_MAP = []
        BETA_MAP = []
        BETA = 0
        LAMBDA = 0
        file_name = FILE_LIST[i]
        file_name = file_name.replace('.csv','')
        filename_split = file_name.split("_")
        TRUE_LAMBDA = float(filename_split[1])
        TRUE_BETA = float(filename_split[2]+'.'+filename_split[3])
        print(TRUE_BETA,TRUE_LAMBDA)

        f = open(FILE_LIST[i])
        content = f.read()
        total_data = []
        total_data = content.split('\n')

        #split into test and train
        content_array = np.array(total_data)
        np.random.shuffle(content_array)
        train_data,test_data = np.split(content_array,[900])        
        
        limits_map = {}
        count=1
        differences_limits_map = {}
        averages_limits_map = {}
        for line in train_data:
            if(line==''): continue
            rewards = [float(x.strip()) for x in line.split(",")]
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


            for i in range(1,len(limits_map.keys())+1):
                differences_limits_map[i] = limits_map[i][1] - limits_map[i][0]
                averages_limits_map[i] = (limits_map[i][1] + limits_map[i][0])/2

            sorted_differences = sorted(differences_limits_map, key=(lambda key:differences_limits_map[key]))
            log_file.write("---------------------------------\n") if(LOG_FLAG) else print("---------------------------------")
            log_file.write("Sorted Array of index with minimum difference in the limits {}\n".format(sorted_differences)) if(LOG_FLAG) else print("Sorted Array of index with minimum difference in the limits {}".format(sorted_differences))

            log_file.write("---------------------------------\n") if(LOG_FLAG) else print("---------------------------------")
            log_file.write("Average Limits HashMap {}\n".format(averages_limits_map)) if(LOG_FLAG) else print("Average Limits HashMap {}".format(averages_limits_map))

            BETA = (averages_limits_map[sorted_differences[1]]/averages_limits_map[sorted_differences[0]])**(1/(sorted_differences[1]-sorted_differences[0])) if(sorted_differences[1]>sorted_differences[0]) else (averages_limits_map[sorted_differences[0]]/averages_limits_map[sorted_differences[1]])**(1/(sorted_differences[0]-sorted_differences[1]))
            log_file.write("---------------------------------\n") if(LOG_FLAG) else print("---------------------------------")
            log_file.write("BETA: {}\n".format(BETA)) if(LOG_FLAG) else print("BETA: {}".format(BETA))

            if sorted_differences[0] == 1:
                LAMBDA = averages_limits_map[sorted_differences[0]]
            else:
                LAMBDA = averages_limits_map[sorted_differences[0]] / (BETA ** (sorted_differences[0] -1))

            #log_file_write("---------------------------------") if(LOG_FLAG) else print("---------------------------------")
            log_file.write("LAMBDA: {}\n".format(LAMBDA)) if(LOG_FLAG) else print("LAMBDA: {}".format(LAMBDA))
            LAMBDA_MAP.append((TRUE_LAMBDA - LAMBDA)**2)
            BETA_MAP.append((TRUE_BETA - BETA)**2)

        log_file.write("-----------Testing Started----------------------\n") if(LOG_FLAG) else print("-----------Testing Started----------------------")
        exactPredictions = 0
        #predict stoppping time on test data with estimated lambda and beta
        for line_test in test_data:
            if (line_test == ''): continue
            test_rewards = [float(x.strip()) for x in line_test.split(",")]
            stoppingTime = 0
            isSatisfied = False
            while not isSatisfied:
                if(sum(test_rewards[:stoppingTime]) >= (TRUE_BETA**stoppingTime) * TRUE_LAMBDA):
                    isSatisfied = True
                stoppingTime+= 1
            
            if stoppingTime == len(test_rewards):
                exactPredictions += 1

        modelAccuracy = (exactPredictions/len(test_data))*100
        log_file.write("-----------Testing Results----------------------\n") if(LOG_FLAG) else print("-----------Testing Results----------------------")
        log_file.write("Total Testing Data: {}\n".format(len(test_data))) if(LOG_FLAG) else print("Total Testing Data: {}".format(len(test_data)))
        log_file.write("Exact Number of Predictions: {}\n".format(exactPredictions)) if(LOG_FLAG) else print("Exact Number of Predictions: {}".format(exactPredictions))
        log_file.write("Model Accuracy: {}\n".format(modelAccuracy)) if(LOG_FLAG) else print("Model Accuracy: {}".format(modelAccuracy))
        log_file.write("---------------------------------\n") if(LOG_FLAG) else print("---------------------------------")

        #fig, axs = plt.subplots(2)
        #fig.suptitle(r'$\lambda =$'+str(TRUE_LAMBDA)+r'        $\beta = $'+str(TRUE_BETA))
        #axs[0].plot(LAMBDA_MAP,'tab:orange',marker='*',label=r'$MSE(\lambda)$')
        #axs[0].set(ylabel=r"$(\lambda - \hat{\lambda})^2$",xlabel='Number of Simulations')
        #axs[0].grid()
        #axs[0].legend()
        #axs[0].set_ylim([0,TRUE_LAMBDA+100])
        #axs[1].plot(BETA_MAP,'tab:blue',marker='+',label=r'$MSE(\beta)$')
        #axs[1].set(ylabel=r"$(\beta - \hat{\beta})^2$",xlabel='Number of Simulations')
        #axs[1].grid()
        #axs[1].legend()

        #plt.rcParams['axes.grid'] = True
        #plt.legend()
    #plt.show()

if __name__ == "__main__":
    main()

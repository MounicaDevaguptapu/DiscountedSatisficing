import numpy as np

class DataGeneration:
    def __init__(self):
        self.MAX = 35
        self.MIN = 0
        self.TRUE_LAMBDA = 100
        self.TRUE_BETA = 0.35
        self.totalAttributes = 2
        self.totalDataPoints = 1000
        #self.WEIGHTS = [1/self.totalAttributes for x in range(self.totalAttributes)]
        self.WEIGHTS = [0.15,0.85]
        self.rewards_line = []


    def generateReward(self):
        # return np.random.uniform(self.MIN,self.MAX)
        return np.random.gamma(2,scale=4)

    
    def DataGeneration(self):

        for i in range(self.totalDataPoints):
            isSatisfied = False
            accUtility = 0
            time = 1
            rewards = []
            while(not isSatisfied):
                #print("xxx")
                attrRewards = [0]*self.totalAttributes
                reward_part_line = []
                for j in range(self.totalAttributes):
                    attrRewards[j] = self.generateReward()
                    reward_part_line.append(attrRewards[j])
                    #print("----",reward_part_line)

                attWiseUtility = [self.WEIGHTS[i]*attrRewards[i] for i in range(self.totalAttributes)]
                #print("attr wise utility -- {}".format(attWiseUtility))
                rewards.append(reward_part_line)
                taskUtility = sum(attWiseUtility)
                #print("task utility -- {}".format(taskUtility))
                accUtility += taskUtility
                #print('acumulated utility',accUtility)

                if(accUtility >= (self.TRUE_BETA ** (time-1)) * self.TRUE_LAMBDA):
                    isSatisfied = True 
            self.rewards_line.append(rewards)
            #print(self.rewards_line)

    def writeCSV(self):
        dataFile = 'gammadataset_100_0_35_0_15_0_85.csv'
        f = open(dataFile,'w')
        for i in self.rewards_line:
            line = ''
            for j in i:
                line += str(j)+","
            f.write(line[:-1]+'\n')

        f.close()
  
def main():
    dataGenOBJ = DataGeneration()
    dataGenOBJ.DataGeneration()
    dataGenOBJ.writeCSV()

if __name__ == '__main__':
    main()

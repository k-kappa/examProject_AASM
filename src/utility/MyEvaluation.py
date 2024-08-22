
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random



class MyEvaluation:

    def __init__(self, epochsOfEvaluation):

        self.__epochsOfEvaluation = epochsOfEvaluation
        self.__array = np.full((self.__epochsOfEvaluation,3),-1) #steps,rew,info(0=win,1=loose,2=no-end)
        self.__randomArray = np.full((self.__epochsOfEvaluation,3),-1) #steps,rew,info(0=win,1=loose,2=no-end)

    def incrementStep(self,epoch):
        if (self.__array[epoch,0]==-1):
            self.__array[epoch,0]==0
        self.__array[epoch][0]+= 1

    def clearArray(self):
        self.__array = np.full((self.__array),-1)

    def addReward(self,reward,epoch):
        self.__array[epoch,1]=reward
        #print(self.__array)

    def addInfo(self,reasonEnd,epoch):
        self.__array[epoch,2] = reasonEnd   

    def randomPlayer(self,actions):
        self.__actions = actions

    def randomPlay(self,epoch):
        if (self.__randomArray[epoch,0]==-1):
            self.__randomArray[epoch,0]==0
        self.__randomArray[epoch,0]+=1
        return random.choice(self.__actions)    

    def addRandomReward(self,reward,epoch):
        #print("hey")
        #print(reward)
        self.__randomArray[epoch,1]=reward
        #print(self.__randomArray[epoch,1])

    def addRandomInfo(self,reasonEnd,epoch):
        self.__randomArray[epoch,2] = reasonEnd

    def addWins(self,numberWins):
        self.__numberWins = numberWins

    def addWinsRandom(self,numberWinsRandom):
        self.__numberWinsRandom = numberWinsRandom

    def getAverageReward(self):
        temp=0
        for x in self.__array:
            temp += x[1]
            #print(x)
            #print(x[1])
        return temp/self.__epochsOfEvaluation
    
    def getAverageRewardRandom(self):
        temp=0
        for x in self.__randomArray:
            temp += x[1]
        return temp/self.__epochsOfEvaluation

    def printRealPlayer(self):
        print("Real player has completed the games with an average of: "+str(self.__array[:,0].sum()/self.__epochsOfEvaluation)+
              " steps, "+str(self.getAverageReward())+" rewards, "+
              str(self.__numberWins)+ " wins")
        
    def str(self):
        print("Real player has completed the games with an average of: "+str(self.__array[:,0].sum()/self.__epochsOfEvaluation)+
              " steps, "+str(self.getAverageReward())+" rewards, "+
              str(self.__numberWins)+ " wins")
        print("Random player has completed the games with an average of: "+str(self.__randomArray[:,0].sum()/self.__epochsOfEvaluation)+
              " steps, "+str(self.getAverageRewardRandom())+" rewards, "+
              str(self.__numberWinsRandom)+ " wins")
    '''
    def showGraphLoss(self):
        plt.subplot(2, 2, 1)  # (rows, columns, subplot number)
        plt.plot(np.arange(1, self.__epochs + 1), self.__losses, label='Losses', color='blue')  # Plot the data
        plt.title('Losses')  # Add a title
        plt.xlabel('Epochs')  # Label for the x-axis
        plt.ylabel('Loss')  # Label for the y-axis
        plt.legend()  # Add a legend
        plt.grid(True)  # Add a grid for better readability
        #plt.show()  # Display the plot

    def showGraphReward(self):
        plt.subplot(2, 2, 2)  # (rows, columns, subplot number)
        plt.plot(np.arange(1, self.__epochs + 1), self.__rewards, label='rewards', color='green')  # Plot the data
        plt.title('Rewards')  # Add a title
        plt.xlabel('Epochs')  # Label for the x-axis
        plt.ylabel('Reward')  # Label for the y-axis
        plt.legend()  # Add a legend
        plt.grid(True)  # Add a grid for better readability
        #plt.show()  # Display the plot
    
    def showGraphDuration(self):
        plt.subplot(2, 2, 3)  # (rows, columns, subplot number)
        plt.plot(np.arange(1, self.__epochs + 1), self.__durations, label='Durations', color='yellow')  # Plot the data
        plt.title('Durations')  # Add a title
        plt.xlabel('Epochs')  # Label for the x-axis
        plt.ylabel('Duration')  # Label for the y-axis
        plt.legend()  # Add a legend
        plt.grid(True)  # Add a grid for better readability
        #plt.show()  # Display the plot

    def showAllGraphs(self):
        plt.figure(figsize=(12, 10))
        
        self.showGraphLoss()
        self.showGraphReward()
        self.showGraphDuration()
        plt.tight_layout()
        plt.show()  # Display the plot


    def __str__(self):
        return f"MyMetrics(instance_variable1={self.instance_variable1}, rewards={self.__rewards}, durations={self.__durations}, total duration = {self.__totalDuration})"
'''
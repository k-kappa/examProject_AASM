
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time



class MyMetrics:

    def __init__(self, epochs):

        self.__rewards = np.zeros(epochs)
        self.__losses = np.zeros(epochs)
        self.__durations = np.zeros(epochs)
        self.__epsilons = np.zeros(epochs)
        self.__totalDuration = 0.
        self.__epochs = epochs
        self.__start_time = 0
        self.__end_time = 0
        self.__loss_critic = np.zeros(epochs)
        self.__loss_actor = np.zeros(epochs)

    def addLoss(self,loss,epoch):
        self.__losses[epoch] = loss

    def addLossActorCritic(self,actorloss,criticloss,epoch):
        self.__loss_actor[epoch] = actorloss
        self.__loss_critic[epoch] = criticloss

    def addReward(self,reward,epoch):
        self.__rewards[epoch] = reward

    def addDuration(self,duration,epoch):
        self.__durations[epoch] = duration

    def addEpsilon(self,epsilon,epoch):
        self.__epsilons[epoch] = epsilon

    def saveMetrics(self):
        np.savez('./arraytest.npy', self.__losses, 
                 self.__rewards, self.__durations, 
                 self.__epsilons,
                 np.array(self.__totalDuration) )
    def getEpochs(self):
        return self.__epochs
    
    def startTimer(self):
        self.__start_time = time.time()

    def endTimer(self):
        self.__end_time = time.time()
    
    def getDuration(self):
        return self.__end_time - self.__start_time
    
    def showGraphLoss(self,actorcritic):
        if actorcritic:
            plt.subplot(2, 2, 1)  # (rows, columns, subplot number)
            plt.plot(np.arange(1, self.__epochs + 1), self.__loss_critic, label='Loss_critic', color='blue')  # Plot the data
            plt.plot(np.arange(1, self.__epochs + 1), self.__loss_actor, label='Loss_actor', color='purple')  # Plot the data
            plt.title('Losses')  # Add a title
            plt.xlabel('Epochs')  # Label for the x-axis
            plt.ylabel('Loss')  # Label for the y-axis
            plt.legend()  # Add a legend
            plt.grid(True)  # Add a grid for better readability
            #plt.show()  # Display the plot
        else:
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

    def showGraphEpsilon(self):
        plt.subplot(2, 2, 4)  # (rows, columns, subplot number)
        plt.plot(np.arange(1, self.__epochs + 1), self.__epsilons, label='Epsilon', color='red')  # Plot the data
        plt.title('Epsilon')  # Add a title
        plt.xlabel('Epochs')  # Label for the x-axis
        plt.ylabel('Epsilon')  # Label for the y-axis
        plt.legend()  # Add a legend
        plt.grid(True)  # Add a grid for better readability
        #plt.show()  # Display the plot

    def showAllGraphs(self):
        plt.figure(figsize=(12, 10))
        
        self.showGraphLoss(True) ##pass True if Actorcritic, False otherwise
        self.showGraphReward()
        self.showGraphDuration()
        self.showGraphEpsilon()
        plt.tight_layout()
        plt.show()  # Display the plot


    def __str__(self):
        return f"MyMetrics(instance_variable1={self.instance_variable1}, rewards={self.__rewards}, durations={self.__durations}, total duration = {self.__totalDuration})"

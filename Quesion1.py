#Quesion 1
import numpy as np

class Question1():
    def __init__(self):
        self.cluster1 = np.empty((20,2))      #cluster variable
        self.cluster2 = np.empty((20,2))      #cluster variable
        self.cluster3 = np.empty((20,2))      #cluster variable
        self.cluster4 = np.empty((20,2))      #cluster variable
        self.clusters = np.empty((80, 2))     #combined clusters
        self.generateClusters()

    def generateClusters(self):
        self.cluster1 = np.random.normal((1,-1), (1,1), (20,2))
        self.cluster2 = np.random.normal((-1,1), (1,1), (20,2))
        self.cluster3 = np.random.normal((-1,-1), (1,1), (20,2))
        self.cluster4 = np.random.normal((1,1), (1,1), (20,2))

        return self.cluster1, self.cluster2, self.cluster3, self.cluster4 

    def getClusters(self):
        self.clusters = np.concatenate((self.cluster1,self.cluster2,self.cluster3,self.cluster4), axis=0)
        return self.clusters
from multiprocessing.dummy import current_process
import numpy as np
import math

class kMeans():
    def __init__(self, clusters, k):
        self.clusters = clusters
        self.k = k
        self.centres = np.empty((self.k, 2), np.float64)
        self.distances = np.empty((80, self.k))

    def firstCentres(self):
        self.centres = np.random.normal((0,0), (1.5,1.5), (self.k,2))
        return self.centres

    def getDistances(self):
        for i in range(80):
            for j in range(self.k):
                self.distances[i][j] = np.sqrt(np.sum(np.square(self.clusters[i]-self.centres[j])))
        return self.distances

    def getNewClusters(self):
        p1 = np.amin(self.distances, axis=1)
        pos = np.empty((80), np.int8)
        temp0 = np.empty((0, 2))
        temp1 = np.empty((0, 2))
        temp2 = np.empty((0, 2))
        temp3 = np.empty((0, 2))

        for i in range(80):
            for j in range(4):
                if(p1[i]==self.distances[i][j]):
                    pos[i] = j

        for i in range(80):
            if(pos[i]==0):
                temp0 = np.append(temp0, np.array([self.clusters[i]]), axis=0)
            elif(pos[i]==1):
                temp1 = np.append(temp1, np.array([self.clusters[i]]), axis=0)
            elif(pos[i]==2):
                temp2 = np.append(temp2, np.array([self.clusters[i]]), axis=0)
            elif(pos[i]==3):
                temp3 = np.append(temp3, np.array([self.clusters[i]]), axis=0)

        self.cluster0 = temp0
        self.cluster1 = temp1
        self.cluster2 = temp2
        self.cluster3 = temp3
        
        return self.cluster0, self.cluster1, self.cluster2, self.cluster3
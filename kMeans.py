from multiprocessing.dummy import current_process
import numpy as np
import math

class kMeans():
    def __init__(self, clusters, k):
        self.clusters = clusters
        self.k = k
        self.centres = np.empty((self.k, 2), np.float64)
        self.distances = np.empty((80, self.k))
        self.firstCentres()

    def firstCentres(self):
        self.centres = np.random.normal((0,0), (1.5,1.5), (self.k,2))
        return self.centres

    def getDistances(self):
        for i in range(80):
            for j in range(self.k):
                self.distances[i][j] = np.sqrt(np.sum(np.square(self.clusters[i]-self.centres[j])))
        
        return self.distances

    def getNewClusters(self):
        self.getDistances()

        p1 = np.amin(self.distances, axis=1)
        self.pos = np.empty((80), np.int8)
        temp0 = np.empty((0, 2))
        temp1 = np.empty((0, 2))
        temp2 = np.empty((0, 2))
        temp3 = np.empty((0, 2))
        temp4 = np.empty((0, 2))
        temp5 = np.empty((0, 2))

        for i in range(80):
            for j in range(self.k):
                if(p1[i]==self.distances[i][j]):
                    self.pos[i] = j

        for i in range(80):
            if(self.pos[i]==0):
                temp0 = np.append(temp0, np.array([self.clusters[i]]), axis=0)
            elif(self.pos[i]==1):
                temp1 = np.append(temp1, np.array([self.clusters[i]]), axis=0)
            elif(self.pos[i]==2):
                temp2 = np.append(temp2, np.array([self.clusters[i]]), axis=0)
            elif(self.pos[i]==3):
                temp3 = np.append(temp3, np.array([self.clusters[i]]), axis=0)
            elif(self.pos[i]==4):
                temp4 = np.append(temp4, np.array([self.clusters[i]]), axis=0)
            elif(self.pos[i]==5):
                temp5 = np.append(temp5, np.array([self.clusters[i]]), axis=0)

        self.cluster0 = temp0
        self.cluster1 = temp1
        self.cluster2 = temp2
        self.cluster3 = temp3
        self.cluster4 = temp4
        self.cluster5 = temp5
        self.clusters = np.concatenate((self.cluster0,self.cluster1,self.cluster2,self.cluster3,self.cluster4,self.cluster5), axis=0)

        return self.cluster0, self.cluster1, self.cluster2, self.cluster3, self.cluster4, self.cluster5

    def getNewCentre(self):
        self.centres[0] = np.average(self.cluster0, axis=0)
        self.centres[1] = np.average(self.cluster1, axis=0)
        self.centres[2] = np.average(self.cluster2, axis=0)
        self.centres[3] = np.average(self.cluster3, axis=0)
        self.centres[4] = np.average(self.cluster4, axis=0)
        self.centres[5] = np.average(self.cluster5, axis=0)

        return self.centres

    def getObjectiveFunction(self):
        self.obj = 0.0

        for i in range(self.cluster0.shape[1]):
            self.obj += np.sum(np.square(self.cluster0[i]-self.centres[0]))
        
        for i in range(self.cluster1.shape[1]):
            self.obj += np.sum(np.square(self.cluster1[i]-self.centres[1]))
        
        for i in range(self.cluster2.shape[1]):
            self.obj += np.sum(np.square(self.cluster2[i]-self.centres[2]))
        
        for i in range(self.cluster3.shape[1]):
            self.obj += np.sum(np.square(self.cluster3[i]-self.centres[3]))

        for i in range(self.cluster4.shape[1]):
            self.obj += np.sum(np.square(self.cluster4[i]-self.centres[4]))

        for i in range(self.cluster5.shape[1]):
            self.obj += np.sum(np.square(self.cluster5[i]-self.centres[5]))

        self.obj = self.obj/80

        return self.obj
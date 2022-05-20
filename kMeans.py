import numpy as np

class kMeans():
    def __init__(self, clusters, k):
        self.clusters = clusters
        self.k = k
        self.centres = np.empty((self.k, 3), np.float64)
        self.distances = np.empty((self.clusters.shape[0], self.k))
        self.firstCentres()

    def firstCentres(self):
        for i in range(self.k):
            for j in range(3):
                self.centres[i][j] = np.random.randint(0, 255)
        return self.centres

    def getDistances(self):
        for i in range(self.clusters.shape[0]):
            for j in range(self.k):
                self.distances[i][j] = np.sqrt(np.sum(np.square(self.clusters[i]-self.centres[j])))
        return self.distances

    def getNewClusters(self):
        self.getDistances()

        p1 = np.amin(self.distances, axis=1)
        self.pos = np.empty((self.clusters.shape[0]), np.int8)
        temp0 = np.empty((0, 3))
        temp1 = np.empty((0, 3))
        
        for i in range(self.clusters.shape[0]):
            for j in range(self.k):
                if(p1[i]==self.distances[i][j]):
                    self.pos[i] = j

        for i in range(self.clusters.shape[0]):
            if(self.pos[i]==0):
                temp0 = np.append(temp0, np.array([self.clusters[i]]), axis=0)
            elif(self.pos[i]==1):
                temp1 = np.append(temp1, np.array([self.clusters[i]]), axis=0)
        self.cluster0 = temp0
        self.cluster1 = temp1
        self.clusters = np.concatenate((self.cluster0,self.cluster1), axis=0)

        return self.cluster0, self.cluster1

    def getNewCentre(self):
        self.centres[0] = np.average(self.cluster0, axis=0)
        self.centres[1] = np.average(self.cluster1, axis=0)
        return self.centres

    def getObjectiveFunction(self):
        self.obj = 0.0

        for i in range(self.cluster0.shape[1]):
            self.obj += np.sum(np.square(self.cluster0[i]-self.centres[0]))
        
        for i in range(self.cluster1.shape[1]):
            self.obj += np.sum(np.square(self.cluster1[i]-self.centres[1]))
        self.obj = self.obj/self.clusters.shape[0]

        return self.obj
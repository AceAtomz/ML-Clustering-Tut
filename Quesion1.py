#Quesion 1
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

class Question1():
    def __init__(self):
        self.cluster1 = np.empty((20,2))      #cluster variable
        self.cluster2 = np.empty((20,2))      #cluster variable
        self.cluster3 = np.empty((20,2))      #cluster variable
        self.cluster4 = np.empty((20,2))      #cluster variable

    def generateClusters(self):
        self.cluster1 = np.random.normal((1,-1), (1,1), (20,2))
        self.cluster2 = np.random.normal((-1,1), (1,1), (20,2))
        self.cluster3 = np.random.normal((-1,-1), (1,1), (20,2))
        self.cluster4 = np.random.normal((1,1), (1,1), (20,2))

        return self.cluster1, self.cluster2, self.cluster3, self.cluster4 
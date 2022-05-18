#Quesion 1

from turtle import shape
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import math

class Question1():
    def __init__(self, numClusters):
        self.num = numClusters
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


np.random.seed(4567) #set seed to generate same data for training

Q1 = Question1(4)
cluster1, cluster2, cluster3, cluster4 = Q1.generateClusters()
print("ow ",cluster1)


x11 = cluster1[:, 0]
x12 = cluster1[:, 1]
x21 = cluster2[:, 0]
x22 = cluster2[:, 1]
x31 = cluster3[:, 0]
x32 = cluster3[:, 1]
x41 = cluster4[:, 0]
x42 = cluster4[:, 1]


fig, ax1 = plt.subplots(figsize=(7,5))
ax1.scatter(x11, x12,  color = 'blue')
ax1.scatter(x21, x22,  color = 'blue') #red
ax1.scatter(x31, x32,  color = 'blue') #hotpink
ax1.scatter(x41, x42,  color = 'blue') #lime
#ax1.scatter(x=1, y=-1, color='red')
ax1.set_title('Training Data for Clustering')
#fig.legend()
ax1.set_xlabel("x1", fontsize=15)
ax1.set_ylabel("x2", fontsize=15)
fig.tight_layout()
plt.show()
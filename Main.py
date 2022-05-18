from turtle import shape
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from Quesion1 import *

np.random.seed(4567) #set seed to generate same data for training

Q1 = Question1()
cluster1, cluster2, cluster3, cluster4 = Q1.generateClusters()
clusters = np.concatenate((cluster1,cluster2,cluster3,cluster4), axis=0)

x11 = clusters[:, 0]
x12 = clusters[:, 1]

fig, ax1 = plt.subplots(figsize=(7,5))
ax1.scatter(x11, x12,  color = 'blue')
ax1.set_title('Training Data for Clustering')
#fig.legend()
ax1.set_xlabel("x1", fontsize=15)
ax1.set_ylabel("x2", fontsize=15)
fig.tight_layout()
plt.show()
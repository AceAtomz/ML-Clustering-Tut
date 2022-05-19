from turtle import shape
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from Quesion1 import *
from kMeans import *

np.random.seed(4567) #set seed to generate same data for training

Q1 = Question1()
clusters = Q1.getClusters()

np.random.seed(4109) #new seed for new initialization of centres

kM = kMeans(clusters, 7)
obj=0.0
for i in range(10):
    cluster1, cluster2, cluster3, cluster4, cluster5, cluster6, cluster7 = kM.getNewClusters()
    centres = kM.getNewCentre()
    newobj = kM.getObjectiveFunction()
    if(newobj==obj):
        print(i)
        break
    else:
        obj = newobj

x11 = cluster1[:, 0]
x12 = cluster1[:, 1]
x21 = cluster2[:, 0]
x22 = cluster2[:, 1]
x31 = cluster3[:, 0]
x32 = cluster3[:, 1]
x41 = cluster4[:, 0]
x42 = cluster4[:, 1]
x51 = cluster5[:, 0]
x52 = cluster5[:, 1]
x61 = cluster6[:, 0]
x62 = cluster6[:, 1]
x71 = cluster7[:, 0]
x72 = cluster7[:, 1]

fig, ax1 = plt.subplots(figsize=(7,5))
ax1.scatter(x11, x12,  color = 'blue')
ax1.scatter(centres[0][0], centres[0][1], color='blue', marker="s", s=150)
ax1.scatter(x21, x22,  color = 'red')
ax1.scatter(centres[1][0], centres[1][1], color='red', marker="s", s=150)
ax1.scatter(x31, x32,  color = 'hotpink')
ax1.scatter(centres[2][0], centres[2][1], color='hotpink', marker="s", s=150)
ax1.scatter(x41, x42,  color = 'lime') 
ax1.scatter(centres[3][0], centres[3][1], color='lime', marker="s", s=150)
ax1.scatter(x51, x52,  color = 'purple')
ax1.scatter(centres[4][0], centres[4][1], color='purple', marker="s", s=150)
ax1.scatter(x61, x62,  color = 'orange')
ax1.scatter(centres[5][0], centres[5][1], color='orange', marker="s", s=150)
ax1.scatter(x71, x72,  color = 'green')
ax1.scatter(centres[6][0], centres[6][1], color='green', marker="s", s=150)

ax1.set_title('Eighth Iteration with k=7 Clusters  (OBJ = ' + str(obj) + ')')
ax1.set_xlabel("x1", fontsize=15)
ax1.set_ylabel("x2", fontsize=15)
fig.tight_layout()
plt.show()
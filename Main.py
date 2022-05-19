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

#np.random.seed(4109) #new seed for new initialization of centres

kM = kMeans(clusters, 4)
obj=np.array([0.0])
count=np.array([], np.int8)
for i in range(10):
    count = np.append(count, np.array([i+1]))
    cluster1, cluster2, cluster3, cluster4 = kM.getNewClusters()
    centres = kM.getNewCentre()
    newobj = kM.getObjectiveFunction()
    if(newobj==obj[i]):
        print(i)
        break
    else:
        obj= np.append(obj, np.array([newobj]), axis=0)

obj = np.delete(obj, 0)
obj = np.append(obj, np.array([newobj]), axis=0)

x11 = cluster1[:, 0]
x12 = cluster1[:, 1]
x21 = cluster2[:, 0]
x22 = cluster2[:, 1]
x31 = cluster3[:, 0]
x32 = cluster3[:, 1]
x41 = cluster4[:, 0]
x42 = cluster4[:, 1]

fig, (ax1, ax2) = plt.subplots(2, figsize=(7,7))

ax1.scatter(x11, x12,  color = 'blue')
ax1.scatter(centres[0][0], centres[0][1], color='blue', marker="s", s=150)
ax1.scatter(x21, x22,  color = 'red') #red
ax1.scatter(centres[1][0], centres[1][1], color='red', marker="s", s=150)
ax1.scatter(x31, x32,  color = 'hotpink') #hotpink
ax1.scatter(centres[2][0], centres[2][1], color='hotpink', marker="s", s=150)
ax1.scatter(x41, x42,  color = 'lime') #lime
ax1.scatter(centres[3][0], centres[3][1], color='lime', marker="s", s=150)

ax1.set_title('Sixth Iteration with k=4 Clusters  (OBJ = ' + str(obj[obj.shape[0]-1]) + ')')
ax1.set_xlabel("x1", fontsize=15)
ax1.set_ylabel("x2", fontsize=15)
fig.tight_layout()

ax2.plot(count, obj)
ax2.scatter(count, obj)
plt.show()
"""
#Plotting errors of different k
x=np.array([2,3,4,5,6,7,8])
y=np.array([0.14756, 0.10897, 0.09194, 0.11486, 0.15551, 0.14122, 0.12864])
ax1.plot(x,y)
ax1.scatter(x,y)
ax1.set_title('Number of k-means clusters vs Objective function at convergence')
ax1.set_xlabel("k", fontsize=15)
ax1.set_ylabel("obj", fontsize=15)
fig.tight_layout()
plt.show()
"""
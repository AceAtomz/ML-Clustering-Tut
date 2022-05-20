#Question 3
import numpy as np
import matplotlib.pyplot as plt
from kMeans import *

originalImg = plt.imread("peppers1.jpg") 
img = np.reshape(originalImg, (originalImg.shape[0]*originalImg.shape[1], originalImg.shape[2]))

np.random.seed(4109) #new seed for new initialization of centres
k=2
kM = kMeans(img, k)
obj=0.0

for i in range(10):
    cluster1, cluster2 = kM.getNewClusters()
    centres = kM.getNewCentre()
    newobj = kM.getObjectiveFunction()
    if(newobj==obj):
        print(i)
        break
    else:
        obj = newobj

cluster1 = np.array(cluster1, dtype=np.int16)
cluster2 = np.array(cluster2, dtype=np.int16)
print(centres)
for i in range(img.shape[0]):
    for j in range(cluster1.shape[0]):
        if(img[i][0]==cluster1[j][0] and img[i][1]==cluster1[j][1] and img[i][2]==cluster1[j][2]):
            img[i] = centres[0]
            break
    for j in range(cluster2.shape[0]):
        if(img[i][0]==cluster2[j][0] and img[i][1]==cluster2[j][1] and img[i][2]==cluster2[j][2]):
            img[i] = centres[1]
            break
        
img = np.reshape(img, (originalImg.shape[0], originalImg.shape[1], originalImg.shape[2]))

fig, ax1 = plt.subplots(figsize=(7,6))
ax1.imshow(img)
ax1.axis('off')
ax1.set_title('Peppers k=2-means clustering (4th iteration)')
fig.tight_layout()
plt.show()
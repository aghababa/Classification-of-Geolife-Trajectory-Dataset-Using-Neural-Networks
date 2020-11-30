#!/usr/bin/env python
# coding: utf-8

# @ Hasan November 29, 2020, @ 23:58


import glob
import numpy as np
import time
import math
import random
from scipy import linalg as LA
import pandas as pd


# In[2]:


def read_file(file_name):
    data = []
    c = 0
    with open(file_name, "r") as f:
        for line in f:
            c = c + 1
            if c > 6:
                item = line.strip().split(",")
                data.append(list(map(float,item[:2])))
    return data


# In[3]:


J = glob.glob('Geolife Trajectories 1.3/**/', recursive=True)[2:]
K = [J[2*i] for i in range(182)]
F = [K[i][30:33] for i in range(182)]
int1 = np.vectorize(int)
folder_numbers = int1(F)
I = glob.glob('Geolife Trajectories 1.3/**/*.plt', recursive=True)


# In[4]:


# runtime about 65s
Start_time = time.time()

data_indexed = [0] * 182
data = [0] * 182

for i in range(182):
    data_indexed[folder_numbers[i]] = []
    data[folder_numbers[i]] = []
    j = 0
    for file_name in I:
        if file_name[30:33] == F[i]:
            a = np.array(read_file(file_name))
            data[folder_numbers[i]].append(a)
            data_indexed[folder_numbers[i]].append(([j, folder_numbers[i], a]))
            j = j+1
    
print('total time =', time.time() - Start_time)


# In[5]:


data_indexed[-1][6]


# In[6]:


data_reduced = [0] * 182
data_fol_num = set()

for i in range(182):
    data_reduced[i] = []
    for j in range(len(data_indexed[i])):
        if len(data_indexed[i][j][2]) > 10: 
            data_reduced[i].append(data_indexed[i][j])
            data_fol_num.add(data_indexed[i][j][1])

data_fol_num = np.sort(list(data_fol_num))


# In[8]:


l = 0
index = [] 
for i in range(182):
    if len(data_reduced[i]) < 10: #10
        l = l + 1
    else:
        index.append(i)
        
l, 182-l


# In[9]:


data_1 = [0] * (182-l)
data_fol_num1 = set()

j = 0 
for i in range(182):
    if len(data_reduced[i]) >= 10: #10:
        data_1[j] = np.array(data_reduced[i])
        data_fol_num1.add(data_reduced[i][0][1])
        j = j + 1
data_fol_num1 = np.sort(list(data_fol_num1))
# or we could set data_1 = np.array(data_reduced)[index]


# In[10]:


f = 0
for i in range(182-l):
    if len(data_1[i]) > 200:
        f = f + 1
        #print(len(data_1[i]))
f


# In[11]:


data_final = data_1
for i in range(182-l):
    if len(data_1[i]) > 200:
        R1 = random.sample(range(len(data_1[i])), 200)
        R = np.sort(R1)
        data_final[i] = data_1[i][R]

data_final = np.array(data_final)


# In[12]:


data_train = [0] * (182-l)
data_test = [0] * (182-l)

for i in range(182-l):
    p = len(data_final[i])
    R1 = random.sample(range(p), int(0.3 * p))
    R = np.sort(R1)
    R_c = np.sort(list(set(range(p)) - set(R)))
    data_train[i] = data_final[i][R_c]
    data_test[i] = data_final[i][R]


# In[13]:


# creating box boundaries for landmark points for whole dataset
A = []
B = []
U = []
V = []
M1 = []
M2 = []

for j in range(len(data_final)):
    a = min([min(data_final[j][i][2][:,0]) for i in range(len(data_final[j]))])
    b = max([max(data_final[j][i][2][:,0]) for i in range(len(data_final[j]))])
    u = min([min(data_final[j][i][2][:,1]) for i in range(len(data_final[j]))])
    v = max([max(data_final[j][i][2][:,1]) for i in range(len(data_final[j]))])
    A.append(a)
    B.append(b)
    U.append(u)
    V.append(v)
    
    m1 = np.mean([np.mean(data_final[j][i][2][:,0]) for i in range(len(data_final[j]))])
    m2 = np.mean([np.mean(data_final[j][i][2][:,1]) for i in range(len(data_final[j]))])
    M1.append(np.mean(m1))
    M2.append(np.mean(m2))


print("x-coordinate minimum is: ", min(A))
print("x-coordinate maximum is: ", max(B)) 
print("x-coordinate mean is: ", np.mean(M1))
print("x-coordinate median is: ", np.median(M1))
print("x-coordinate variance is: ", np.var(M1))
print("x-coordinate standard deviation is: ", np.sqrt(np.var(M1)), '\n')

print("y-coordinate minimum is: ", min(U)) 
print("y-coordinate maximun is: ", max(V))
print("y-coordinate mean is: ", np.mean(M2))
print("y-coordinate median is: ", np.median(M2))
print("y-coordinate variance is: ", np.var(M2))
print("y-coordinate standard deviation is: ", np.sqrt(np.var(M2)))


# # Choosing Landmarks

# ## Landmarks of size 20

# In[14]:


Q_20 = np.array([[39.6,114.8],[40,116.34], [39, 114], [40,115], [41,116], [39.5,113.5], 
                [40.5, 115.5], [40.3, 114.4], [38.5, 113.5], [39.3,117], [39.5,115], 
                [39.4, 116], [39,114.6], [40.5, 115.5], [41.2,115], [42, 116], [41.7, 114.8], 
               [38,115], [37, 116], [37.5, 114]])


# In[16]:


np.savetxt('DL Project/Q_20.csv', Q_20, delimiter=',')


# ## Landmarks of size 50

# In[19]:


# This is extracted from map via visualization
Q_50 = np.array([[40.02784036172628, 116.50178611278535],
[40.085652349925475, 116.39329612255098],
[40.08039874140986, 116.24772727489473],
[40.05202225054736, 116.19966208934785],
[39.97418906423696, 116.1694496870041],
[39.899427860534345, 116.16807639598848],
[39.831967504781055, 116.2518471479416],
[39.82880367321281, 116.3617104291916],
[39.87097611205365, 116.4935463666916],
[39.95419054874236, 116.43724143505098],
[40.00575375653265, 116.37956321239473],
[40.024685570075626, 116.31501853466034],
[39.9699793365808, 116.2738198041916],
[39.90996254870152, 116.2793129682541],
[39.91733586660022, 116.3617104291916],
[39.967874375523536, 116.34385764598848],
[39.71269591644774, 116.47981345653535],
[39.66091215703148, 116.37407004833223],
[39.6989611532183, 116.2573403120041],
[39.74226919672803, 116.06095969676973],
[39.90996254870152, 116.01426780223848],
[40.129766654747705, 116.07743918895723],
[40.21895804658573, 116.0760658979416],
[40.257745909206854, 116.25596702098848],
[40.31640959147105, 116.10215842723846],
[40.20008024183078, 116.58418357372285],
[40.07724638171063, 116.79155051708223],
[39.897320728525514, 116.76133811473848],
[40.095087305350994, 116.98068380355835],
[39.93416561718383, 117.04934835433961],
[39.60272060950412, 117.02874898910524],
[39.537087506313945, 116.85022115707399],
[39.32177827012156, 116.49766623973848],
[39.35045657030925, 116.33699119091035],
[39.45338894346977, 116.05271995067598],
[40.030995007481806, 116.04585349559785],
[40.28917921376592, 116.4660805463791],
[40.41685741933303, 115.90852439403535],
[39.41933055574406, 116.00316211581233],
[39.4341814883212, 115.56096240878108],
[38.994696721865594, 115.50629228353503],
[39.1578093485124, 115.17120927572252],
[38.91460188981307, 115.06134599447253],
[39.96682187068797, 115.84260642528535],
[40.444612929944775, 116.78323835134509],
[39.037848825595745, 117.04134196043016],
[39.18809007444354, 117.30501383543016],
[39.65284221780906, 116.5751498937607],
[39.80176509832639, 115.9763950109482], 
[39.70176509836639, 116.9763900109482]])


# In[33]:


np.savetxt('DL Project/Q_50.csv', Q_50, delimiter=',')


# ## Landmarks of size 250

# In[22]:


x = list(np.linspace(37, 43, 11)) 
y = list(np.linspace(110, 120, 17))

Q = np.zeros((len(x) * len(y), 2))
k = 0 

for i in range(len(x)):
    for j in range(len(y)):
        Q[k+j] = [x[i],y[j]]
    k = k + len(y)

Q_250 = list(Q) + [[1, -180], [1,180], [400, -180], [400, 180], [100, 35], [80, 45], 
                    [50, 120], [35, 130], [40, 125], [40, 105], [34,115], [46, 115], 
                    [39,0]] + list(Q_50)
Q_250 = np.array(Q_250)


# In[23]:


np.savetxt('DL Project/Q_250.csv', Q_250, delimiter=',')


# In[27]:


Q = Q_250 # Q_50, Q_20


# # First Feature Map (Orientation Preserving)

# In[ ]:


def dist_signed_point_closed(Q, gamma, sigma): 
    
    p1 = gamma[:-1]
    p2 = gamma[1:]
    L = np.sqrt(((p2-p1)*(p2-p1)).sum(axis =1)) + 10e-6
    
    w = (p1-p2)*(-1,1)/(L * np.ones((2,1))).T
    w[:,[0, 1]] = w[:,[1, 0]]
    
# signed distance to the extended lines of segments
    dist_signed = np.sum(w * (Q.reshape(len(Q),1,2) - p1), axis=2)
    x = abs(dist_signed.copy())
    R = (L**2).reshape(-1,1)
# u = argmin points on the extended lines of segments
    u = p1 + ((((np.sum(((Q.reshape(len(Q),1,2) - p1) * (p2 - p1)),axis=2).reshape(len(Q)
                ,-1,1,1) * (p2-p1).reshape(len(p2-p1),1,2))).reshape(len(Q),len(p1),2))/R)

    G = np.sqrt(np.sum((u-p1)*(u-p1), axis=2))
    H = np.sqrt(np.sum((u-p2)*(u-p2), axis=2))
# d1 = distance to start points
    d1 = np.sqrt(np.sum((Q.reshape(len(Q),1,2)-p1)*(Q.reshape(len(Q),1,2)-p1), axis=2))
# d2 = distance to end points
    d2 = np.sqrt(np.sum((Q.reshape(len(Q),1,2)-p2)*(Q.reshape(len(Q),1,2)-p2), axis=2))
    d = np.where(d1 < d2, d1, d2)
    dist_segment = np.where(abs(G + H - L) < np.ones(len(L)) * (10e-6), dist_signed, d)
    
    J2 = [0] * len(Q)
    for i in range(len(Q)): 
        J2[i] = np.where(abs(G + H - L)[i] > 10e-6)[0]
    J2 = np.array(J2)

    dist_segment_copy = dist_segment.copy()
    dist = abs(dist_segment_copy)


    j = np.argmin(dist, axis =1)

    sign = np.ones(len(Q))
    for k in range(len(Q)): 
        if j[k] in J2[k]:
            if j[k] == 0 and LA.norm(Q[k] - gamma[0]) < LA.norm(Q[k] - gamma[1]):
                
                y = LA.norm(gamma[0]-gamma[1]) - LA.norm(gamma[-1] - gamma[-2])
                if y < 0:
                    x = gamma[0] + 0.1 * LA.norm(gamma[0]-gamma[1])*(gamma[-2]-gamma[-1])/LA.norm(gamma[-2]-gamma[-1])
                    z = gamma[0] + 0.1 * LA.norm(gamma[0]-gamma[1])*(gamma[1]-gamma[0])/LA.norm(gamma[1]-gamma[0])
                    q = 2 * gamma[0] - (x + z)/2
                else: 
                    x = gamma[0] + 0.1 * LA.norm(gamma[-1]-gamma[-2])*(gamma[1]-gamma[0])
                    z = gamma[0] + 0.1 * LA.norm(gamma[-1]-gamma[-2])*(gamma[-2]-gamma[-1])
                    q = 2 * gamma[0] - (x + z)/2
                sign[k] = np.sign((q-gamma[-1]).dot(w[-1] + w[0]))
                
            elif j[k] == len(gamma)-2 and LA.norm(Q[k] - gamma[-1]) < LA.norm(Q[k] - gamma[-2]):
                s = w[-1].dot((Q[k] - gamma[-1])/ LA.norm(Q[k] - gamma[-1]) + 10e-6)
                sign[k] = np.sign(s)
            
            elif LA.norm(Q[k] - gamma[j[k]]) < LA.norm(Q[k] - gamma[j[k]+1]):  
                q = 2 * gamma[j[k]] - (gamma[j[k]-1] + gamma[j[k]+1])/2
                sign[k] = np.sign((q-gamma[j[k]]).dot(w[j[k]-1] + w[j[k]]))
                    
            elif LA.norm(Q[k] - gamma[j[k]+1]) <= LA.norm(Q[k] - gamma[j[k]]):
                q = 2 * gamma[j[k]+1] - (gamma[j[k]] + gamma[j[k]+2])/2
                sign[k] = np.sign((q-gamma[j[k]+1]).dot(w[j[k]] + w[j[k]+1]))

    E = dist_segment[np.arange(len(dist_segment)),j] 
    F = dist[np.arange(len(dist)),j] 
    dist_weighted = sign * (1/sigma) * (E.reshape(-1,1) * np.exp(-(F/sigma)**2).reshape(-1,1)).reshape(1,-1)

    return dist_weighted.reshape(len(Q))


# In[ ]:


def dist_signed_point_unclosed(Q, gamma, sigma): 
    
    p1 = gamma[:-1]
    p2 = gamma[1:]
    L = np.sqrt(((p2-p1)*(p2-p1)).sum(axis =1)) + 10e-6
    w = (p1-p2)*(-1,1)/(L * np.ones((2,1))).T
    w[:,[0, 1]] = w[:,[1, 0]]
    
# signed distance to the extended lines of segments
    dist_signed = np.sum(w * (Q.reshape(len(Q),1,2) - p1), axis=2)
    x = abs(dist_signed.copy())
    R = (L**2).reshape(-1,1)
# u = argmin points on the extended lines of segments
    u = p1 + ((((np.sum(((Q.reshape(len(Q),1,2) - p1) * (p2 - p1)),axis=2).reshape(len(Q)
                ,-1,1,1) * (p2-p1).reshape(len(p2-p1),1,2))).reshape(len(Q),len(p1),2))/R)

    G = np.sqrt(np.sum((u-p1)*(u-p1), axis=2))
    H = np.sqrt(np.sum((u-p2)*(u-p2), axis=2))
# d1 = distance to start points
    d1 = np.sqrt(np.sum((Q.reshape(len(Q),1,2)-p1)*(Q.reshape(len(Q),1,2)-p1), axis=2))
# d2 = distance to end points
    d2 = np.sqrt(np.sum((Q.reshape(len(Q),1,2)-p2)*(Q.reshape(len(Q),1,2)-p2), axis=2))
    d = np.where(d1 < d2, d1, d2)
    dist_segment = np.where(abs(G + H - L) < np.ones(len(L)) * (10e-6), dist_signed, d)
    
    J2 = [0] * len(Q)
    for i in range(len(Q)): 
        J2[i] = np.where(abs(G + H - L)[i] > 10e-6)[0]
    J2 = np.array(J2)

    dist_segment_copy = dist_segment.copy()
    dist = abs(dist_segment_copy)
    
    dist_from_start_1 = np.sqrt(((Q -p1[0])*(Q -p1[0])).sum(axis =1))
    ds_1 = ((Q -p1[0])*w[0]).sum(axis =1)
    #dist_from_start = np.sign(ds_1) * (abs(ds_1) + np.sqrt(dist_from_start_1**2 - ds_1**2 + 10e-6))
    dist_from_start = ds_1 * np.maximum(abs(ds_1), np.sqrt(dist_from_start_1**2 - ds_1**2 + 10e-6))/ (dist_from_start_1 + 10e-6)

    
    dist_from_end_1 = np.sqrt(((Q -p2[-1])*(Q -p2[-1])).sum(axis =1))
    de_1 = ((Q -p2[-1])* w[-1]).sum(axis =1)
    #dist_from_end = np.sign(de_1) * (abs(de_1) + np.sqrt(dist_from_end_1**2 - de_1**2 + 10e-6))
    dist_from_end = de_1 * np.maximum(abs(de_1), np.sqrt(dist_from_end_1**2 - de_1**2 + 10e-6))/ (dist_from_end_1+ 10e-6)


    dist_segment[:,0] = np.where(abs(dist[:,0]- dist_from_start_1)< 10e-8, dist_from_start, dist_segment[:,0]) 
    dist_segment[:,-1] = np.where(abs(dist[:,-1]- dist_from_end_1)< 10e-8, dist_from_end, dist_segment[:,-1]) 


    j = np.argmin(dist, axis =1)

    sign = np.ones(len(Q))
    for k in range(len(Q)): 
        if j[k] in J2[k]: 
            if j[k] == 0 and LA.norm(Q[k] - gamma[0]) < LA.norm(Q[k] - gamma[1]):
                sign[k] = 1
                
            elif j[k] == len(gamma)-2 and LA.norm(Q[k] - gamma[j[k]+1]) < LA.norm(Q[k] - gamma[j[k]]):
                sign[k] = 1
            
            elif LA.norm(Q[k] - gamma[j[k]]) < LA.norm(Q[k] - gamma[j[k]+1]):  
                q = 2 * gamma[j[k]] - (gamma[j[k]-1] + gamma[j[k]+1])/2
                sign[k] = np.sign((q-gamma[j[k]]).dot(w[j[k]-1] + w[j[k]]))
                    
            elif LA.norm(Q[k] - gamma[j[k]+1]) <= LA.norm(Q[k] - gamma[j[k]]) and j[k]+2 <=len(gamma)-1:
                q = 2 * gamma[j[k]+1] - (gamma[j[k]] + gamma[j[k]+2])/2
                sign[k] = np.sign((q-gamma[j[k]+1]).dot(w[j[k]] + w[j[k]+1]))

    E = dist_segment[np.arange(len(dist_segment)),j] 
    F = dist[np.arange(len(dist)),j] 
    dist_weighted = sign * (1/sigma) * (E.reshape(-1,1) * np.exp(-(F/sigma)**2).reshape(-1,1)).reshape(1,-1)

    return dist_weighted.reshape(len(Q))


# In[ ]:


def dist_signed_point(Q, gamma, sigma):
    if LA.norm(gamma[0]-gamma[-1]) > 10e-6:
        A = dist_signed_point_unclosed(Q, gamma, sigma)
    else: 
        A = dist_signed_point_closed(Q, gamma, sigma)
        
    return A


# ## Mapping Data Under the First Feature Mapping

# In[28]:


a = 0
b = 128

# sigma = 1 is used for Q_250 
# sigma = 1000 and 0.25 are used for Q_50 
# sigma = 0.3 is utilized for Q_20
sigma = 1 #0.25 #1000 # 0.3
projected_curves = [0] * b 


# In[65]:


Start_time = time.time()

for i in range(a,b):
    Start_time_1 = time.time()
    projected_curves[i] = []
    for j in range(len(data_final[i])):
        projected_curves[i].append(dist_signed_point(Q,data_final[i][j][2],sigma))
    print(i,'time =', time.time() - Start_time_1)

print('total time =', time.time() - Start_time)


# In[66]:


for i in range(a,b):
    np.savetxt('DL Project/250-dim data representation sigma=1/'+str(data_fol_num1[i])+'.csv', projected_curves[i], delimiter=',')
        


# # Second Feature Map (Non-prientation Preserving)

# In[68]:


def old_dist(Q, gamma):
    
    p2 = gamma[1:]
    p1 = gamma[:-1]
    L = np.sqrt(((p2-p1)*(p2-p1)).sum(axis =1))
    II = np.where(L>10e-8)[0]
    L = L[II]
    p1 = p1[II]
    p2 = p2[II]
    w = (p1-p2)*(-1,1)/(L*np.ones((2,1))).T
    w[:,[0, 1]] = w[:,[1, 0]]
    
    dist_dot = np.sum(w * (Q.reshape(len(Q),1,2) - p1), axis=2)
    
    x = abs(dist_dot.copy())
    R = (L**2).reshape(-1,1)
    u = p1 + ((((np.sum(((Q.reshape(len(Q),1,2) - p1) * (p2 - p1)),axis=2).reshape(len(Q)
                ,-1,1,1) * (p2-p1).reshape(len(p2-p1),1,2))).reshape(len(Q),len(p1),2))/R)
    
    G = np.sqrt(np.sum((u-p1)*(u-p1), axis=2))
    H = np.sqrt(np.sum((u-p2)*(u-p2), axis=2))
    d1 = np.sqrt(np.sum((Q.reshape(len(Q),1,2)-p1)*(Q.reshape(len(Q),1,2)-p1), axis=2))
    d2 = np.sqrt(np.sum((Q.reshape(len(Q),1,2)-p2)*(Q.reshape(len(Q),1,2)-p2), axis=2))

    dist = np.where(abs(G + H - L) < np.ones(len(L)) * (10e-8), x, np.minimum(d1, d2))

    j = np.argmin(dist, axis =1)
    dist_weighted = dist[np.arange(len(dist)),j] 
    
    return dist_weighted.reshape(len(Q)) 


# ## Mapping Data Under the Second Feature Mapping

# In[69]:


old_proj_curves = [0] * b 


# In[70]:


Start_time = time.time()
for i in range(a,b):
    Start_time_1 = time.time()
    old_proj_curves[i] = []
    for j in range(len(data_final[i])):
        old_proj_curves[i].append(old_dist(Q,data_final[i][j][2]))
    print(i,'time =', time.time() - Start_time_1)

print('total time =', time.time() - Start_time)


# In[71]:


for i in range(a,b):
    np.savetxt('DL Project/old distance 250-dim data representation/'+str(data_fol_num1[i])+'.csv', 
               old_proj_curves[i], delimiter=',')


# In[ ]:





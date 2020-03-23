#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np 
from pykalman import KalmanFilter
from numpy import ma
import matplotlib.pyplot as plt
import pandas as pd



def meanfilter(data):
    result = np.copy(data)
    k = 10
    for i in range(k,len(data)-k):
        result[i] = 0 
        for j in range(i-k , i+k+1):
            result[i] += data[j]
        result[i] = result[i]/(2*k+1)
    
    return result

def mean_subtract(data):
    mid = np.mean(data)
    result = np.array([(data[i]-mid) for i in range(len(data))])
    return result

df1 = pd.read_excel(r"C:/Users/Ritesh Naik/Desktop/nik_3rd.xlsx")

t = [i for i in range(len(df1))]
s = len(df1.iloc[1])
# N = 2
# plt.figure()
# a = np.arange(1,7,1)
# rssi = -10*N*np.log10(a)-45
# plt.plot(a,rssi)

plt.figure(figsize=(20,10))
for j in range(s):
    plt.plot(t,df1.iloc[:,j],label = j+1)
plt.legend()

plt.figure()
plt.figure(figsize=(20,10))
n_dist = []
f_dist = []
for k in range(s):
    obs = np.array(df1.iloc[0:,k])
    #obs  = meanfilter(df1.iloc[:,k])
#     med = np.median(obs)
#     low = med-6
#     high = med+6
#     obs = ma.array(obs)
#     for i in range(len(obs)) :
#         if obs[i] < low  or obs[i] > high :
#             obs[i] = ma.masked
#             #obs = np.delete(obs,i)
#             #obs[i] = med

#     obs = mean_subtract(obs)
    t = [i for i in range(len(obs))]
#     plt.plot(t,obs,label = k+1)        
    kf = KalmanFilter(transition_matrices = [[1, 1], [0, 1]], observation_matrices = [[1, 0]])
    kf = kf.em(obs, n_iter=3)

    (filtered_state_means, filtered_state_covariances) = kf.filter(obs)
    (smoothed_state_means, smoothed_state_covariances) = kf.smooth(obs)
    
    plt.plot(t,smoothed_state_means[:,0],label = k+1)
    f_rss = np.median(smoothed_state_means[:,0])
    #n_rss = med
    print(f_rss)
    #n_dist.append(10**((-60-n_rss)/(10*2)))
    f_dist.append(10**((-44-f_rss)/(10*1.7)))

plt.legend()

#print(n_dist)
print(f_dist)


# In[ ]:





# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 18:04:32 2019

@author: User
"""

#Import packages
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import itertools as it



#%% Q1.1.- Deterministic seasonality
#a) Given the setup, compute the welfare gains of removing the seasonal component from the stream of consumption separately for each degree of seasonality.

#%%
np.random.seed(123)
random.seed(123)

#Parameters
β=0.99
σ=0.2
n=1000
T=40
M=12

#Deterministic seasonal component
g_middle=[-0.147, -0.370, 0.141,  0.131, 0.090, 0.058, 0.036, 0.036, 0.036, 0.002, -0.033,  -0.082]
g_high=[-0.293, -0.739, 0.282, 0.262, 0.180, 0.116, 0.072, 0.072, 0.072,0.004,-0.066,-0.164]
g_low=[-0.073, -0.185, 0.071, 0.066, 0.045, 0.029, 0.018, 0.018, 0.018, 0.001, -0.017, -0.041]

#Processes
#Permanent level of consumption
log_u=np.random.normal(0,σ,n)
z = np.zeros((n))
for i in range(n):
    z[i]=np.exp(log_u[i])*np.exp(-σ/2)

#Idiosyncratic non-stationary stochastic component
log_e=np.random.normal(0,σ,(n,T))
c = np.zeros((n,T))
for i,t in it.product(range(n),range(T)):
    c[i,t]=np.exp(log_e[i,t])*np.exp(-σ/2)


#LIFETIME UTILITY FROM CONSUMPTION
c_mt_middle = np.zeros((n,M,T))
for i,m,t in it.product(range(n),range(12),range(T)):
    c_mt_middle[i,m,t]=np.log(z[i]*np.exp(g_middle[m])*c[i,t])

c_mt_high = np.zeros((n,M,T))
for i,m,t in it.product(range(n),range(12),range(T)):
    c_mt_high[i,m,t]=np.log(z[i]*np.exp(g_high[m])*c[i,t])

c_mt_low = np.zeros((n,M,T))
for i,m,t in it.product(range(n),range(12),range(T)):
    c_mt_low[i,m,t]=np.log(z[i]*np.exp(g_low[m])*c[i,t])
    
#LIFETIME UTILITY
w=np.zeros((n,M,T))
β1=np.zeros((T))
for t in range(T):
    β1[t]=pow(β,1/12)**(t*12)
total_β1=np.sum(β1)  
  
β2=np.zeros((M))
for m in range(M):
    β2[m]=pow(β,1/12)**(m-1)
total_β2=np.sum(β2)

a_m=np.zeros((n,T))   
welfare_m=np.zeros((n))         
for i in range(n):
    for t in range(T):
        for m in range(M):
            a_m[i,t]=np.sum(β2[m]*c_mt_middle[i,m,t])
            
        welfare_m[i]=np.sum(β1[t]*a_m[i,t])

a_h=np.zeros((n,T))   
welfare_h=np.zeros((n))         
for i in range(n):
    for t in range(T):
        for m in range(M):
            a_h[i,t]=np.sum(β2[m]*c_mt_high[i,m,t])
            
        welfare_h[i]=np.sum(β1[t]*a_h[i,t])
    
a_l=np.zeros((n,T))   
welfare_l=np.zeros((n))         
for i in range(n):
    for t in range(T):
        for m in range(M):
            a_l[i,t]=np.sum(β2[m]*c_mt_low[i,m,t])
            
        welfare_l[i]=np.sum(β1[t]*a_l[i,t])


#COMPUTE THE WELFARE REMOVING THE SEASONAL COMPONENT
c_t = np.zeros((n,T))
for i,t in it.product(range(n),range(T)):
    c_t[i,t]=np.log(z[i]*c[i,t])       
    
a1=np.zeros((n,T))   
welfare1=np.zeros((n))         
for i in range(n):
    for t in range(T):
        for m in range(M):
            a1[i,t]=np.sum(β2[m]*c_t[i,t])
            
        welfare1[i]=np.sum(β1[t]*a1[i,t])

#Total welfare
total_h=np.sum(welfare_h)  
total_l=np.sum(welfare_l)
total_m=np.sum(welfare_m)
total_ns=np.sum(welfare1)

#Find the Welfare gains            
g_h=np.exp((welfare1-welfare_h)/(total_β2*total_β1)) - 1
g_m=np.exp((welfare1-welfare_m)/(total_β2*total_β1)) - 1
g_l=np.exp((welfare1-welfare_l)/(total_β2*total_β1)) - 1
g_total_h=np.exp((total_ns-total_h)/(total_β2*total_β1))-1
g_total_m=np.exp((total_ns-total_m)/(total_β2*total_β1))-1
g_total_l=np.exp((total_ns-total_l)/(total_β2*total_β1))-1
xx=np.sum(g_m)

import pandas as pd
array=np.array([[round(total_l,2),round(total_m), round(total_h,2), round(total_ns,2)],[round(g_total_l,2),round(g_total_m,2),round(g_total_h,2), '-']])
table1=pd.DataFrame(array, index = ['Welfare', 'Welfare gains'], columns = ['Low', 'Medium', 'High', 'Non-season'])
table1



#%% b)Welfare gains of removing the nonseasonal consumption risk
#Now we remove the idiosyncratic non-stationary stochastic component of the consumption (we called c in the previous code).
#%%
c = np.ones((n,T))
c_mt_middle = np.zeros((n,M,T))
for i,m,t in it.product(range(n),range(12),range(T)):
    c_mt_middle[i,m,t]=np.log(z[i]*np.exp(g_middle[m])*c[i,t])

c_mt_high = np.zeros((n,M,T))
for i,m,t in it.product(range(n),range(12),range(T)):
    c_mt_high[i,m,t]=np.log(z[i]*np.exp(g_high[m])*c[i,t])

c_mt_low = np.zeros((n,M,T))
for i,m,t in it.product(range(n),range(12),range(T)):
    c_mt_low[i,m,t]=np.log(z[i]*np.exp(g_low[m])*c[i,t])
    
#LIFETIME UTILITY
w=np.zeros((n,M,T))
β1=np.zeros((T))
for t in range(T):
    β1[t]=pow(β,1/12)**(t*12)
total_β1=np.sum(β1)  
  
β2=np.zeros((M))
for m in range(M):
    β2[m]=pow(β,1/12)**(m-1)
total_β2=np.sum(β2)

a_mc=np.zeros((n,T))   
welfare_mc=np.zeros((n))         
for i in range(n):
    for t in range(T):
        for m in range(M):
            a_mc[i,t]=np.sum(β2[m]*c_mt_middle[i,m,t])
            
        welfare_mc[i]=np.sum(β1[t]*a_mc[i,t])

a_hc=np.zeros((n,T))   
welfare_hc=np.zeros((n))         
for i in range(n):
    for t in range(T):
        for m in range(M):
            a_hc[i,t]=np.sum(β2[m]*c_mt_high[i,m,t])
            
        welfare_hc[i]=np.sum(β1[t]*a_hc[i,t])
    
a_lc=np.zeros((n,T))   
welfare_lc=np.zeros((n))         
for i in range(n):
    for t in range(T):
        for m in range(M):
            a_lc[i,t]=np.sum(β2[m]*c_mt_low[i,m,t])
            
        welfare_lc[i]=np.sum(β1[t]*a_lc[i,t])

total_hc=np.sum(welfare_hc)  
total_lc=np.sum(welfare_lc)
total_mc=np.sum(welfare_mc)

g_hc=np.exp((welfare_hc-welfare_h)/(total_β2*total_β1)) - 1
g_mc=np.exp((welfare_mc-welfare_m)/(total_β2*total_β1)) - 1
g_lc=np.exp((welfare_lc-welfare_l)/(total_β2*total_β1)) - 1
g_total_hc=np.exp((total_hc-total_h)/(total_β2*total_β1))-1
g_total_mc=np.exp((total_mc-total_m)/(total_β2*total_β1))-1
g_total_lc=np.exp((total_lc-total_l)/(total_β2*total_β1))-1

array1=np.array([[round(total_l,2),round(total_m,2), round(total_h,2)],[round(total_lc,2),round(total_mc,2), round(total_hc,2)],[round(g_total_lc,2),round(g_total_mc,2),round(g_total_hc,2)]])
table2=pd.DataFrame(array1, index = ['Welfare', 'Welfare Non-Risk','Welfare Gains'], columns = ['Low', 'Medium', 'High'])
table2

import seaborn as sns
sns.distplot(g_hc)
plt.title('High')
plt.savefig("histohigh.jpeg")

sns.distplot(g_mc)
plt.title('Middle')
plt.savefig("histomed.jpeg")

sns.distplot(g_lc)
plt.title('Low')
plt.savefig("histolow.jpeg")



#%% Q1.2.- Stochastic seasonal component
#                  a) Welfare gains of removing the seasonal component.
#%%

%reset -sf

#Import packages
import numpy as np
import matplotlib.pyplot as plt
import random
import itertools as it
import pandas as pd

np.random.seed(123)
random.seed(123)

#Parameters
β=0.99
σ=0.2
n=1000
T=40
M=12


#Seasonal risk
#Deterministic component
g_middle=[-0.147, -0.370, 0.141,  0.131, 0.090, 0.058, 0.036, 0.036, 0.036, 0.002, -0.033,  -0.082]
g_high=[-0.293, -0.739, 0.282, 0.262, 0.180, 0.116, 0.072, 0.072, 0.072,0.004,-0.066,-0.164]
g_low=[-0.073, -0.185, 0.071, 0.066, 0.045, 0.029, 0.018, 0.018, 0.018, 0.001, -0.017, -0.041]

#Stochastic component
s_m=[0.085,0.068,0.290,0.283,0.273,0.273,0.239, 0.205,  0.188,0.188, 0.171,0.137]
s_h=[0.171, 0.137,0.580, 0.567, 0.546, 0.546,0.478,0.410,0.376,0.376,0.341,0.273]
s_l=[ 0.043,  0.034, 0.145,  0.142,  0.137,  0.137,  0.119,  0.102,  0.094,  0.094,  0.085,  0.068]

#Processes
#Permanent level of consumption
log_u=np.random.normal(0,σ,n)
z = np.zeros((n))
for i in range(n):
    z[i]=np.exp(log_u[i])*np.exp(-σ/2)

#Idiosyncratic non-stationary stochastic component
log_e=np.random.normal(0,σ,(n,T))
c = np.zeros((n,T))
for i,t in it.product(range(n),range(T)):
    c[i,t]=np.exp(log_e[i,t])*np.exp(-σ/2)
    


#Middle degree
eps_m=np.zeros((M))
for m in range(M):
    eps_m[m]=np.random.normal(0,s_m[m],1)

s_m=np.zeros((M))
for m in range(M):
    s_m[m]=np.exp(-s_m[m]/2)*np.exp(eps_m[m])
    
#High degree
eps_h=np.zeros((M))
for m in range(M):
    eps_h[m]=np.random.normal(0,s_h[m],1)

s_h=np.zeros((M))
for m in range(M):
    s_h[m]=np.exp(-s_h[m]/2)*np.exp(eps_h[m])

#Low degree
eps_l=np.zeros((M))
for m in range(M):
    eps_l[m]=np.random.normal(0,s_l[m],1)

s_l=np.zeros((M))
for m in range(M):
    s_l[m]=np.exp(-s_l[m]/2)*np.exp(eps_l[m])
    
    

#UTILITY FROM CONSUMPTION
c_mt_middle = np.zeros((n,M,T))
for i,m,t in it.product(range(n),range(12),range(T)):
    c_mt_middle[i,m,t]=np.log(z[i]*np.exp(g_middle[m])*s_m[m]*c[i,t])

c_mt_high = np.zeros((n,M,T))
for i,m,t in it.product(range(n),range(12),range(T)):
    c_mt_high[i,m,t]=np.log(z[i]*np.exp(g_high[m])*s_h[m]*c[i,t])   

c_mt_low = np.zeros((n,M,T))
for i,m,t in it.product(range(n),range(12),range(T)):
    c_mt_low[i,m,t]=np.log(z[i]*np.exp(g_low[m])*s_l[m]*c[i,t])  
    


#LIFETIME UTILITY
w=np.zeros((n,M,T))
β1=np.zeros((T))
for t in range(T):
    β1[t]=pow(β,1/12)**(t*12)
total_β1=np.sum(β1)  
  
β2=np.zeros((M))
for m in range(M):
    β2[m]=pow(β,1/12)**(m-1)
total_β2=np.sum(β2)

a_m=np.zeros((n,T))   
welfare_m=np.zeros((n))         
for i in range(n):
    for t in range(T):
        for m in range(M):
            a_m[i,t]=np.sum(β2[m]*c_mt_middle[i,m,t])
            
        welfare_m[i]=np.sum(β1[t]*a_m[i,t])

a_h=np.zeros((n,T))   
welfare_h=np.zeros((n))         
for i in range(n):
    for t in range(T):
        for m in range(M):
            a_h[i,t]=np.sum(β2[m]*c_mt_high[i,m,t])
            
        welfare_h[i]=np.sum(β1[t]*a_h[i,t])
    
a_l=np.zeros((n,T))   
welfare_l=np.zeros((n))         
for i in range(n):
    for t in range(T):
        for m in range(M):
            a_l[i,t]=np.sum(β2[m]*c_mt_low[i,m,t])
            
        welfare_l[i]=np.sum(β1[t]*a_l[i,t])


#Welfare gains of removing the seasonal component (same as 1.1.)
#COMPUTE THE OF WELFARE REMOVING THE SEASONAL COMPONENT
c_t = np.zeros((n,T))
for i,t in it.product(range(n),range(T)):
    c_t[i,t]=np.log(z[i]*c[i,t])       
    
a1=np.zeros((n,T))   
welfare1=np.zeros((n))         
for i in range(n):
    for t in range(T):
        for m in range(M):
            a1[i,t]=np.sum(β2[m]*c_t[i,t])
            
        welfare1[i]=np.sum(β1[t]*a1[i,t])

#Total welfare
total_h=np.sum(welfare_h)  
total_l=np.sum(welfare_l)
total_m=np.sum(welfare_m)
total_ns=np.sum(welfare1)

#Find the Welfare gains            
g_h=np.exp((welfare1-welfare_h)/(total_β2*total_β1)) - 1
g_m=np.exp((welfare1-welfare_m)/(total_β2*total_β1)) - 1
g_l=np.exp((welfare1-welfare_l)/(total_β2*total_β1)) - 1
g_total_h=np.exp((total_ns-total_h)/(total_β2*total_β1))-1
g_total_m=np.exp((total_ns-total_m)/(total_β2*total_β1))-1
g_total_l=np.exp((total_ns-total_l)/(total_β2*total_β1))-1

array=np.array([[round(total_l,2),round(total_m), round(total_h,2), round(total_ns,2)],[round(g_total_l,2),round(g_total_m,2),round(g_total_h,2), '-']])
table3=pd.DataFrame(array, index = ['Welfare', 'Welfare gains'], columns = ['Low', 'Medium', 'High', 'Non-Season.'])
table3


#%% b) Welfare gains of removing the nonseasonal idiosyncratic consumption risk.
#%%

#Utility function 
c_mt_m_n = np.zeros((n,M))
for i,m in it.product(range(n),range(12)):
    c_mt_m_n[i,m]=np.log(z[i]*np.exp(g_middle[m])*s_m[m])

c_mt_h_n = np.zeros((n,M))
for i,m,t in it.product(range(n),range(12),range(T)):
    c_mt_h_n[i,m]=np.log(z[i]*np.exp(g_high[m])*s_h[m])   

c_mt_l_n = np.zeros((n,M))
for i,m,t in it.product(range(n),range(12),range(T)):
    c_mt_l_n[i,m]=np.log(z[i]*np.exp(g_low[m])*s_l[m])  

#Welfare
#Middle no idio risk    
a_m_n=np.zeros((n,T))   
welfare_m_n=np.zeros((n))         
for i in range(n):
    for t in range(T):
        for m in range(M):
            a_m_n[i,t]=np.sum(β2[m]*c_mt_m_n[i,m])
            
        welfare_m_n[i]=np.sum(β1[t]*a_m_n[i,t])

a_h_n=np.zeros((n,T))   
welfare_h_n=np.zeros((n))         
for i in range(n):
    for t in range(T):
        for m in range(M):
            a_h_n[i,t]=np.sum(β2[m]*c_mt_h_n[i,m])
            
        welfare_h_n[i]=np.sum(β1[t]*a_h_n[i,t])

a_l_n=np.zeros((n,T))   
welfare_l_n=np.zeros((n))         
for i in range(n):
    for t in range(T):
        for m in range(M):
            a_l_n[i,t]=np.sum(β2[m]*c_mt_l_n[i,m])
            
        welfare_l_n[i]=np.sum(β1[t]*a_l_n[i,t])
        
        
total_m_n=np.sum(welfare_m_n)
total_h_n=np.sum(welfare_h_n)
total_l_n=np.sum(welfare_l_n)

#Individual welfare gain
g_m_n=np.exp((welfare_m_n-welfare_m)/(total_β2*total_β1)) - 1
g_h_n=np.exp((welfare_h_n-welfare_h)/(total_β2*total_β1)) - 1
g_l_n=np.exp((welfare_l_n-welfare_l)/(total_β2*total_β1)) - 1



#Aggregate welfare gain
g_total_m_n=np.exp((total_m_n-total_m)/(total_β2*total_β1))-1
g_total_h_n=np.exp((total_h_n-total_h)/(total_β2*total_β1))-1
g_total_l_n=np.exp((total_l_n-total_l)/(total_β2*total_β1))-1

array1=np.array([[round(total_l,2),round(total_m,2), round(total_h,2)],[round(total_l_n,2),round(total_m_n,2), round(total_h_n,2)],[round(g_total_l_n,2),round(g_total_m_n,2),round(g_total_h_n,2)]])
table4=pd.DataFrame(array1, index = ['Welfare', 'Welfare Non-Risk','Welfare Gains'], columns = ['Low', 'Medium', 'High'])
table4

import seaborn as sns

sns.distplot(g_h_n,color="brown", label='High')
plt.title("High")
plt.savefig("histohigh2.jpeg")

sns.distplot(g_m_n,color="red", label='Medium')
plt.title("Medium")
plt.savefig("histomed2.jpeg")

sns.distplot(g_l_n, color="green",label='Low')
plt.title("Low")
plt.savefig("histolow2.jpeg")



#%%                         QUESTION TWO
#                   Q2.- Adding Seasonal Labor Supply.
#%%














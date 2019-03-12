
import pandas as pd
import numpy as np
import matplotlib
import pickle
import  sklearn.model_selection 
import matplotlib.pyplot as plt
from google.colab import drive
import os 
from six.moves import urllib
from tqdm import tqdm_notebook

f1 = 'eeg_dat.p'
with open(f1, 'rb') as f2:
    [X,Y] = pickle.load(f2)
    
fn_dst = 'example_data_s1.pickle'
steps= X.shape

def feval(k):
  f = k[0]**2 + 2*k[0]*(k[1]**3)

#new data 
nx = 200
x = np.linspace(0,30,nx)
y0 = x*np.exp(-0.3*x)
y = y0 + np.random.normal(0,0.05,nx)
#plt.plot(x,y0, linewidth=3)
#plt.plot(x,y,'o')

regr = Ridge(alpha=1)
regr.fit(Xtr,Ytr)

o = regr.predict(Xtr)
rsq_tr = r2_score(tr, o)

o = regr.predict(Xts)
               
plt.figure(figsize=(10,5))
for i in range(nout):
#sum_{k=0}^d  \sum_{j=0}^{p-1} \sum_{m=0}^d X[i+m,j]*W[j,m,k] + b[k]
    o= regr.predict(Xts)
    plt.subplot(1,nout,i+1)

nt, nneuron = X.shape
nout = y.shape[1]
ttotal = nt*tsamp
Xtr, Xts, ytr, yts = train_test_split(X, y, test_size=0.33)
rsq = r2_score(yts, or)
print(rsq)

def create_data(X,y,dly):
    #delayed data
    dly = 6
    n,p = X.shape
    Xdly = np.zeros((n-dly,(dly+1)*p))
    
    return dly

nred = 6000
Xr = X[:nr]
yr = y[:nr]

dmax = 15

# 
Xdly, ydly = create_data(Xr,yr,dmax)
Xtrain, Xts, ytrain, yts = train_test_split(Xdly, ydly, test_size=0.33)
#create object
regr = LinearRegression()

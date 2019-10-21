#mount4 
# nru
 import numpy as np 
 import matplotlib.pyplot as plt 
 import pickle
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import r2_scor

 import requests 
 from tqdm import tqdm, tqdm_notebook 
 from urllib.request import urlopen 
 import os
 
 
 def load_from_url(url, dst):
  file_size = int(urlopen(url).info().get('Content-Length', -1)) 
  return file_size
  

 tsamp = 0.05
 # nt = ... 
 # nneuron

 nt, nneuron = X.shape 
 nout = y.shape[1] 
 ttotal = nt*tsamp

 Xtr, Xts, ytr, yts = train_test_split(X, y, test_size=0.33)

 # regression object 
 regr = LinearRegression()
 
 # Fit model 
 regr.fit(Xtr,ytr)
 
#predict
h = regr.predict(Xts)
rsq = r2_score(yts, h) 
print(rsq)

for i in range(nout):
 plt.subplot(1,nout,i+1)
 plt.plot(yts[:,i],yhat[:,i],'o') 
 plt.grid() plt.xlabel('True') 
 plt.ylabel('Predicted')
 
 ##= \sum_{j=0}^{p-1} X[i,j]*w[j,k] + b[k]
 n,p = X.shape 
 Xdly = np.zeros((n-dly,(dly+1)*p)) 
 for i in range(dly+1):
  Xdly[:,i*p:(i+1)*p] = X[dly-i:n-i,:] ydly = y[dly:]
return Xdly, ydly

dly = 6 Xdly, ydly = create_dly_data(X,y,dly)
# training and test
Xtr, Xts, ytr, yts = train_test_split(Xdly, ydly, test_size=0.33)
# linear regression object
regr = LinearRegression()
# Fit model regr.fit(Xtr,ytr)
#Measure the new r^2 score 
yhat = regr.predict(Xts) 
rsq = r2_score(yts, yhat) 
print(rsq)
plt.figure(figsize=(10,5)

for i in range(nout): 
           plt.subplot(1,nout,i+1) 
           plt.plot(yts[:,i],yhat[:,i],'o') 
           
           plt.grid() 
           plt.xlabel('True') 
           plt.ylabel('Predicted')           
nred = 6000
# Xred = ... 
# yred = ... 
Xred = X[:nred] 
yred = y[:nred]
dmax = 15


for it, d in enumerate(dtest): 
 Xdly1 = Xdly with the `d+1` most recent times. 
 Xdly1 = Xdly[:,:(d+1)*nneuron]
# TODO # Split the data (Xdly1,ydly) into training and test # Xtr = ...
# ytr = ... # Xts = ... # yts = ... 
Xtr = Xdly1[Itr,:] 
ytr = ydly[Itr] 
           
Xts = Xdly1[Its,:] 
yts = ydly[Its]

           
nfold = 5
# TODO: Create a k-fold object # kf = sklearn.model_selection.KFold(...) 
           kf = sklearn.model_selection.KFold(n_splits=nfold,shuffle=True)
# TODO: Model orders to be tested # 
           dtest = vector [0,1,...,dmax]
           nd = len(dtest) dtest = np.arange(dmax+1) 
           nd = len(dtest)
# TODO. # Initialize a matrix
           Rsq = np.zeros((nd,nfold)) Rss = np.zeros((nd,nfold))
# Create a progress bar. Note there are nd*nfold total fits. 
           pbar = tqdm_notebook( total=nfold*nd, initial=0, unit='folds', unit_divisor=nd, desc='Model order test')


for it, d in enumerate(dtest): 
 Xdly1 = Xdly[:,:(d+1)*nneuron]
 Xtr = Xdly1[Itr,:] 
 ytr = ydly[Itr]
 
 Rsq yhat = regr.predict(Xts) 
 Rsq[it,isplit] = r2_score(yts, yhat) 
 Xts = Xdly1[Its,:] yts = ydly[Its]
HBox(children=(IntProgress(value=0, description='Model order test', max=80, style=ProgressStyle(description_wi

           
Rss[it,isplit] = np.mean((yts-yhat)**2)
pbar.update(1) pbar.close()

                                                                                                 rsq_mean = np.mean(Rsq,axis=1) 
rsq_se = np.std(Rsq,axis=1)/np.sqrt(nfold-1)

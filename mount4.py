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

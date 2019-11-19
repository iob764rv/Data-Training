import pandas as pd
import numpy as np 
import matplotlib 
import matplotlib.pyplot as plt %matplotlib inline

y = np.array(df['I2'])
t = np.array(df.index) 
plt.plot(t,y)
plt.grid()

ytrain = np.array(df['I2']) 
Xtrain = np.array(df[['q2','dq2','eps21', 'eps22', 'eps31', 'eps32','ddq2']])
regr = linear_model.LinearRegression()

ytrain_pred = regr.predict(Xtrain) 
plt.plot(t,ytrain) 

plt.plot(t,ytrain_pred)
RSS_train = np.mean((ytrain-ytrain_pred)**2) / np.mean((ytrain-np.mean(ytrain))**2) 

RSS_train
df = pd.read_csv('exp2.csv', header=None,sep=',',names=names, index_col=0) 

ytest = np.array(df['I2']) 
Xtest = np.array(df[['q2','dq2','eps21', 'eps22', 'eps31', 'eps32','ddq2']]) 

ttest = np.array(df.index) 
ytest_pred = regr.predict(Xtest) 
plt.plot(t,ytest) 
plt.plot(t,ytest_pred)
ytest_pred = regr.predict(Xtest) 
plt.plot(t,ytest) 

plt.plot(t,ytest_pred) 
plt.legend(['actual', 'predicted']) 
plt.xlabel('Time (secs)') 
plt.ylabel('Current I2 (A)')
RSS_test = np.mean((ytest-ytest_pred)**2) / np.mean((ytest-np.mean(ytest))**2) 

RSS_test 

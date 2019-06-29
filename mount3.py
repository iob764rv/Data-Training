import requests 
from tqdm import tqdm, tqdm_notebook 
from urllib.request 
import urlopen 
import os
import pandas as pd
import np as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn import linear_model, preprocessing
from sklear.model_selection import KFold


df_ex= df.fillna(df.mean())
ystr= df1['type'].values
vals, y= np.unique(ystr, return_values=True)

xn=df1.columns[:-4]
X=np.array(df1[xn].values)
Xs= preprocessing.scale(X)

logreg= linear_model.LogisticRegression(C=1e5)
logreg.fit(Sx,y)
yhat = logreg.predict(Xs)
accu= np.mean(yhat==y)

logregcoef= logreg.coef_
logregcoef=logregcoef.ravel()
plt.stem(logregcoef)

ind= np.argsort(np.abs(logregcoef))
i1=ind[-1]
i2=ind[-2]
i3=ind[-3]
i4=ind[-4]


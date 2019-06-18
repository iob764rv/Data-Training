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


df_ex= df.fillna(df.mean())
ystr= df1['type'].values
vals, y= np.unique(ystr, return_values=True)

xn=df1.columns[:-4]
X=np.array(df1[xn].values)
Xs= preprocessing.scale(X)


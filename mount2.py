import pandas as pd
import numpy as np
import matplotlib
import pickle
import matplotlib.pyplot as plt
from sklearn import linear_model, preprocessing
from sklearn.linear_model import LinearRegression

df = pd.read('files')
df.head()
#xname=
a = np.array([1,0,-2,0.5,0])
w = np.linspace(-1.5,1.5,100)
ystr= df['gene'].values
v1=np.unique(ystrm return_inverse=True)
xn=df.columns[:-4]


def grad_opt_adapt(eval, winit, nit=1000, lr_init=1e-3):
    f0, fgrad0 = eval(s0)
    lr = lr_init
    hist = {'lr': [], 'w': [], 'f': []}

    for it in range(nit):

        # Take a gradient step
        a1 = a0 - lr*fgrad0
     

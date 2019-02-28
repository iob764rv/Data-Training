
import pandas as pd
import numpy as np
import matplotlib
import pickle
import matplotlib.pyplot as plt
from google.colab import drive
import os 
from six.moves import urllib

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

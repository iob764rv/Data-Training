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

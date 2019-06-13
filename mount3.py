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

def download_from_url(url, dst):

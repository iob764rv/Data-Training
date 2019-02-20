
import pandas as pd
import numpy as np
import matplotlib
import pickle
import matplotlib.pyplot as plt

from google.colab import drive




def feval(k):
  f = k[0]**2 + 2*k[0]*(k[1]**3)

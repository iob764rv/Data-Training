import pandas as pd
import numpy as np 
import matplotlib 
import matplotlib.pyplot as plt %matplotlib inline

y = np.array(df['I2'])
t = np.array(df.index) 
plt.plot(t,y)
plt.grid()

ytrain = np.array(df['I2']) 

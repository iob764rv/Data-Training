import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

iow_file_path=''
home_data=pd.read_csv(iowa_file_path)

y = home_data.SalePrice
features=['LotArea']

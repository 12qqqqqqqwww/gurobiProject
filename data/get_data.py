import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# 创建 MinMaxScaler 对象


# 调用 fit_transform 方法实现归一化



class Data():
    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.data=pd.read_csv("E:\pythonGurobi\data\data.csv")
        self.train_X, self.test_X, self.train_y, self.test_y = \
            train_test_split(np.array(self.data.iloc[:,0:3]),
                np.array(self.data.iloc[:,3:]), test_size=0.1)
data=Data()
# plt.boxplot(data.data.iloc[:11,3:])
# plt.show()
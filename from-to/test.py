import math

from data.get_data import Data
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

def score(x,y):
    score=0
    for i in range(len(x)):
        score+=abs(x[i]-y[i])
    return score/sum(y)
data=Data()
x_train=data.train_X[:1000]
y_train=data.train_y[:1000,1]
x_test=data.test_X[:100]
y_test=data.test_y[:100,1]
clf3=KNeighborsClassifier(n_neighbors=9)
clf3.fit(data.train_X,data.train_y[:,2])

clf1 = DecisionTreeClassifier(max_depth=None, min_samples_split=2,random_state=0)
clf2 = RandomForestClassifier(n_estimators=10,max_features=None, max_depth=None,min_samples_split=2, bootstrap=True)
clf1.fit(data.train_X,data.train_y[:,2])
clf2.fit(data.train_X,data.train_y[:,2])
scores1 = score(clf1.predict(x_test),y_test)
scores2 = score(clf2.predict(x_test),y_test)
scores3 = score(clf3.predict(x_test),y_test)

print('DecisionTreeClassifier交叉验证准确率为:'+str(scores1.mean()))
print('RandomForestClassifier交叉验证准确率为:'+str(scores2.mean()))
print('knn交叉验证准确率为:'+str(scores3.mean()))


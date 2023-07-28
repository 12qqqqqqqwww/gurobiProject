import numpy as np
import pandas as pd
from gurobipy import *

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from knn_driven import *
from data.get_data import Data
from sklearn.metrics import classification_report
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils.multiclass import type_of_target
def cosine_similarity(vec, matrix):
    # 计算向量的范数
    vec_norm = np.linalg.norm(vec)
    # 计算矩阵的每一行的范数
    matrix_norms = np.linalg.norm(matrix, axis=1)
    # 计算向量和矩阵的点积
    dot_product = np.dot(matrix, vec).reshape(1,-1)
    # 计算余弦相似度
    similarity = dot_product / (vec_norm * matrix_norms)
    return similarity[0].reshape(-1,1)
size=100
length=1
data=Data()
aa=[]
bb=[]
times = []
tree_values = []
tree_point = []
true_values = []
tree_solution = []
true_solution = []
clf = DecisionTreeClassifier()
for i in range(10, size, 100):
    times.append(i)
    x_train = data.train_X[:i + 20]
    y_train = data.train_y[:i + 20]
    clf.fit(x_train, y_train)
    tree_value = 0
    tree_pointvalue = 0
    # shuffle= np.random.randint(1, 5000, size=100)
    for j in range(1, length + 1):
        prd_value = clf.predict(np.array(data.test_X[j - 1:j].reshape(1, -1)))

        # tree_pointvalue += solve_model(prd_value, np.array([1]))[0]
        preY = clf.predict(x_train)
        similarities = cosine_similarity(prd_value.reshape(-1,1), preY)
        print(similarities)
        # 找到大于0.9的行
        indices = np.where(similarities > 0.9)[0]
        print(indices)
        filtered_value = y_train[indices]
        # inner_dot=np.dot(preY-prd_value,preY-prd_value).reshape(-1,1)
        aa.append(prd_value)
        bb.append(filtered_value)
print(bb[0])
print("\n")
print(aa[0])
a, b = solve_model(np.array(bb[0]), np.ones(len(bb[0])) / len(bb[0]))
tree_value += a
tree_solution.append(b)
tree_point.append(tree_pointvalue / length)
tree_values.append(tree_value / length)

#
#
#
# print("*************************")
# print("*************************")
# print(len(times), len(tree_values), len(true_values), len(tree_point))
# mymatric = np.array([times, tree_values, true_values * len(times), tree_point])
# tree_solution = np.array(tree_solution)
# solution = np.array(true_solution)
# np.savetxt("tree_solution.csv", tree_solution, delimiter=',')
# np.savetxt("tree_truesolution.csv", solution, delimiter=',')
# np.savetxt("tree.csv", mymatric, delimiter=',')




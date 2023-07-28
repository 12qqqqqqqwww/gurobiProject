import numpy as np
import pandas as pd
from gurobipy import *
from data.get_data import Data
from sklearn.metrics import classification_report
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils.multiclass import type_of_target
from sklearn.tree import DecisionTreeClassifier
import multiprocessing

def solve_model(demand, weight):
    supplyNum = 4  # 仓库数量
    locationNum = 12  # 需求地数量
    sampleNum=len(weight)
    C = np.array([
        [0.15   , 1.3124 , 1.85   ,    1.3124],
        [0.50026, 0.93408, 1.7874 ,    1.6039],
        [0.93408, 0.50026, 1.6039 ,   1.7874],
        [1.3124 , 0.15   , 1.3124 ,   1.85],
        [1.6039 , 0.50026, 0.93408,   1.7874],
        [1.7874 , 0.93408, 0.50026,   1.6039],
        [1.85   , 1.3124 , 0.15   ,   1.3124],
        [1.7874 , 1.6039 , 0.50026,   0.93408],
        [1.6039 , 1.7874 , 0.93408,   0.50026],
        [1.3124 , 1.85   , 1.3124 ,   0.15],
        [0.93408, 1.7874 , 1.6039 ,   0.50026],
        [0.50026, 1.6039 , 1.7874 ,   0.93408]
    ])  # 运输成本
    p1=5#第一次生产成本
    p2=100#第二次生产成本
    supply = [i for i in range(supplyNum)]  # 仓库点集合
    warehouse = [i for i in range(locationNum)]  # 需求地点集合
    sample=[i for i in range(sampleNum)]
    # 构建模型
    model = Model('two stage transportation')
    x = {}  # 创造一个决策变量的储存列表
    z={}
    t={}
    v={}
    # 添加决策变量
    #运输
    for i in supply:
        for j in warehouse:
            for k in sample:
                name = 'x' + str(i) + '_' + str(j)+'_' + str(k)
                x[i,j,k] = model.addVar(0, GRB.INFINITY, vtype=GRB.CONTINUOUS, name=name)
    #生产
    for i in supply:
        name = 'z' + str(i)
        z[i] = model.addVar(0,GRB.INFINITY,vtype=GRB.CONTINUOUS, name=name)
    #第二次生产
    for i in supply:
        for k in sample:
            name="t"+ str(i)+'_'+str(k)
            t[i,k] = model.addVar(0, GRB.INFINITY, vtype=GRB.CONTINUOUS, name=name)
    for i in sample:
        name="v"+str(i)
        v[i]=model.addVar(0, GRB.INFINITY, vtype=GRB.CONTINUOUS, name=name)
    model.update()
    # 添加目标函数
    model.setObjective(quicksum(z[i] for i in supply)*p1
                       +quicksum(v[k]*weight[k] for k in sample),
                       GRB.MINIMIZE)

    # 添加产量约束
    for i in supply:
        for k in sample:
            model.addConstr(quicksum(x[i,j,k] for j in warehouse) <= (z[i] + t[i,k]))

    # 添加销量约束
    for j in warehouse:
        for k in sample:
            model.addConstr(quicksum(x[i,j,k] for i in supply) >= demand[k,j])

    for k in sample:
        model.addConstr((quicksum(x[i,j,k] * C[j,i] for i in supply for j in warehouse )*10
                       +quicksum(t[i,k] for i in supply )*p2)<=v[k])

    # 模型求解
    model.optimize()
    Z=[]
    V=[]
    T=[]
    # 结果输出
    for key in z.keys():
        Z.append(z[key].X)
    for key in v.keys():
        V.append(v[key].X)

    for key in t.keys():
        T.append(t[key].X)
    return model.ObjVal,Z
# solve_model(np.ones((3,12)),np.array([1,1,1]))
data=Data()
#KNN

# knn.fit(data.train_X,data.train_y[:,0])
# print(knn.score(data.test_X,data.test_y[:,0]))


def tree_driven(Samplerange):
    times = []
    tree_values = []
    tree_point = []
    true_values = []
    tree_solution = []
    true_solution = []
    clf=DecisionTreeClassifier()
    step = 0
    for i in Samplerange:
        step += 1
        times.append(i)
        x_train=data.train_X[:i + 20]
        y_train =data.train_y[:i + 20]
        clf.fit(x_train,y_train )
        tree_value = 0
        tree_pointvalue = 0
        true_value = 0
        # shuffle= np.random.randint(1, 5000, size=100)
        length = i // 10 if i>=10 else 1
        for j in range(2, length+2):
            prd_value=clf.predict(np.array(data.test_X[j-1:j].reshape(1,-1)))
            tree_pointvalue += solve_model(prd_value, np.array([1]))[0]
            preY = clf.predict(x_train)
            # 找到大于0.9的行
            mask = np.all(preY ==prd_value, axis=1)
            filtered_value = y_train[mask]
            a, b = solve_model(np.array(filtered_value), np.ones(len(filtered_value))/len(filtered_value))
            tree_value += a
            tree_solution.append(b)
            c, d = solve_model(data.test_y[j - 1:j], np.array([1]))
            true_value += c
            true_solution.append(d)
        length = 1 if length == 0 else length
        tree_point.append(tree_pointvalue / length )
        tree_values.append(tree_value / length )
        true_values.append(true_value/length)
    print(step,"  *************************")
    print(step,"  *************************")
    mymatric = np.array([times, tree_values, true_values, tree_point])
    tree_solution = np.array(tree_solution)
    solution = np.array(true_solution)
    np.savetxt("tree_solution.csv", tree_solution, delimiter=',')
    np.savetxt("tree_truesolution.csv", solution, delimiter=',')
    np.savetxt("tree.csv", mymatric, delimiter=',')

def knn_driven(Samplerange):
    times = []
    K_error = []
    K_point = []
    true_values = []
    K_solution = []
    true_solution = []
    knn = KNeighborsClassifier(n_neighbors=8)
    step=0
    for i in Samplerange:
        step+=1
        times.append(i)
        knn.fit(data.train_X[:i+20],data.train_y[:i+20])
        knn_value=0
        true_value = 0
        k_pointvalue=0
        # shuffle= np.random.randint(1, 5000, size=100)
        length = i // 10 if i>=10 else 1
        for j in range(2,length+2):
            row_iloc,distance=knn.kneighbors(data.test_X[j-1:j])
            k_pointvalue+=solve_model(knn.predict(data.test_X[j-1:j]),np.array([1]))[0]
            sample=data.data.iloc[row_iloc[0],3:]
            distance=sum(distance[0]+0.000001)/(distance[0]+0.000001)
            Knn_weight = np.array(distance/sum(distance))
            print(distance[0])
            print(Knn_weight)
            a,b=solve_model(np.array(sample),Knn_weight)
            knn_value+=a
            c,d=solve_model(data.test_y[j-1:j], np.array([1]))
            print(c)
            true_value+=c
            K_solution.append(b)
            true_solution.append(d)
        length=1 if length==0 else length
        K_point.append(k_pointvalue/(length))
        K_error.append(knn_value/(length))
        true_values.append(true_value/(length))
        print(step,"   *************************")
        print(step,"   *************************")
        mymatric=np.array([times,K_error,true_values,K_point])
        knn_solution=np.array(K_solution)
        solution = np.array(true_solution)
        np.savetxt("knn_solution.csv", knn_solution, delimiter=',')
        np.savetxt("solution.csv", solution, delimiter=',')
        np.savetxt("knn.csv",mymatric,delimiter=',')
# SAA

def SAA(size):
    SAA_time = []
    SAA_error = []
    for i in range(10,size,500):
        SAA_time.append(i)
        SAA_value=0
        # shuffle= np.random.randint(1, 5000, size=100)
        sample = data.data.iloc[0:i, 3:]
        SAA_value=solve_model(np.array(sample),np.array(np.ones(len(sample))/len(sample)))[0]
        SAA_error.append(SAA_value)
        print("第  ",i,"  *************************"+"次")
        print("第  ",i,"  *************************"+"次")
    mymatric=np.array([SAA_time,SAA_error])
    np.savetxt("saa.csv",mymatric,delimiter=',')
# tree_driven(10000,100)
# knn_driven(150000)
# tree_driven(30000)
SampleRange=[2,5,10,50,100,300,600,1000,3000,6000,8000,10000]
# Sample1Range=[10,50,100,300,600,1000,3000,6000,8000,10000,50000,100000]
knn_driven(SampleRange)
tree_driven(SampleRange)


# knn.fit(data.train_X[:50000],data.train_y[:50000])
# row_iloc=knn.kneighbors(data.test_X[10:11],return_distance=False)[0]
# sample=data.data.iloc[row_iloc,3:]
# knn_value=solve_model(np.array(sample),np.array(np.ones(len(sample))/len(sample)))
# true_values=solve_model(data.test_y[10:11],np.array([1]))

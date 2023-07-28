import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def generateData(size):
    mean=np.array([0,0,0])
    cov=np.array([[1, 0.5, 0],[0.5, 1.2, 0.5],[0 ,0.5 ,0.8]])
    u=np.random.multivariate_normal(mean,cov,size)
    A=np.array([[0.8 ,0.1, 0.1],
    [0.1 ,0.8, 0.1],
    [0.1, 0.1 ,0.8],
    [0.8 ,0.1, 0.1],
    [0.1 ,0.8, 0.1],
    [0.1, 0.1, 0.8],
    [0.8, 0.1, 0.1],
    [0.1, 0.8, 0.1],
    [0.1, 0.1, 0.8],
    [0.8, 0.1, 0.1],
    [0.1 ,0.8, 0.1],
    [0.1, 0.1, 0.8]])*2.5
    B=np.array([[0,-1,-1 ],
    [-1,0,-1 ],
    [-1,-1 ,0],
    [0,-1,-1 ],
    [-1,0,1 ],
    [-1,1,0 ],
    [0,1,-1 ],
    [1,0,-1 ],
    [1,-1,0 ],
    [0,1,1 ],
    [1,0,1 ],
    [1,1,0 ]])*7.5
    phi1=np.array([[0.5,-0.9,0],[1.1,-0.7,0],[0,0,0.5]])
    phi2=np.array([[0,-0.5,0],[-0.5,0,0],[0,0,0]])
    theta1=np.array([[0.4,0.8,0],[-1.1,-0.3,0],[0,0,0]])
    theta2=np.array([[0,-0.8,0],[-1.1,0,0],[0,0,0]])
    X=np.zeros((size,3))
    for i in range(len(X)):
        X[i]=np.random.normal(10,4,3)
    Y=np.zeros((size,12))
    for i in range(len(Y)):
        delta=np.random.normal(0,1,3)
        epsilon=np.random.normal(0,0.1,12)
        # +np.dot(B, X[i]) * epsilon
        # +delta / 4
        Y[i]=np.round(np.dot(A,X[i])+np.dot(B, X[i]) * epsilon).reshape(1,12)
    Y=np.where(Y>0, Y, 0)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    data=np.concatenate((X,Y),axis=1)

    np.savetxt("data.csv",data,delimiter=',')
generateData(150000)


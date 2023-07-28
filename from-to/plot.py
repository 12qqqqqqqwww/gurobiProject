from itertools import cycle

import numpy as np
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
import math

knn_matrix = np.loadtxt(open("knn.csv", "rb"), delimiter=",", skiprows=0)
saa_matrix = np.loadtxt(open("saa.csv", "rb"), delimiter=",", skiprows=0)
Knn_solution=np.loadtxt(open("knn_solution.csv", "rb"), delimiter=",", skiprows=0)
solution=np.loadtxt(open("solution.csv", "rb"), delimiter=",", skiprows=0)
tree_matrix= np.loadtxt(open("tree.csv", "rb"), delimiter=",", skiprows=0)[:3, :]
marker_styles = cycle(plt.Line2D.filled_markers)



def plot(matrix,string):
    for matrix,string in zip(matrix,string):
        times=matrix[0]
        error=np.abs((matrix[1]-matrix[2]))/matrix[2]+0.05

        #
        plt.plot(times,error,label=string, marker=next(marker_styles))
        if(len(matrix)>3):
            point_error = np.abs((matrix[3] - matrix[2])) / matrix[2] + 0.05
            plt.plot(times,point_error,label=string+'_point', marker=next(marker_styles))
        # plt.plot(times,knn_matrix[1],label="knn")
        # plt.plot(times,knn_matrix[2],label='true')
        # plt.plot(times,knn_matrix[3],label='knn_point')
        # plt.plot(times,saa_matrix[1],label='saa')
    times=matrix[0]
    plt.plot(times, np.zeros(len(times)) + 0.05, label='optimal',linestyle="--")
    plt.xscale("log")
    plt.ylim(0, 0.6)
    plt.xlabel("Testing Sample Size")
    plt.legend()
    plt.show()
plot([knn_matrix,tree_matrix],["knn","tree"])
# plot(knn_matrix,"knn")


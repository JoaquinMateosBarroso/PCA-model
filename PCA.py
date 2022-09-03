import pandas as pd
import numpy as np


def covariance(x : pd.DataFrame, y : pd.DataFrame):
    covariance = 0
    for i in range(x.size):
        covariance += (x[i]-x.mean())*(y[i]-y.mean())
    return covariance/(x.size-1)    

def make_covariance_matrix(data : pd.DataFrame):
        matrix = np.array([[]])
        for i in range(data.columns.size):
            new_column = []
            for j in range(data.columns.size):
                new_column.append(covariance(data.iloc[:, i], data.iloc[:, j]))
            if i == 0:
                matrix = np.array([new_column])
            else:
                matrix = np.append(matrix, [new_column], axis = 0)
        return matrix

class PCA():
    def __init__(self):
        self.matrix = np.array([[]])
        self.components = np.array([[]])
        self.values = np.array([])
    
    #The data must be normalized in order to train the object  
    def train(self, data : pd.DataFrame):
        self.matrix = make_covariance_matrix(data)
        self.values, self.components = np.linalg.eig(self.matrix)
    
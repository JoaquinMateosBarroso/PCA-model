import pandas as pd
import numpy as np
class PCA():
    matrix = np.array([[]])
    
    #The data must be normalized in order to train the object
    def covariance(self, x : pd.DataFrame, y : pd.DataFrame):
        covariance = 0
        for i in range(x.size):
            covariance += (x[i]-x.mean())*(y[i]-y.mean())
        return covariance/(x.size-1)
    def make_covariance_matrix(self, data : pd.DataFrame):        
        for i in range(data.columns.size):
            new_column = []
            for j in range(data.columns.size):
                new_column.append(PCA.covariance(self, data.iloc[:, i], data.iloc[:, j]))
            if i == 0:
                self.matrix = np.array([new_column])
            else:
                self.matrix = np.append(self.matrix, [new_column], axis = 0)
        return
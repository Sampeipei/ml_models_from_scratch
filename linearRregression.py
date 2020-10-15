import numpy as np
import matplotlib.pyplot as plt


class linearRegression:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.dNum = X.shape[0]
        self.nDim = X.shape[1]


    def train(self):
        # Concat vector with one for bias term
        Z = np.concatenate([self.X, np.ones([self.dNum, 1])], axis=1)
        # Calc parts for optimization
        ZZ = np.matmul(Z.T, Z) / self.dNum
        ZY = np.matmul(Z.T, self.Y) / self.dNum
        # Optimize params
        v_opt = np.matmul(np.linalg.inv(ZZ), ZY)
        # Determine parameters w, b
        self.w = v_opt[-1:]
        self.b = v_opt[-1]

    def trainRegularized(self, lamb=0.1):
        # Concatenate one vectors for bias
        Z = np.concatenate([self.X, np.ones([self.dNum, 1])], axis=1)
        # Calc parts for optimization
        ZZ = 1 / self.dNum * np.matmul(Z.T, Z) + lamb * np.eye(self.nDim)
        ZY = 1 / self.dNum * np.matmul(Z.T, self.Y)
        # Optimize param
        v_opt = np.matmul(np.linalg.inv(ZZ), ZY)

        # Save params
        self.w = v_opt[:-1]
        self.b = v_opt[-1]

    def predict(self, X):
        y_pred = np.matmul(X, self.w) + self.b
        return y_pred

    def RMSE(self, X, Y):
        return np.sqrt(np.mean(Y - self.predict(X)))

    def R2(self, X, Y):
        return 1 - np.sum(np.square(self.predict(X) - Y)) / np.sum(np.square(Y - np.meam(Y,axis=0)))

    def plotResult(self, X=[], Y=[], xLabel="", yLabel="", fName=""):
        # Result plot function
        # New Comments
        pass
import pandas as pd
import numpy as np


# construct design matrix from linearized polynomial and measurements
def design_matrix(xn, yn):
    dm = np.empty(shape=(len(xn), 6))

    for i in range(len(xn)):
        dm[i] = [1, xn[i], yn[i], xn[i] * yn[i], xn[i] ** 2, yn[i] ** 2]

    return dm


# least squares adjustment using y = (At*A)-1 * At*x
def least_squares_adjustment(x, y, z):
    A = design_matrix(x, y)
    A_T = np.transpose(A)
    ATA_inv = np.linalg.inv(np.matmul(A_T, A))
    AT_X = np.dot(A_T, np.asarray(z))

    return np.dot(ATA_inv, AT_X)


df = pd.read_excel('./DEM values LSA and RMSE Assignment 1920.xlsx')
x, y, z = df["x [m]"].tolist(), df["y [m]"].tolist(), df["h(x,y) [m]"].tolist()
parameters = least_squares_adjustment(x, y, z)




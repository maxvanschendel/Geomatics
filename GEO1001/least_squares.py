import pandas as pd
import numpy as np


def second_order_poly_surf(x, y, a0, a1, a2, a3, a4, a5):
    return a0 + a1*x + a2*y + a3*x*y + a4*x**2 + a5*y**2


# construct design matrix from linearized polynomial and measurements
def design_matrix(x, y):
    dm = np.empty(shape=(len(x), 6))

    for i in range(len(x)):
        dm[i] = [1, x[i], y[i], x[i] * y[i], x[i] ** 2, y[i] ** 2]

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
print(parameters)
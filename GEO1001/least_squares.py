import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def second_order_poly_surf(x, y, params):
    print(params)
    return params[0] + params[1] * x + params[2] * y + params[3] * x * y + params[4] * x ** 2 + params[5] * y ** 2


# construct design matrix from linearized polynomial and measurements
def design_matrix(x, y):
    dm = np.empty(shape=(len(x), 6))

    for i in range(len(x)):
        dm[i] = [1, x[i], y[i], x[i] * y[i], x[i] ** 2, y[i] ** 2]

    return dm


# least squares adjustment using y = (At*A)-1 * At*x
def least_squares_adjustment(x, y, z):
    A = design_matrix(x, y)
    A_transp = np.transpose(A)
    ATA_inv = np.linalg.inv(np.matmul(A_transp, A))
    AT_x = np.dot(A_transp, np.asarray(z))

    return np.dot(ATA_inv, AT_x)


df = pd.read_excel('./DEM values LSA and RMSE Assignment 1920.xlsx')
x, y, z = df["x [m]"].tolist(), df["y [m]"].tolist(), df["h(x,y) [m]"].tolist()
parameters = least_squares_adjustment(x, y, z)


fig = plt.figure()
ax = Axes3D(fig)

s = 100
X = np.arange(-s, s, 1)
Y = np.arange(-s, s, 1)
X, Y = np.meshgrid(X, Y)
vfunc = np.vectorize(second_order_poly_surf, excluded=['params'])

Z = vfunc(X, Y, params = parameters)


ax.plot_surface(X, Y, Z)
R = np.sqrt(X**2 + Y**2)
plt.show()
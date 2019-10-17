from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# computes root mean square error between two numpy arrays
def compute_rmse(predicted, observed):
    return np.mean(np.square(predicted - observed)) ** 0.5


# plots vertical error vectors in 3d
def plot_error_vector(x, y, z, errors):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.quiver(x, y, z, 0, 0, errors, arrow_length_ratio=0.5)

    ax.scatter(x, y, z, depthshade=True, marker='x')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('h(x, y) - H(x, y) [m]')
    plt.show()


# compute rmse of height measurements and plot error vector in 3d
if __name__ == '__main__':
    df = pd.read_excel('./DEM values LSA and RMSE Assignment 1920.xlsx')
    x, y, h_xy, H_xy = df["x [m]"].to_numpy(), \
                       df["y [m]"].to_numpy(), \
                       df["h(x,y) [m]"].to_numpy(), \
                       df["H(x,y) [m]"].to_numpy()

    print(compute_rmse(H_xy, h_xy))
    plot_error_vector(x, y, H_xy, h_xy - H_xy)

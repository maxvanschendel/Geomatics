from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# computes root mean square error between two numpy arrays
def compute_rmse(predicted, observed):
    return np.mean(np.square(predicted - observed)) ** 0.5


# plots vertical error vectors in 3d
def plot_error_vector(x, y, z, errors):
    plt.rcParams["figure.figsize"] = (10,7)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.quiver(x, y, z, 0, 0, errors, arrow_length_ratio=0.5)

    ax.scatter(x, y, z, depthshade=True, marker='x')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('h(x, y) [m] - H(x, y) [m]')
    plt.show()


# compute rmse of height measurements and plot error vector in 3d
if __name__ == '__main__':
    filter_outliers = False

    if filter_outliers:
        pass
    else:
        data = pd.read_csv("A4_point.tsv", sep='\t')

        x = data['x'].to_numpy()
        y = data['y'].to_numpy()
        h = data['h'].to_numpy()
        H = data['H'].to_numpy()

print(compute_rmse(h, H))
plot_error_vector(x, y, H, h - H)
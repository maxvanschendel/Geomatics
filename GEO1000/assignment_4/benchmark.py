# GEO1000 - Assignment 4
# Authors: Max van Schendel
# Studentnumbers: 4384644

import time
import delaunay
import matplotlib.pyplot as plt
import numpy as np

def benchmark():
    # Your implementation here
    marks = [5, 10, 50, 100, 150, 200, 250]
    results = []

    for i in marks:
        t_start = time.clock()
        pts = delaunay.make_random_points(i)
        dt = delaunay.DelaunayTriangulation(pts)
        dt.triangulate()
        results.append(time.clock() - t_start)

    # fit 2nd order polynomial through data using least squares
    poly = np.poly1d(np.polyfit(marks, results, 2))
    x = np.arange(5, 5000)
    y = poly(x)
    plt.plot(x, y, c='grey', zorder=1)

    extra_values = [500, 1000, 5000]

    for i in extra_values:
        plt.scatter(i, poly(i), c='red', zorder=2)

    # plot scatterplot of benchmark results
    plt.scatter(marks, results, c='black', zorder=2)
    plt.show()






if __name__ == "__main__":
    benchmark()

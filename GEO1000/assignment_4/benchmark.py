# GEO1000 - Assignment 4
# Authors:
# Studentnumbers:

import time
import delaunay
import matplotlib.pyplot as plt

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

    plt.scatter(marks, results)
    plt.show()






if __name__ == "__main__":
    benchmark()

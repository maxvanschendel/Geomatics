# -- mycode_hw03.py
# -- GEO1015.2019--hw03
# -- [YOUR NAME]
# -- [YOUR STUDENT NUMBER]

import math
from laspy.file import File
import numpy as np
import startin
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt


def point_cloud_to_grid(point_cloud, cell_size, thinning_factor):

    # create empty sets for ground points (gp) and discarded points (dp)
    gp, dp = set(), set()

    X, Y, Z = point_cloud.X[::thinning_factor], point_cloud.Y[::thinning_factor], point_cloud.Z[::thinning_factor]
    X_min, X_max, Y_min, Y_max = X.min(), X.max(), Y.min(), Y.max()

    # move (X_min, Y_min) to (0, 0)
    X = X - X_min
    Y = Y - Y_min

    stacked_array = np.vstack((X, Y, Z)).transpose()
    point_cloud_ar = np.zeros((X_max - X_min + 1, Y_max - Y_min + 1))

    # converts 1D array to a 2D X, Y array with Z elements
    for x, y, z in stacked_array:
        if point_cloud_ar[x][y] == 0 or z < point_cloud_ar[x][y]:
            point_cloud_ar[x][y] = z

    # construct initial grid by slicing numpy array in chunks
    for x in range(int((X_max-X_min)/cell_size)):
        for y in range(int((Y_max-Y_min)/cell_size)):
            x_range, y_range = (x*cell_size, (x+1)*cell_size), (y*cell_size, (y+1)*cell_size)
            cell_points = point_cloud_ar[x_range[0]: x_range[1], y_range[0]: y_range[1]]

            non0_i = np.nonzero(cell_points)
            non0_it = cell_points[non0_i]

            if non0_it.size:
                # add local minimum to ground points
                min_i = np.argmin(non0_it)
                gp.add((non0_i[0][min_i]+x_range[0]+X_min, non0_i[1][min_i]+y_range[0]+X_min, non0_it[min_i]))

                # get points that are not the local minimum
                non_min_x = np.delete(non0_i[0], min_i)
                non_min_y = np.delete(non0_i[1], min_i)
                non_min_elements = np.delete(non0_it, min_i)

                # add points that are not the local minimum to non-ground set
                for index, x_cur in enumerate(non_min_x):
                    y_cur = non_min_y[index]
                    dp.add((x_cur+x_range[0]+X_min, y_cur+y_range[0]+X_min, non_min_elements[index]))

    return gp, dp


def perpendicular_distance(p, tri):
    # finds perpendicular distance of point to a triangle

    tri_normal = np.cross(tri[0] - tri[1], tri[2] - tri[1])
    tri_normal_unit = tri_normal / np.linalg.norm(tri_normal)

    vec2p_dot = np.dot(p - tri[0], tri_normal_unit)

    intersection = p - (tri_normal_unit * vec2p_dot)
    if point_in_tri(intersection, tri):
        return np.linalg.norm(intersection - p)

    return None


def point_in_tri(p, tri):
    # checks if point is in a triangle using barycentric coordinates

    u, v, w = tri[1] - tri[0], tri[2] - tri[0], p - tri[0]
    n = np.cross(u, v)

    gamma = np.dot(np.cross(u, w), n) / np.dot(n, n)
    beta = np.dot(np.cross(w, v), n) / np.dot(n, n)
    alpha = 1 - gamma - beta

    if 0 <= alpha <= 1 and 0 <= beta <= 1 and 0 <= gamma <= 1:
        return True

    return False


def dt_to_grid(dt, cell_size):
    # converts delaunay mesh to grid by interpolating

    convex_hull = [dt.get_point(p) for p in dt.convex_hull()]
    X, Y = [p[0] for p in convex_hull], [p[1] for p in convex_hull]
    X_min, X_max, Y_min, Y_max = int(min(X)), int(max(X)), int(min(Y)), int(max(Y))

    grid = np.empty(((X_max - X_min)//cell_size, (Y_max - Y_min)//cell_size))

    for x in range((X_max - X_min)//cell_size):
        for y in range((Y_max - Y_min)//cell_size):
            p = (x * cell_size + X_min, y * cell_size + Y_min)

            try:
                grid[x][y] = dt.interpolate_tin_linear(p[0], p[1])

            except OSError:
                # this means the point is outside the dt's convex hull
                # solution: find closest point on convex hull and set value to that

                # ch_cp = np.argmin(np.asarray([np.linalg.norm(np.asarray(i[0:2]) - np.asarray(p)) for i in convex_hull]))
                # grid[x][y] = convex_hull[ch_cp][2]

                pass

    return grid


def idw_to_grid(dt, cell_size):
    pass


def filter_ground(jparams):
    """
  !!! TO BE COMPLETED !!!
    
  Function that reads a LAS file, performs thinning, then performs ground filtering, and creates a two rasters of the ground points. One with IDW interpolation and one with TIN interpolation.

  !!! You are free to subdivide the functionality of this function into several functions !!!
    
  Input:
    a dictionary jparams with all the parameters that are to be used in this function:
      - input-las:        path to input .las file,
      - thinning-factor:  thinning factor, ie. the `n` in nth point thinning method,
      - gf-cellsize:      cellsize for the initial grid that is computed as part of the ground filtering algorithm,
      - gf-distance:      distance threshold used in the ground filtering algorithm,
      - gf-angle:         angle threshold used in the ground filtering algorithm,
      - idw-radius:       radius to use in the IDW interpolation,
      - idw-power:        power to use in the IDW interpolation,
      - output-las:       path to output .las file that contains your ground classification,
      - grid-cellsize:    cellsize of the output grids,
      - output-grid-tin:  filepath to the output grid with TIN interpolation,
      - output-grid-idw:  filepath to the output grid with IDW interpolation
  """
    # load las file and relevant parameters
    point_cloud = File(jparams['input-las'], mode='r')
    max_distance = jparams['gf-distance']
    max_angle = jparams['gf-angle']

    print('- Flattening point cloud')
    gridded_pc = point_cloud_to_grid(point_cloud, jparams['gf-cellsize'], jparams['thinning-factor'])
    ground_points, unprocessed_points = gridded_pc[0], gridded_pc[1]

    print('- Creating initial mesh')
    dt = startin.DT()
    dt.insert(list(ground_points))

    print('- Writing initial mesh')
    dt.write_obj('./start.obj')

    print('- Growing terrain')
    keep_running = True
    while keep_running:
        keep_running = False

        for x, y, z in unprocessed_points:
            if (x, y, z) not in ground_points:
                # get vertices of triangle intersecting with vertical projection of point
                triangle_vertices = [dt.get_point(p) for p in dt.locate(x, y)]

                if triangle_vertices:
                    # check if perpendicular distance to triangle is below threshold
                    pdist = perpendicular_distance(np.asarray([x, y, z]), np.asarray(triangle_vertices))

                    if pdist is not None and pdist < max_distance:
                        # check if max angle between point and triangle vertices is below threshold
                        dvx = [np.linalg.norm(np.asarray([x, y, z]) - np.asarray(i)) for i in triangle_vertices]
                        v_angles = np.asarray([math.degrees(math.asin(pdist/i)) for i in dvx])

                        if v_angles.max() < max_angle:
                            # add point to dt and mark it as ground point, set flag to keep running
                            dt.insert([(x, y, z)])
                            ground_points.add((x, y, z))
                            keep_running = True

    print('- Writing final mesh')
    dt.write_obj('./end.obj')

    print('- Interpolating grid (Delaunay)')
    delaunay_grid = dt_to_grid(dt, jparams['grid-cellsize'])

    return

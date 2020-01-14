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
    flat_pc = np.zeros((X_max - X_min + 1, Y_max - Y_min + 1), dtype=np.float32)

    # converts 1D array to a 2D X, Y array with Z elements
    for x, y, z in stacked_array:
        if flat_pc[x][y] == 0 or z < flat_pc[x][y]:
            flat_pc[x][y] = z

    # construct initial grid by slicing numpy array in chunks
    for y in range(int((Y_max-Y_min)/cell_size)):
        for x in range(int((X_max-X_min)/cell_size)):
            x_range, y_range = (x*cell_size, (x+1)*cell_size), (y*cell_size, (y+1)*cell_size)
            cell_points = flat_pc[x_range[0]: x_range[1], y_range[0]: y_range[1]]

            non0_i = np.nonzero(cell_points)
            non0_it = cell_points[non0_i]

            if non0_it.size:
                # add local minimum to ground points
                min_i = np.argmin(non0_it)
                gp.add((non0_i[0][min_i]+x_range[0]+X_min, non0_i[1][min_i]+y_range[0]+X_min, non0_it[min_i]))

                # delete local minimum
                non_min_x = np.delete(non0_i[0], min_i)
                non_min_y = np.delete(non0_i[1], min_i)
                non_min_z = np.delete(non0_it, min_i)

                # add points that are not the local minimum to non-ground set
                for index, x_cur in enumerate(non_min_x):
                    y_cur = non_min_y[index]
                    dp.add((x_cur+x_range[0]+X_min, y_cur+y_range[0]+X_min, non_min_z[index]))

    return gp, dp


def grow_terrain(tin, p, gp, max_distance, max_angle):
    keep_running = True
    while keep_running:
        keep_running = False

        for x, y, z in p:
            if (x, y, z) not in gp:
                tri_v = [tin.get_point(p) for p in tin.locate(x, y)]

                if tri_v:
                    # check if perpendicular distance to triangle is below threshold
                    p_cur = np.asarray([x, y, z])
                    pdist = perpendicular_distance(p_cur, np.asarray(tri_v))

                    # check if max angle between point and triangle vertices is below threshold
                    if pdist is not None and pdist < max_distance:
                        dvx = [euclidean_distance(p_cur, np.asarray(i)) for i in tri_v]
                        v_angles = np.asarray([math.degrees(math.asin(pdist / i)) for i in dvx])

                        # add point to dt and mark it as ground point, set flag to keep running
                        if v_angles.max() < max_angle:
                            tin.insert([(x, y, z)])
                            gp.add((x, y, z))
                            keep_running = True

    return tin


def euclidean_distance(a, b):
    return np.linalg.norm(a - b)


def perpendicular_distance(p, tri):
    # finds perpendicular distance of point to a triangle
    # returns none if projection falls outside triangle

    normal = np.cross(tri[0] - tri[1], tri[2] - tri[1])
    normal_hat = normal / np.linalg.norm(normal)
    vec2p_dot = np.dot(p - tri[0], normal_hat)
    plane_int = p - (normal_hat * vec2p_dot)

    if point_in_tri(plane_int, tri):
        return euclidean_distance(plane_int, p)

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


def write_asc(grid, cell_size, fn, origin):
    header = "NCOLS {}\nNROWS {}\nXLLCORNER {}\nYLLCORNER {}\nCELLSIZE {}\nNODATA_VALUE 0.0".format(
                grid.shape[0], grid.shape[1], origin[0], origin[1], cell_size)

    np.savetxt(fn, grid, delimiter=' ', header=header, comments='', fmt='%1.1f')


def dt_to_grid(dt, cell_size):
    # converts Delaunay mesh to grid using TIN interpolation

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
                pass

    return grid, (X_min, Y_max)


def idw_to_grid(dt, cell_size, radius, power):
    # converts Delaunay mesh to grid using IDW interpolation

    convex_hull = [dt.get_point(p) for p in dt.convex_hull()]
    X, Y = [p[0] for p in convex_hull], [p[1] for p in convex_hull]
    X_min, X_max, Y_min, Y_max = int(min(X)), int(max(X)), int(min(Y)), int(max(Y))

    grid = np.empty((int((X_max - X_min) // cell_size), int((Y_max - Y_min) // cell_size)))

    vertices = dt.all_vertices()
    tree = cKDTree([v[:2] for v in vertices])

    # iterate over grid, interpolating values based on corresponding point in TIN (p)
    for x in range(int((X_max - X_min) // cell_size)):
        for y in range(int((Y_max - Y_min) // cell_size)):
            p = (x * cell_size + X_min, y * cell_size + Y_min)
            nbs = [vertices[i] for i in tree.query_ball_point(p, radius)]

            if nbs:
                grid[x][y] = sum([i[2]/euclidean_distance(np.asarray(i[:2]), np.asarray(p))**power for i in nbs]) / \
                             sum([1/euclidean_distance(np.asarray(i[:2]), np.asarray(p))**power for i in nbs])

    return grid, (X_min, Y_max)


def filter_ground(jparams):
    """

  Function that reads a LAS file, performs thinning, then performs ground filtering,
  and creates a two rasters of the ground points. One with IDW interpolation and one with TIN interpolation.

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

    print('- Flattening point cloud')
    gridded_pc = point_cloud_to_grid(point_cloud, jparams['gf-cellsize'], jparams['thinning-factor'])
    ground_points, unprocessed_points = gridded_pc[0], gridded_pc[1]

    print('- Creating initial mesh')
    dt = startin.DT()
    dt.insert(list(ground_points))

    print('- Writing initial mesh')
    dt.write_obj('./start.obj')

    print('- Growing terrain')

    dt = grow_terrain(dt, unprocessed_points, ground_points, jparams['gf-distance'], jparams['gf-angle'])

    print('- Writing final mesh')
    dt.write_obj('./end.obj')

    print('- Writing labeled point cloud')
    out_file = File(jparams['output-las'], mode='w', header=point_cloud.header)
    gp = dt.all_vertices()
    out_file.X, out_file.Y, out_file.Z = [p[0] for p in gp], [p[1] for p in gp], [p[2] for p in gp]

    out_file.close()

    print('- Creating raster (TIN)')
    dg = dt_to_grid(dt, jparams['grid-cellsize'])

    write_asc(dg[0], jparams['grid-cellsize'], jparams['output-grid-tin'], dg[1])

    print('- Creating raster (IDW)')
    ig = idw_to_grid(dt, jparams['grid-cellsize'], jparams['idw-radius'], jparams['idw-power'])
    write_asc(ig[0], jparams['grid-cellsize'], jparams['output-grid-idw'], ig[1])

    return

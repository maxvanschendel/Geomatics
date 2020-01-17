# -- mycode_hw03.py
# -- GEO1015.2019--hw03
# -- Max van Schendel
# -- 4384644

import math
from laspy.file import File
import numpy as np
import startin
from scipy.spatial import cKDTree


def point_cloud_to_grid(point_cloud, cell_size, tf):

    # create empty sets for ground points (gp) and discarded points (dp)
    gp, dp = set(), set()

    X, Y, Z = point_cloud.X[::tf], point_cloud.Y[::tf], point_cloud.Z[::tf]

    X_min, X_max, Y_min, Y_max = X.min(), X.max(), Y.min(), Y.max()

    # move (X_min, Y_min) to (0, 0)
    X = X - X_min
    Y = Y - Y_min
    Z = Z - Z.min()

    stacked_array = np.vstack((X, Y, Z)).transpose()
    flat_pc = np.zeros((X_max - X_min + 1, Y_max - Y_min + 1), dtype=np.float32)

    # converts 1D array to a 2D X, Y array with Z elements
    for x, y, z in stacked_array:
        if flat_pc[x][y] == 0 or z < flat_pc[x][y]:
            flat_pc[x][y] = z

    # construct initial grid by slicing numpy array in chunks
    for y in range(int((Y_max-Y_min) / cell_size)):
        for x in range(int((X_max-X_min) / cell_size)):
            x_range, y_range = (x*cell_size, (x+1)*cell_size), (y*cell_size, (y+1)*cell_size)
            cell_points = flat_pc[x_range[0]: x_range[1], y_range[0]: y_range[1]]

            non0_i = np.nonzero(cell_points)
            non0_it = cell_points[non0_i]

            if non0_it.size:
                # add local minimum to ground points
                min_i = np.argmin(non0_it)
                gp.add((non0_i[0][min_i]+x_range[0], non0_i[1][min_i]+y_range[0], non0_it[min_i]))

                # delete local minimum
                non_min_x = np.delete(non0_i[0], min_i)
                non_min_y = np.delete(non0_i[1], min_i)
                non_min_z = np.delete(non0_it, min_i)

                # add points that are not the local minimum to non-ground set
                for index, x_cur in enumerate(non_min_x):
                    y_cur = non_min_y[index]
                    dp.add((x_cur+x_range[0], y_cur+y_range[0], non_min_z[index]))

    return gp, dp, (X_min, Y_max)


def grow_terrain(tin, p, gp, max_distance, max_angle):
    keep_running = True
    gp = set()
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
                        v_angles = [math.degrees(math.asin(pdist / i)) for i in dvx]

                        # add point to dt and mark it as ground point, set flag to keep running
                        if max(v_angles) < max_angle:
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

    if alpha >= 0 and beta >= 0 and gamma >= 0:
        return True

    return False


# write numpy array to
def write_asc(grid, cell_size, fn, origin, depth):
    header = "NCOLS {}\nNROWS {}\nXLLCENTER {}\nYLLCENTER {}\nCELLSIZE {}\nNODATA_VALUE -9999".format(
                grid.shape[1], grid.shape[0], origin[0], origin[1], cell_size)

    grid = np.nan_to_num(grid, nan=-9999)
    np.savetxt(fn, grid, delimiter=' ', header=header, comments='', fmt='%1.{}f'.format(depth))


def tin_interp(tin, cell_size):
    # converts Delaunay mesh to grid using TIN interpolation

    convex_hull = [tin.get_point(p) for p in tin.convex_hull()]
    X, Y = [p[0] for p in convex_hull], [p[1] for p in convex_hull]
    X_min, X_max, Y_min, Y_max = int(min(X)), int(max(X)), int(min(Y)), int(max(Y))

    grid = np.empty(((X_max - X_min)//cell_size, (Y_max - Y_min)//cell_size))

    for x in range((X_max - X_min)//cell_size):
        for y in range((Y_max - Y_min)//cell_size):
            p = (x * cell_size + X_min, y * cell_size + Y_min)

            try:
                grid[x][y] = tin.interpolate_tin_linear(p[0], p[1])
            except OSError:
                grid[x][y] = np.nan

    return grid, (X_min, Y_min)


def idw_interp(tin, cell_size, radius, power):
    # converts Delaunay mesh to grid using IDW interpolation

    convex_hull = [tin.get_point(p) for p in tin.convex_hull()]
    X, Y = [p[0] for p in convex_hull], [p[1] for p in convex_hull]
    X_min, X_max, Y_min, Y_max = int(min(X)), int(max(X)), int(min(Y)), int(max(Y))

    grid = np.empty((int((X_max - X_min) // cell_size), int((Y_max - Y_min) // cell_size)))

    vertices = tin.all_vertices()
    tree = cKDTree([v[:2] for v in vertices])

    # iterate over grid, interpolating values based on corresponding point in TIN (p)
    for x in range(int((X_max - X_min) // cell_size)):
        for y in range(int((Y_max - Y_min) // cell_size)):
            p = (x * cell_size + X_min, y * cell_size + Y_min)
            nbs = [vertices[i] for i in tree.query_ball_point(p, radius)]

            if nbs:
                grid[x][y] = sum([i[2]/euclidean_distance(np.asarray(i[:2]), np.asarray(p))**power for i in nbs]) / \
                             sum([1/euclidean_distance(np.asarray(i[:2]), np.asarray(p))**power for i in nbs])
            else:
                grid[x][y] = np.nan

    return grid, (X_min, Y_min)


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
    scale = point_cloud.header.scale[0]
    print(point_cloud.header.min)
    print('- Flattening point cloud')
    gridded_pc = point_cloud_to_grid(point_cloud=point_cloud, tf=jparams['thinning-factor'],
                                     cell_size=int(jparams['gf-cellsize'] / scale))

    ground_points, unprocessed_points, ll_origin = gridded_pc[0], gridded_pc[1], gridded_pc[2]

    print('- Growing terrain')
    dt = startin.DT()
    dt.insert(list(ground_points))
    dt = grow_terrain(tin=dt, p=unprocessed_points, gp=ground_points,
                      max_distance=int(jparams['gf-distance'] / scale),
                      max_angle=jparams['gf-angle'])

    print('- Writing point cloud')
    with File(jparams['output-las'], mode='w', header=point_cloud.header) as out_file:
        gp = dt.all_vertices()[1:]
        out_file.X = [p[0] for p in gp]
        out_file.Y = [p[1] for p in gp]
        out_file.Z = [p[2] for p in gp]

    print('- Creating raster (TIN)\n\t- Interpolating (TIN)')
    dg = tin_interp(tin=dt, cell_size=int(jparams['grid-cellsize'] / scale))

    print('\t- Writing Esri Ascii (TIN)')
    write_asc(grid=np.rot90(dg[0]) * scale + point_cloud.header.min[2],
              cell_size=jparams['grid-cellsize'],
              fn=jparams['output-grid-tin'],
              origin=(point_cloud.header.min[0]+dg[1][0]*scale, point_cloud.header.min[1] + dg[1][1]*scale),
              depth=2)

    print('- Creating raster (IDW)\n\t- Interpolating (IDW)')
    ig = idw_interp(tin=dt, cell_size=int(jparams['grid-cellsize'] / scale),
                    radius=jparams['idw-radius'] / scale, 
                    power=jparams['idw-power'])

    print('\t- Writing Esri Ascii (IDW)')
    write_asc(grid=np.rot90(ig[0]) * scale + point_cloud.header.min[2],
              cell_size=jparams['grid-cellsize'],
              fn=jparams['output-grid-idw'],
              origin=(point_cloud.header.min[0]+ig[1][0]*scale, point_cloud.header.min[1]+ig[1][1]*scale),
              depth=2)

    return

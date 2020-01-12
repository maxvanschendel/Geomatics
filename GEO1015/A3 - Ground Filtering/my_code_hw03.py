# -- mycode_hw03.py
# -- GEO1015.2019--hw03
# -- [YOUR NAME]
# -- [YOUR STUDENT NUMBER]

import math

# for reading LAS files
from laspy.file import File
import numpy as np
import seaborn
# triangulation for ground filtering algorithm and TIN interpolation 
import startin

# kdtree for IDW interpolation
from scipy.spatial import cKDTree


def point_cloud_to_grid(point_cloud, cell_size, thinning_factor):
    print('Thinning grid')
    ground_points, discarded_points = set(), set()

    X, Y, Z = point_cloud.X[::thinning_factor], point_cloud.Y[::thinning_factor], point_cloud.Z[::thinning_factor]
    X_min, X_max, Y_min, Y_max = X.min(), X.max(), Y.min(), Y.max()

    stacked_array = np.vstack((X, Y, Z)).transpose()
    rect_array = np.zeros((X_max+1, Y_max+1))

    # converts 1D array to a 2D X, Y array with Z elements
    for x, y, z in stacked_array:
        if rect_array[x][y] == 0 or z < rect_array[x][y]:
            rect_array[x][y] = z

    # construct initial grid by slicing numpy array in chunks
    for x in range(int((X_max-X_min)/cell_size)):
        for y in range(int((Y_max-Y_min)/cell_size)):
            x_range, y_range = (x*cell_size, (x+1)*cell_size), (y*cell_size, (y+1)*cell_size)
            cell_points = rect_array[x_range[0]: x_range[1], y_range[0]: y_range[1]]

            nonzero_indices = np.nonzero(cell_points)
            nonzero_elements = cell_points[nonzero_indices]

            if nonzero_elements.size:
                min_index = np.argmin(nonzero_elements)

                ground_points.add((nonzero_indices[0][min_index]+x_range[0],
                                   nonzero_indices[1][min_index]+y_range[0],
                                   nonzero_elements[min_index]))

                non_min_x = np.delete(nonzero_indices[0], min_index)
                non_min_y = np.delete(nonzero_indices[1], min_index)
                non_min_elements = np.delete(nonzero_elements, min_index)

                # add points that aren't the local minimum to non-ground set
                for index, x_cur in enumerate(non_min_x):
                    y_cur = non_min_y[index]
                    discarded_points.add((x_cur + x_range[0], y_cur + y_range[0], non_min_elements[index]))

    return ground_points, discarded_points


def find_perp_distance(p, tri):
    # finds perpendicular distance of point to a triangle

    tri_normal = np.cross(tri[0] - tri[1], tri[2] - tri[1])
    tri_normal_unit = tri_normal / np.linalg.norm(tri_normal)

    vec2p = p - tri[0]
    vec2p_dot = np.dot(vec2p, tri_normal_unit)

    intersection = p - (tri_normal_unit * vec2p_dot)

    if point_in_tri(intersection, tri):
        return np.linalg.norm(intersection - p)

    return None


def point_in_tri(p, tri):
    # checks if point is on a face using barycentric coordinates

    u, v, w = tri[1] - tri[0], tri[2] - tri[0], p - tri[0]
    n = np.cross(u, v)

    gamma = np.dot(np.cross(u, w), n) / np.dot(n, n)
    beta = np.dot(np.cross(w, v), n) / np.dot(n, n)
    alpha = 1 - gamma - beta

    if 0 <= alpha <= 1 and 0 <= beta <= 1 and 0 <= gamma <= 1:
        return True

    return False


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

    gridded_pc = point_cloud_to_grid(point_cloud, jparams['gf-cellsize'], jparams['thinning-factor'])
    ground_points, unprocessed_points = gridded_pc[0], gridded_pc[1]

    print('Constructing initial Delaunay')
    delaunay = startin.DT()
    delaunay.insert(list(ground_points))
    delaunay.write_obj('./start.obj')

    print('Iteratively constructing ground')
    keep_running = True
    while keep_running:
        keep_running = False

        for x, y, z in unprocessed_points:
            if (x, y, z) not in ground_points:
                triangle_vertices = [delaunay.get_point(p) for p in delaunay.locate(x, y)]
                if triangle_vertices:
                    perp_distance = find_perp_distance(np.asarray([x, y, z]), np.asarray(triangle_vertices))

                    if perp_distance is not None and perp_distance < max_distance:
                        dvx = [np.linalg.norm(np.asarray([x, y, z]) - np.asarray(i)) for i in triangle_vertices]
                        v_angles = np.asarray([math.degrees(math.asin(perp_distance/i)) for i in dvx])

                        if v_angles.max() < max_angle:
                            delaunay.insert([(x, y, z)])
                            ground_points.add((x, y, z))

                            keep_running = True

    delaunay.write_obj('./end.obj')

    return

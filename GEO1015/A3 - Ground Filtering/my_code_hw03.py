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


def point_cloud_to_grid(point_cloud, cell_size):
    ground_points, discarded_points = set(), set()

    X, Y, Z = point_cloud.X, point_cloud.Y, point_cloud.Z
    X_min, X_max, Y_min, Y_max = X.min(), X.max(), Y.min(), Y.max()

    stacked_array = np.vstack((X, Y, Z)).transpose()
    rect_array = np.zeros((X_max+1, Y_max+1))

    # converts the 1D array to a 2D X, Y array with Z elements
    for x, y, z in stacked_array:
        if rect_array[x][y] == 0 or z < rect_array[x][y]:
            rect_array[x][y] = z

    # create initial grid made of empty elements
    gf_initgrid_shape = (int((X_max-X_min)/cell_size), int((Y_max-Y_min)/cell_size))

    # construct initial grid by slicing numpy array in chunks
    for x in range(gf_initgrid_shape[0]):
        for y in range(gf_initgrid_shape[1]):
            x_range, y_range = (x*cell_size, (x+1)*cell_size), (y*cell_size, (y+1)*cell_size)
            cell_points = rect_array[x_range[0]: x_range[1], y_range[0]: y_range[1]]

            nonzero_indices = np.nonzero(cell_points)
            nonzero_elements = cell_points[nonzero_indices]

            try:
                min_index = np.argmin(nonzero_elements)
                ground_points.add((nonzero_indices[0][min_index]+x_range[0],
                                   nonzero_indices[1][min_index]+y_range[0],
                                   nonzero_elements[min_index]))

                non_min_x = np.delete(nonzero_indices[0], min_index)
                non_min_y = np.delete(nonzero_indices[1], min_index)
                non_min_elements = np.delete(nonzero_elements, min_index)

                for index, x in enumerate(non_min_x):
                    y = non_min_y[index]

                    discarded_points.add((x + x_range[0], y + y_range[0], non_min_elements[index]))

            except ValueError:
                pass

    return ground_points, discarded_points


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

    gridded_pc = point_cloud_to_grid(point_cloud, jparams['gf-cellsize'])
    ground_points, unprocessed_points = gridded_pc[0], gridded_pc[1]
    init_delaunay = startin.DT()
    init_delaunay.insert(list(ground_points))

    for x, y, z in unprocessed_points:
        vert_proj_intersector = init_delaunay.locate(x, y)
        print('-----')
        print(vert_proj_intersector)


    return

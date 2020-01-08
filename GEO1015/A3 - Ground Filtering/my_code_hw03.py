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
    discarded_points = set()
    ground_points = set()

    X, Y, Z = point_cloud.X, point_cloud.Y, point_cloud.Z
    X_min, X_max, Y_min, Y_max = X.min(), X.max(), Y.min(), Y.max()

    stacked_array = np.vstack((X, Y, Z)).transpose()
    rect_array = np.zeros((X_max + 1, Y_max + 1))

    # converts the 1D array to a 2D X, Y array with Z elements
    for x, y, z in stacked_array:
        if rect_array[x][y] == 0:
            ground_points.add((x, y, z))
            rect_array[x][y] = z

        elif rect_array[x][y] > z:
            discarded_points.add((x, y, rect_array[x][y]))
            ground_points.add((x, y, z))
            rect_array[x][y] = z

        else:
            discarded_points.add((x, y, z))

    # create initial grid made of empty elements
    gf_initgrid_shape = (int((X_max - X_min) / cell_size), int((Y_max - Y_min) / cell_size))
    grid = np.empty(gf_initgrid_shape)

    # construct initial grid by slicing numpy array in chunks with cell size
    for x in range(gf_initgrid_shape[0]):
        for y in range(gf_initgrid_shape[1]):
            x_range, y_range = (x * cell_size, (x + 1) * cell_size), (y * cell_size, (y + 1) * cell_size)

            points_in_cell = rect_array[x_range[0]:x_range[1], y_range[0]: y_range[1]]
            pic_nozero = points_in_cell[points_in_cell != 0]

            if pic_nozero.size:
                grid[x][y] = pic_nozero.min()

    return grid, discarded_points, ground_points


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
    init_grid, unprocessed_points, ground_points = gridded_pc[0], gridded_pc[1], gridded_pc[2]

    init_delaunay = startin.DT(init_grid)





    return

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
    input_pc = File(jparams['input-las'], mode='r')
    cell_size = jparams['gf-cellsize']

    # get relevant data (coordinates) from input dataset
    X, Y, Z = input_pc.X, input_pc.Y, input_pc.Z
    X_min, X_max, Y_min, Y_max = X.min(), X.max(), Y.min(), Y.max()

    stacked_array = np.vstack((X, Y, Z)).transpose()
    rect_array = np.zeros((X_max + 1, Y_max + 1))

    # converts the 1D array to a 2D X, Y array with Z elements
    for x, y, z in stacked_array:
        if rect_array[x][y] == 0 or rect_array[x][y] > z:
            rect_array[x][y] = z

    # create initial grid made of empty elements
    gf_initgrid_shape = (int((X_max - X_min) / cell_size), int((Y_max - Y_min) / cell_size))
    initgrid = np.empty(gf_initgrid_shape)

    # construct initial grid by slicing numpy array in chunks with cell size
    # the minimum value of each chunk is written to the initial grid
    for x in range(gf_initgrid_shape[0]):
        for y in range(gf_initgrid_shape[1]):
            x_range, y_range = (x * cell_size, (x + 1) * cell_size), (y * cell_size, (y + 1) * cell_size)

            points_in_cell = rect_array[x_range[0]:x_range[1], y_range[0]: y_range[1]]
            pic_nozero = points_in_cell[points_in_cell != 0]

            if pic_nozero.size:
                initgrid[x][y] = pic_nozero.min()

    return

#-- mycode_hw02.py
#-- Assignment 02 GEO1015 (2019-2020)
#-- [YOUR NAME] 
#-- [YOUR STUDENT NUMBER]

import math
import rasterio
import numpy as np
import matplotlib.pyplot as plt


def get_outlets(elevation):
	outlets = np.ones(elevation.shape)

	# mask outlets on edge of array
	outlets[1:elevation.shape[0] - 1, 1:elevation.shape[1] - 1] = 0

	# mask all points that have a zero neighbours but are not zero themselves
	zero_elevation_neighbours = np.zeros(elevation.shape)
	for (x, y), item in np.ndenumerate(elevation):
		if 0 in elevation[x-1:x+2, y-1:y+2] and item != 0:
			zero_elevation_neighbours[x, y] = 1

	outlets += zero_elevation_neighbours
	outlets = np.array(outlets, dtype=bool)

	return outlets


def flow_direction(elevation):
	"""
    !!! TO BE COMPLETED !!!
     
    Function that computes the flow direction
     
    Input:
        elevation: grid with height values
    Output:
        returns grid with flow directions (encoded in whichever way you decide)
 
    """

	outlets = get_outlets(elevation)



def flow_accumulation(directions):
	"""
    !!! TO BE COMPLETED !!!
     
    Function that computes the flow accumulation
     
    Input:
        directions: grid with flow directions (encoded in whichever way you decide)
    Output:
        returns grid with accumulated flow (in number of upstream cells)
 
    """  

def write_directions_raster(raster, input_profile):
	"""
    !!! TO BE COMPLETED !!!
     
    Function that writes the output flow direction raster
     
    Input:
        raster: grid with flow directions (encoded in whichever way you decide)
        input_profile: profile of elevation grid (which you can copy and modify)
 
    """  

def write_accumulation_raster(raster, input_profile):
	"""
    !!! TO BE COMPLETED !!!
     
    Function that writes the output flow accumulation raster
     
    Input:
        raster: grid with accumulated flow (in number of upstream cells)
        input_profile: profile of elevation grid (which you can copy and modify)
 
    """  
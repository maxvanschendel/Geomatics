# -- mycode_hw02.py
# -- Assignment 02 GEO1015 (2019-2020)
# -- [YOUR NAME]
# -- [YOUR STUDENT NUMBER]

import math
import rasterio
import numpy as np
import matplotlib.pyplot as plt


# gets all points that are outlets, these are either:
# points on the border of the terrain
# points that have a neighbour with an elevation of zero but are not zero themselves
def get_outlets(elevation):
	outlets = np.ones(elevation.shape)

	# get points on border of array
	outlets[1:elevation.shape[0]-1, 1:elevation.shape[1]-1] = 0

	# get points that have a zero neighbours but are not zero themselves
	for (x, y), item in np.ndenumerate(elevation):
		if 0 in elevation[x-1:x+2, y-1:y+2] and item:
			outlets[x, y] = 1

	return outlets


# gets index of the lower elevation in the terrain
def lowest_elevation_index(elevation):
	return np.unravel_index(np.argmin(elevation, axis=None), elevation.shape)


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
	masked_terrain = np.ma.masked_array(elevation,outlets)
	ind = lowest_elevation_index(masked_terrain)


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

# -- mycode_hw02.py
# -- Assignment 02 GEO1015 (2019-2020)
# -- [YOUR NAME]
# -- [YOUR STUDENT NUMBER]

import math
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from heapq import *


def get_neighbours(data, x, y):
	return data[x-1:x+2, y-1:y+2]


# gets all points that are outlets, these are either:
# points on the border of the terrain
# points that have a neighbour with an elevation of zero but are not zero themselves
def get_outlets(elevation):
	outlets = np.ones(elevation.shape)

	# get points on border of array
	outlets[1:elevation.shape[0]-1, 1:elevation.shape[1]-1] = 0

	# get points that have a zero neighbours but are not zero themselves
	for (x, y), item in np.ndenumerate(elevation):
		if 0 in get_neighbours(elevation, x, y) and item:
			outlets[x, y] = 1

	return outlets


def minimum_elevation(neighbours):
	minimum = (math.inf, (None, None))
	for (x, y), item in np.ndenumerate(neighbours):
		if item < minimum[0] and item != 0.0 and (x, y) != (1, 1):
			minimum = (item, (x, y))
	return minimum


# converts 2d numpy array to priority queue
def np_to_pq(ar):
	# construct priority queue
	pq = []
	heapify(pq)

	for (x, y), item in np.ndenumerate(ar):
		heappush(pq, (item, (x, y)))

	return pq


def least_steep_uphill_slope(elevation, x, y):
	neighbours = get_neighbours(elevation, x, y)
	minimum_neighbour = minimum_elevation(neighbours)

	x_elev = x - minimum_neighbour[1][0] - 1
	y_elev = y - minimum_neighbour[1][1] - 1

	return elevation[x_elev][y_elev], (x_elev, y_elev)


def flow_direction(elevation):
	"""
	!!! TO BE COMPLETED !!!

	Function that computes the flow direction

	Input:
		elevation: grid with height values
	Output:
		returns grid with flow directions (encoded in whichever way you decide)
 
	"""

	flow_direction = np.zeros(elevation.shape)

	# get outlets
	outlets = get_outlets(elevation)
	masked_terrain = np.ma.masked_array(elevation, outlets)

	# construct priority queue from outlets
	priority_queue = np_to_pq(masked_terrain)

	searching = True
	while searching:
		cur = heappop(priority_queue)

		climbing = True
		while climbing:
			cur = least_steep_uphill_slope(elevation, cur[1][0], cur[1][1])

		searching = False



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

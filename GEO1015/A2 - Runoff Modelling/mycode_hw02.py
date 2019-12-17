# -- mycode_hw02.py
# -- Assignment 02 GEO1015 (2019-2020)
# -- Max van Schendel
# -- 4384644

import math
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import time
from heapq import *
import random
from matplotlib.colors import LogNorm
import seaborn as sn
from random import shuffle

# get neighbouring pixels by slicing np array
def get_neighbours(data, x, y):
    return data[x - 1:x + 2, y - 1:y + 2]


# gets all points that are outlets, these are either:
# points on the border of the terrain
# points that have a neighbour with an elevation of zero but are not zero themselves
def get_outlets(elevation):
    outlets = np.ones(elevation.shape)

    # get points on border of array
    outlets[1:elevation.shape[0] - 1, 1:elevation.shape[1] - 1] = 0

    # get points that have a zero neighbours but are not zero themselves
    for (x, y), item in np.ndenumerate(elevation):
        if 0 in get_neighbours(elevation, x, y) and item:
            outlets[x, y] = 1

    return outlets


# get local index and value of neighbour with lowest elevation
def minimum_elevation(neighbours):
    if neighbours.size:
        ind = np.unravel_index(np.argmin(neighbours, axis=None), neighbours.shape)

        if tuple(ind) != (1, 1):
            val = neighbours[ind[0]][ind[1]]
            if val != 0:
                return val, ind


# converts 2d numpy array to priority queue
def np_to_pq(ar):
    pq = []
    heapify(pq)

    for (x, y), item in np.ndenumerate(ar):
        if item == 0:
            heappush(pq, (item, (x, y)))

    return pq


# map indices in sliced array to larger array; used for getting values of neighbour elements
def local_to_global_coords(local_coords, global_center):
    return global_center[0] + local_coords[0] - 1, global_center[1] + local_coords[1] - 1


def minimum_neighbour(elevation, x, y):
    neighbours = get_neighbours(elevation, x, y)
    min_neighbour = minimum_elevation(neighbours)

    if min_neighbour is not None:
        global_coords = local_to_global_coords(min_neighbour[1], (x, y))
        return elevation[global_coords[0]][global_coords[1]], global_coords

    else:
        return None


def find_upstream_pixels(flow_directions, coords):
    flow_direction_decode = {(0, 0): 1, (1, 0): 2, (2, 0): 3,
                             (0, 1): 4, (1, 1): 5, (2, 1): 6,
                             (0, 2): 7, (1, 2): 8, (2, 2): 9}

    upstream_pixels = []
    neighbours = get_neighbours(flow_directions, coords[0], coords[1])

    for index in flow_direction_decode:
        try:
            val_at_index = neighbours[index[1]][index[0]]
        except IndexError:
            val_at_index = None

        if flow_direction_decode[index] == val_at_index:
            upstream_pixels.append(index)

    return upstream_pixels


def flow_direction(elevation):
    """
	Function that computes the flow direction

	Input:
		elevation: grid with height values
	Output:
		returns grid with flow directions (encoded in whichever way you decide)
	"""

    # flow direction array which the results will be written to
    flow_dir = np.zeros(elevation.shape)

    # get outlets and construct priority queue
    outlets = get_outlets(elevation)
    priority_queue = np_to_pq(np.ma.masked_array(elevation, outlets))

    # get the current pixel in the array, in this case the first outlet in the priority queue
    cur = heappop(priority_queue)

    # keep searching until priority queue is empty
    while priority_queue:

        i = 1
        for coords, item in np.ndenumerate(get_neighbours(elevation, cur[1][0], cur[1][1])):
            coords = local_to_global_coords(coords, cur[1])

            # check if item has already been calculated and if it not in the pq yet
            if flow_dir[coords[0]][coords[1]] == 0 and (item, coords) not in priority_queue and (item, coords):
                heappush(priority_queue, (item, coords))
                flow_dir[coords[0]][coords[1]] = i

            i += 1

        # neighbour with lowest elevation value
        min_neighbour = minimum_neighbour(elevation, cur[1][0], cur[1][1])

        # check if lowest neighbour is invalid or if its flow direction has already been set
        if min_neighbour is None or flow_dir[min_neighbour[1][0]][min_neighbour[1][1]] != 0:
            cur = heappop(priority_queue)
        else:
            cur = min_neighbour

    return flow_dir


def flow_accumulation(directions):
    """
	Function that computes the flow accumulation

	Input:
		directions: grid with flow directions (encoded in whichever way you decide)
	Output:
		returns grid with accumulated flow (in number of upstream cells)
	"""

    accumulated_flow = np.ones(directions.shape)
    # for every pixel, count number of pixels that are uphill from it

    for coords, item in np.ndenumerate(directions):
        processed = set([])
        # initialize search queue
        search_queue = [(0, coords)]
        heapify(search_queue)

        # processed pixels are marked to prevent walking in circles

        c = 0
        while search_queue:
            cur = heappop(search_queue)
            upstream_pixels = find_upstream_pixels(directions, cur[1])
            for i in upstream_pixels:
                global_coords = local_to_global_coords(i, cur[1])
                if global_coords not in processed:
                    heappush(search_queue, (c, global_coords))
                    processed.add(global_coords)

                    accumulated_flow[coords[0]][coords[1]] += 1
                    c += 1

    return accumulated_flow


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

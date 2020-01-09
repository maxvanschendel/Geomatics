# -- mycode_hw02.py
# -- Assignment 02 GEO1015 (2019-2020)
# -- Max van Schendel
# -- 4384644

import rasterio
import numpy as np
from heapq import *
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
        min_neighbours = np.argwhere(neighbours==neighbours.min())

        if len(min_neighbours) == 1:
            min = min_neighbours.tolist()[0]
            if min != [1,1]:
                val = neighbours[min[0]][min[1]]
                return val, min
            else:
                return None

        else:
            min_nb_nocenter = list(min_neighbours.tolist())
            try:
                min_nb_nocenter.remove([1,1])
            except ValueError:
                pass

            shuffle(min_nb_nocenter)
            min = min_nb_nocenter[0]

        val = neighbours[min[0]][min[1]]

        if val != 0:
            return val, min
        else:
            return None


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


# get neighbouring pixel with the lowest elevation value in local coordinates (center = (1, 1))
def minimum_neighbour(elevation, x, y):
    neighbours = get_neighbours(elevation, x, y)
    min_neighbour = minimum_elevation(neighbours)

    if min_neighbour is not None:
        global_coords = local_to_global_coords(min_neighbour[1], (x, y))
        return elevation[global_coords[0]][global_coords[1]], global_coords

    else:
        return None


# find neighbouring pixels with flow direction pointing towards the center
def find_upstream_pixels(flow_directions, coords):
    flow_direction_decode = {(0, 0): 3, (1, 0): 4, (2, 0): 5,
                             (0, 1): 6, (1, 1): 7, (2, 1): 8,
                             (0, 2): 9, (1, 2): 10, (2, 2): 11}

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
        i = 3

        for coords, item in np.ndenumerate(get_neighbours(elevation, cur[1][0], cur[1][1])):
            coords = local_to_global_coords(coords, cur[1])

            # check if item has already been calculated and if it not in the pq yet
            if flow_dir[coords[0]][coords[1]] == 0 and (item, coords) not in priority_queue and (item, coords):
                heappush(priority_queue, (item, coords))
                flow_dir[coords[0]][coords[1]] = i

            i += 1

        min_neighbour = minimum_neighbour(elevation, cur[1][0], cur[1][1])
        # check if lowest neighbour is invalid or if its flow direction has already been set
        if min_neighbour is None or flow_dir[min_neighbour[1][0]][min_neighbour[1][1]] != 0:
            cur = heappop(priority_queue)
        else:
            cur = min_neighbour

    flow_dir[elevation == 0] = 1
    flow_dir[outlets == 1] = 2

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

    # iterate over all pixels
    for coords, item in np.ndenumerate(directions):
        # ignore 0-elevation pixels
        if item != 1:
            search_stack = [coords]
            processed = set()
            while search_stack:
                cur = search_stack.pop()    # get pixel from top of stack
                upstream_pixels = find_upstream_pixels(directions, cur)  # find neighbour with flow direction towards cur

                for i in upstream_pixels:   # iterate over upstream neighbours
                    upstream_coords = local_to_global_coords(i, cur)
                    if upstream_coords not in processed:
                        search_stack.append(upstream_coords)

                    accumulated_flow[coords[0]][coords[1]] += 1 # increment flow count by 1 for every upstream pixel

                processed.add(cur)

    return accumulated_flow


def write_directions_raster(raster, input_profile):
    """
	!!! TO BE COMPLETED !!!

	Function that writes the output flow direction raster

	Input:
		raster: grid with flow directions (encoded in whichever way you decide)
		input_profile: profile of elevation grid (which you can copy and modify)
 
	"""
    prof = input_profile
    prof.update(
        dtype=rasterio.uint8,
        nodata=0)

    with rasterio.open('./flow_dir.tif', 'w', **input_profile) as dst:
        dst.write(raster.astype(rasterio.uint8), 1)


def write_accumulation_raster(raster, input_profile):
    """
	!!! TO BE COMPLETED !!!

	Function that writes the output flow accumulation raster

	Input:
		raster: grid with accumulated flow (in number of upstream cells)
		input_profile: profile of elevation grid (which you can copy and modify)
 
	"""

    prof = input_profile
    prof.update(
        dtype=rasterio.uint32)

    with rasterio.open('./flow_acc.tif', 'w', **input_profile) as dst:
        dst.write(raster.astype(rasterio.uint32), 1)

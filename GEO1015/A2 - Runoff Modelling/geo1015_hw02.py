#-- geo1015_hw02.py
#-- hw02 GEO1015/2019
#-- 2019-11-18

#------------------------------------------------------------------------------
# DO NOT MODIFY THIS FILE!!!
#------------------------------------------------------------------------------

import json, sys
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns
import time
import mycode_hw02

def main():
	# Open file
	start_time=time.clock()
	with rasterio.open('tasmania_small.tif') as src:
		elevation = np.array(src.read()[0])
		profile = src.profile

	# # Plot input
	# plt.figure(1)
	# im = plt.imshow(elevation)
	# plt.colorbar(im)
	# plt.show()

	# Compute flow directions
	directions = mycode_hw02.flow_direction(elevation)

	# # Plot directions and write them to a file
	plt.figure(2)
	im = plt.imshow(directions)
	plt.colorbar(im)
	plt.show()
	mycode_hw02.write_directions_raster(directions, profile)

	# Compute flow accumulation
	accumulation = mycode_hw02.flow_accumulation(directions)

	# Plot accumulation and write them to a file
	plt.figure(3)
	im = plt.imshow(accumulation, norm=LogNorm())
	plt.colorbar(im)
	plt.show()
	mycode_hw02.write_accumulation_raster(accumulation, profile)
	print(time.clock()-start_time)

if __name__ == '__main__':
  main()


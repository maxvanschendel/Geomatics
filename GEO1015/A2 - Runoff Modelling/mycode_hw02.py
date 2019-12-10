#-- mycode_hw02.py
#-- Assignment 02 GEO1015 (2019-2020)
#-- [YOUR NAME] 
#-- [YOUR STUDENT NUMBER]

import math
import rasterio
import numpy as np

def flow_direction(elevation):
	"""
    !!! TO BE COMPLETED !!!
     
    Function that computes the flow direction
     
    Input:
        elevation: grid with height values
    Output:
        returns grid with flow directions (encoded in whichever way you decide)
 
    """  

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
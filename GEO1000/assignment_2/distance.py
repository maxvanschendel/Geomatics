# GEO1000 - Assignment 2
# Authors:
# Studentnumbers:

from math import radians, cos, sin, asin, sqrt


def haversin(latlonl1, latlon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    
    arguments:
        latlon1 - tuple (lat, lon)
        latlon2 - tuple (lat, lon)

    returns:
        distance between the two coordinates (not rounded)
    """
    latlon1_rad = [radians(i) for i in latlonl1]
    latlon2_rad = [radians(i) for i in latlon2]

    delta_lat = latlon2_rad[0] - latlon1_rad[0]
    delta_long = latlon2_rad[1] - latlon1_rad[1]

    return 6371.0 * 2*asin(sqrt(sin(delta_lat/2)**2 + cos(latlon1_rad[0]) * cos(latlon2_rad[0]) * sin(delta_long/2)**2))


def _test():
    # You can use this function to test the distance calculation

    # distance from amsterdam to london, should be 340.4
    print(haversin([51.5074, 0.1278], [52.3667, 4.8945]))


if __name__ == "__main__":
    _test()

# GEO1000 - Assignment 2
# Authors:
# Studentnumbers:


def dd_dms(decdegrees):
    """Returns tuple (degrees, minutes, seconds) for a value in decimal degrees

    Arguments:

    decdegrees -- float that represents a latitude or longitude value
    
    returns:
    
    tuple of floats (degrees, minutes, seconds) 
    """
    pass


def format_dms(dms, is_latitude):
    """Returns a formatted string for *one* part of the coordinate.

    Arguments:

    dms -- tuple of floats (degrees, minutes, seconds)
           that represents a latitude or longitude value
    is_latitude -- boolean that specifies whether ordinate is latitude or longitude

    If is_latitude == True dms represents latitude (north/south)
    If is_latitude == False dms represents longitude (east/west)

    returns:
    
    Formatted string
    """
    pass


def format_dd_as_dms(coordinate):
    """Returns a formatted string for a coordinate

    Arguments:

    coordinate -- 2-tuple: (latitude, longitude)

    returns:
    
    Formatted string
    """
    pass


def _test():
    """Test whether the format_dd_as_dms function works correctly

    Expected output:

N   0°  0'  0.0000", E   0°  0'  0.0000
N  52°  0'  0.0000", E   4° 19' 43.3200"
S  52°  0'  0.0000", E   4° 19' 43.3200"
N  52°  0'  0.0000", W   4° 19' 43.3200"
S  52°  0'  0.0000", W   4° 19' 43.3200"
N  45°  0'  0.0000", E 180°  0'  0.0000"
S  45°  0'  0.0000", W 180°  0'  0.0000"
S  50° 27' 24.1200", E   4° 19' 43.3200"

    (note, in VS Code you can view the whitespace characters in a text file
     by switching on the option View → Render Whitespace)
    """
    coordinates = ((0.0, 0.0), 
        (52, 4.3287), 
        (-52, 4.3287), 
        (52, -4.3287), 
        (-52, -4.3287), 
        (45.0, 180.0), 
        (-45.0, -180.0), 
        (-50.4567, 4.3287))
    for coordinate in coordinates:
        print(format_dd_as_dms(coordinate))


if __name__ == "__main__":
    _test()

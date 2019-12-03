# GEO1000 - Assignment 2

from urllib.request import urlopen, URLError
from urllib.parse import quote
import json


def nominatim(place):
    """Geocode a place name, returns tuple with latitude, longitude 
    returns empty tuple if no place found, or something went wrong.

    Geocoding happens by means of the Nominatim service.
    Please be aware of the rules of using the Nominatim service:

    https://operations.osmfoundation.org/policies/nominatim/

    arguments:
        place - string
    
    returns:
        2-tuple of floats: (latitude, longitude) or
        empty tuple in case of failure
    """
    url = "http://nominatim.openstreetmap.org/search/"
    params = "?format=json"
    try:
        req = urlopen(url + quote(place) + params)
        lst = json.loads(req.read())
        loc = map(float, [lst[0]['lat'], lst[0]['lon']])
    except:
        # when something goes wrong,
        # e.g. no place found or timeout: return empty tuple
        return ()
    # otherwise, return the found WGS'84 coordinate
    return tuple(loc)


def _test():
    # Expected behaviour
    # unknown place leads to empty tuple
    assert nominatim("unknown xxxyyy") == ()
    # delft leads to coordinates of delft
    assert nominatim("delft") == (51.9994572, 4.362724538544)


if __name__ == "__main__":
    _test()

# GEO1000 - Assignment 2
# Authors:
# Studentnumbers:


from nominatim import nominatim
from dms import format_dd_as_dms
from distance import haversin


def query():
    """Query the WGS'84 coordinates of 2 places and compute the distance
    between them.

    A sample run of the program:

I will find the distance for you between 2 places.
Enter place 1? Delft
Enter place 2? Bratislava
Coordinates for Delft: N  51° 59' 58.0459", E   4° 21' 45.8083"
Coordinates for Bratislava: N  48°  9'  6.1157", E  17°  6' 33.5027"
The distance between Delft and Bratislava is 1003.4 km
Enter place 1? 
Enter place 2? quit
Bye bye.

    And another run:

I will find the distance for you between 2 places.
Enter place 1? where is this place?
Enter place 2? 
I did not understand this place: where is this place?
I did not understand this place: 
Enter place 1? quit
Enter place 2? 
Bye bye.

    """
    pass


if __name__ == "__main__":
    query()

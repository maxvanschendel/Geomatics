# GEO1000 - Assignment 2
# Authors: Max van Schendel
# Studentnumbers: 4384644


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
    print("I will find the distance for you between 2 places.")

    running = True
    while running:
        print("Enter place 1? ", end='')
        place1 = input()
        print("Enter place 2? ", end='')
        place2 = input()

        if place1 == 'quit' or place2 == 'quit':
            print('Bye bye.')
            running = False

        else:

            place1_coords = nominatim(place1)
            place2_coords = nominatim(place2)

            if not place1_coords:
                print("I did not understand this place: {}".format(place1))

            if not place2_coords:
                print("I did not understand this place: {}".format(place2))

            else:
                print('Coordinates for {}:'.format(place1), format_dd_as_dms(place1_coords))
                print('Coordinates for {}:'.format(place2), format_dd_as_dms(place2_coords))
                print('The distance between {} and {} is {} km'.format(place1, place2, round(haversin(place1_coords, place2_coords),1)))


if __name__ == "__main__":
    query()

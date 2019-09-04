#  GEO1000  - Assignment  0
#  Authors: Max van Schendel
#  Studentnumbers : 4384644


def temp_windchill(temp_in_c, windspeed_in_kmh):
    return 13.12 + (0.6215 * temp_in_c) - (11.7*(windspeed_in_kmh ** 0.16)) + (0.3965 * (windspeed_in_kmh ** 0.16))


print(temp_windchill(5.0,  10.0))
print(temp_windchill(-1.0,  35.0))

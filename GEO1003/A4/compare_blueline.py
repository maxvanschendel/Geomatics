import pynmea2
import statistics
import csv
import folium


def calculate_rms(data):
    return statistics.mean([i**2 for i in data])**0.5


def dms_dd(degrees, minutes=0, seconds=0.0):
    if degrees >= 0:
        decimal = degrees + minutes/60.0 + seconds/3600.0
    else:
        decimal = degrees - minutes/60.0 - seconds/3600.0
    return decimal


with open('bl1.txt') as txtfile:
    bl1_nmea = list(filter(lambda x: 'GGA' in x, txtfile.readlines()))

with open('bl2.txt') as txtfile:
    bl2_nmea = list(filter(lambda x: 'GGA' in x, txtfile.readlines()))

with open('gps latitude.csv') as csvfile:
    gps_data = [i for i in csv.reader(csvfile, delimiter=',')]

    gps_lat = [i[1].replace('Â°', ',').replace('\'',',').replace('\"',',').split(',')[0:3] for i in gps_data][1:]
    gps_lon = [i[2].replace('Â°', ',').replace('\'', ',').replace('\"', ',').split(',')[0:3] for i in gps_data][1:]

    gps_lat_dd = [dms_dd(int(i[0]), int(i[1]), float(i[2])) for i in gps_lat]
    gps_lon_dd = [dms_dd(int(i[0]), int(i[1]), float(i[2])) for i in gps_lon]

    gps_latlon = list(zip(gps_lat_dd,gps_lon_dd))


print('bl1_rmse:', calculate_rms([pynmea2.parse(i).latitude - 52 for i in bl1_nmea]))
print('bl2_rmse:', calculate_rms([pynmea2.parse(i).latitude - 52 for i in bl2_nmea]))
print('gps_rmse:', calculate_rms([i-52 for i in gps_lat_dd]))

m = folium.Map(location=gps_latlon[0], tiles='cartodbdark_matter', zoom_start=20)
bl1_group = folium.FeatureGroup(name='Mobile: Stops along line')
bl2_group = folium.FeatureGroup(name='Mobile: Walking along line')
gps_group = folium.FeatureGroup(name='Commercial')
line_group = folium.FeatureGroup(name='Real 52nd latitude')


folium.PolyLine([(52.0,4),(52.0,5.0)], color = 'white', dash_array='20,20').add_to(line_group)

for i in enumerate(gps_latlon):
    folium.CircleMarker(location=i[1],radius=0.7,popup=str(i)).add_to(gps_group)

folium.PolyLine(gps_latlon,).add_to(gps_group)


bl1_latlon = list(zip([pynmea2.parse(i).latitude for i in bl1_nmea],[pynmea2.parse(i).longitude for i in bl1_nmea]))
bl2_latlon = list(zip([pynmea2.parse(i).latitude for i in bl2_nmea],[pynmea2.parse(i).longitude for i in bl2_nmea]))


for i in enumerate(bl1_latlon):
    folium.CircleMarker(location=i[1], color = 'red', radius=0.7, popup=str(i)).add_to(bl1_group)

folium.PolyLine(bl1_latlon, color='red').add_to(bl1_group)

for i in enumerate(bl2_latlon):
    folium.CircleMarker(location=i[1], color = 'green', radius=0.7).add_to(bl2_group)

folium.PolyLine(bl2_latlon, color='green').add_to(bl2_group)

# add elements to map
m.add_child(line_group)
m.add_child(bl1_group)
m.add_child(bl2_group)
m.add_child(gps_group)

m.add_child(folium.map.LayerControl())
m.save('blue_line_measurements.html')


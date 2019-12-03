import pandas as pd
import folium


# load data
df = pd.read_csv('trips.csv')

# get bbox and data center
bbox = (df['start_lon'].min(), df['start_lon'].max(), df['start_lat'].min(), df['start_lat'].max())
center = [(bbox[3] + bbox[2]) / 2, (bbox[1] + bbox[0]) / 2]

# create map at appropriate zoom level
m = folium.Map(location=center, tiles='cartodbdark_matter', zoom_start=15)

# create groups for each element so they can be turned off
trips_group = folium.FeatureGroup(name='Trips')
start_group = folium.FeatureGroup(name='Departures')
end_group = folium.FeatureGroup(name='Arrivals')

# plotting parameters
trip_minimum, stay_minimum = 15, 15

# TRIPS
# get amount of time each trip is taken
df['trip'] = df['start_lat'].map(str) + ',' + \
             df['start_lon'].map(str) + ',' + \
             df['end_lat'].map(str) + ',' + \
             df['end_lon'].map(str)

count = df['trip'].value_counts()
tuples = [list((x.split(','), y)) for x, y in count.items()]

# remove direction from graph: count(a,b) + count(b,a) -> weight
combined_tuples = []
pos = [i[0] for i in tuples]
for i in tuples:
    try:
        mirror = [i[0][2], i[0][3], i[0][0], i[0][1]]
        imirror = pos.index(mirror)
    except ValueError:
        combined_tuples.append(i)
        continue

    combined_tuples.append([i[0], i[1] + tuples[imirror][1]])

# place trips on map.
# Lines are weighted by the square root of occurence divided by a factor
for i in combined_tuples:
    start = (float(i[0][0]), float(i[0][1]))
    end = (float(i[0][2]), float(i[0][3]))
    count = i[1]
    if count > trip_minimum:
        folium.PolyLine([start, end],
                        weight=(count**(1/2))/ 4,
                        color='white',
                        popup='COUNT: ' + str(count),
                        legend_name='trips',
                        opacity=0.2).add_to(trips_group)

# STAYS
df['start_loc'] = df['start_lat'].map(str) + ',' + df['start_lon'].map(str)
df['end_loc'] = df['end_lat'].map(str) + ',' + df['end_lon'].map(str)

start_count = df['start_loc'].value_counts()
end_count = df['end_loc'].value_counts()
start_tuples = [tuple((x, y)) for x, y in start_count.items()]
end_tuples = [tuple((x, y)) for x, y in end_count.items()]

# place markers on nodes of graph
for i in enumerate(start_tuples):
    if i[1][1] > stay_minimum:
        # create stay markers
        folium.CircleMarker(
            location=i[1][0].split(','),
            radius=i[1][1] / 80,
            weight=1,
            fill=True,
            color='lightblue',
            popup='COUNT: ' + str(i[1][1])
        ).add_to(start_group)

        # create depart markers
        folium.CircleMarker(
            location=end_tuples[i[0]][0].split(','),
            radius=end_tuples[i[0]][1] / 80,
            weight=1,
            fill=True,
            dash_array='20,20',
            color='lightblue',
            popup='COUNT: ' + str(end_tuples[i[0]][1])
        ).add_to(end_group)

# add elements to map
m.add_child(trips_group)
m.add_child(start_group)
m.add_child(end_group)
m.add_child(folium.map.LayerControl())

# save map to disk
m.save('trips.html')

import pandas as pd
import folium
import math


# visualize how many devices each scanner has registered
def vis_scanners():
    # load data
    df = pd.read_csv('output.csv')

    # get bbox and data center
    bbox = (df['lon'].min(), df['lon'].max(), df['lat'].min(), df['lat'].max())
    center = [(bbox[3] + bbox[2])/2, (bbox[1] + bbox[0])/2]

    # get amount of times each scanner has registered a bluetooth device
    df['latlon'] = (df['lat'].map(str) + ',' + df['lon'].map(str))
    count = df['latlon'].value_counts()
    tuples = [tuple((x, y)) for x, y in count.items()]

    # create map at appropriate zoom level
    m = folium.Map(location=center, tiles='cartodbdark_matter', zoom_start=15)

    # place markers on map
    for i in tuples:
        folium.CircleMarker(
            location=i[0].split(','),
            radius=i[1] / 100,
            fill=True,
            color='crimson',
            popup='COUNT: ' + str(i[1])
        ).add_to(m)

        folium.CircleMarker(location=i[0].split(','), radius=1, fill=True, color='crimson').add_to(m)

    # save map to disk
    m.save('scanners.html')


# visualize how often each unique trip is taken.
def vis_trips():
    # load data
    df = pd.read_csv('trips.csv')

    # get bbox and data center
    bbox = (df['start_lon'].min(), df['start_lon'].max(), df['start_lat'].min(), df['start_lat'].max())
    center = [(bbox[3] + bbox[2]) / 2, (bbox[1] + bbox[0]) / 2]

    # get amount of times each scanner has registered a bluetooth device
    df['trip'] = df['start_lat'].map(str) + ',' + df['start_lon'].map(str) + ',' + df['end_lat'].map(str) + ',' + df[
        'end_lon'].map(str)

    # get occurence of each trip
    count = df['trip'].value_counts()
    tuples = [list((x.split(','), y)) for x, y in count.items()]

    # create map at appropriate zoom level
    m = folium.Map(location=center, tiles='cartodbdark_matter', zoom_start=15)

    # place trips on map.
    # Lines are weighted by the square root of occurence divided by a factor
    for i in tuples:
        start = (float(i[0][0]), float(i[0][1]))
        end = (float(i[0][2]), float(i[0][3]))
        count = i[1]

        folium.PolyLine([start, end], weight=math.sqrt(count / 30), popup='COUNT: ' + str(count), legend_name='trips').add_to(m)

    m.save('trips.html')


if __name__ == '__main__':
    vis_scanners()
    vis_trips()

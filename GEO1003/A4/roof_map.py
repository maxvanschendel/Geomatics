import folium
import csv
import pyproj

with open('gps surface.csv') as csvfile:
    gps_data = [i[1:3] for i in csv.reader(csvfile, delimiter=',')][2:]




inProj = pyproj.Proj(init='epsg:28992')
outProj = pyproj.Proj(init='epsg:4326')

reproj = [pyproj.transform(inProj,outProj,i[0],i[1]) for i in gps_data]
reproj = [(i[1],i[0]) for i in reproj]


m = folium.Map(location=reproj[0], tiles='cartodbdark_matter', zoom_start=15)
reproj.append(reproj[0])
folium.Polygon(reproj,fill=True).add_to(m)

for i in enumerate(reproj):
    folium.CircleMarker(location=i[1],radius=0.7, popup=str(i)).add_to(m)

m.save('roof_map.html')
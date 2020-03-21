
from shapely.geometry import Point, Polygon
from json import load
import laspy
import numpy as np
from multiprocessing import Pool
from time import perf_counter
from scipy import spatial
import matplotlib.pyplot as plt


def read_geojson(fn):
    with open(fn) as file_data:
        return load(file_data)


def read_laz(fn):
    return laspy.file.File(fn, mode='r')


def get_bbox_points(f, points_minimum, points):
    f_bbox = np.array([[f[:, 0].min(), f[:, 0].max()],
                       [f[:, 1].min(), f[:, 1].max()]])

    frange = ((f_bbox - points_minimum) // grid_size).astype(int)
    row = grid[frange[0][0]:frange[0][1] + 1]
    points_indices = flatten([i[frange[1][0]:frange[1][1] + 1] for i in row])

    return [points[point] for cell in points_indices for point in cell]


def points_in_poly(poly, points):
    points_in_polygon = []

    for p in points:
        if Point(p[0:2]).within(Polygon(poly)):
            points_in_polygon.append(p)

    return points_in_polygon


def flatten(ar):
    return [i for j in ar for i in j]


if __name__ == '__main__':
    # configuration parameters
    footprint_file = './tudcampus.geojson'
    point_cloud_file = './ahn3_clipped_thinned.las'
    grid_size = np.array([4000, 4000])
    process_count = 12
    thinning_factor = 25
    neighbour_count = 10

    # DATA INPUT
    # read input data
    print('Reading data')
    footprints = read_geojson(footprint_file)
    points_raw = read_laz(point_cloud_file)

    # SPATIAL INDEX
    # construct list of lists to be used as spatial index
    print('Building spatial index')

    # two-dimensional bounding box of point cloud in x and y axes

    # points_raw = filter(lambda x: x[5] == 6, points_raw)

    points = np.column_stack((points_raw.X, points_raw.Y, points_raw.Z, points_raw.classification))[::thinning_factor]

    building_points = points[points[:, 3] == 6]

    buildings_bbox = np.array([[building_points[:, 0].min(), building_points[:, 0].max()],
                               [building_points[:, 1].min(), building_points[:, 1].max()]])

    x_cells = (buildings_bbox[0][1] - buildings_bbox[0][0]) // grid_size[0]
    y_cells = (buildings_bbox[1][1] - buildings_bbox[1][0]) // grid_size[1]

    grid = [[[] for x in range(y_cells.astype(int) + 1)] for y in range(x_cells.astype(int) + 1)]
    points_shifted = (building_points[:, :2] - buildings_bbox[:, :1].transpose()).astype(int) // grid_size

    for i, p in enumerate(points_shifted):
        grid[p[0]][p[1]].append(i)

    start = perf_counter()
    # POINTS IN POLYGON
    # for each building footprint, finds the points that fall within it

    print('Finding intersection candidates')
    footprint_coords = [(np.concatenate(np.array(i['geometry']['coordinates'])) * 1000) for i in footprints['features']]
    candidate_points = [get_bbox_points(f, buildings_bbox[:, :1], building_points) for f in footprint_coords]

    print('Finding intersections')
    p = Pool(processes=process_count)
    points_in_polygons = p.starmap(points_in_poly, zip(footprint_coords, candidate_points))
    p.close()
    p.join()

    median_heights = [i[len(i) // 2][2] if len(i) else 0 for i in points_in_polygons]

    print(perf_counter() - start)
    start = perf_counter()

    tree_points = points[points[:, 3] == 6]
    flat_pip = flatten(points_in_polygons)
    plt.scatter(x=[point[0] for point in tree_points],
                y=[point[1] for point in tree_points],
                c=[point[2] for point in tree_points],
                s=1)

    tree = spatial.KDTree(np.array(tree_points))
    neighbours = map(lambda p: map(lambda n: tree.data[n], tree.query(p, neighbour_count)), tree.data)

    # plot results
    print('Plotting')
    for i, f in enumerate(footprint_coords):
        plt.plot(*Polygon(f).exterior.xy, c=[0, 0, 0, 1])

    flat_pip = flatten(points_in_polygons)
    plt.scatter(x=[point[0] for point in flat_pip],
                y=[point[1] for point in flat_pip],
                c=[point[2] for point in flat_pip],
                s=1)
    plt.show()


    pass
from shapely.geometry import Point, Polygon
from json import load
import laspy
import numpy as np
from multiprocessing import Pool
from time import perf_counter
from scipy import spatial
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

### BUILDING GENERATOR FUNCTIONS ###
# read building footprint file
def read_geojson(fn):
    with open(fn) as file_data:
        return load(file_data)

# read point cloud file
def read_laz(fn):
    return laspy.file.File(fn, mode='r')

# get all points that fall within a rectangle
def get_bbox_points(f, points_minimum, points):
    f_bbox = np.array([[f[:, 0].min(), f[:, 0].max()],
                       [f[:, 1].min(), f[:, 1].max()]])

    frange = ((f_bbox - points_minimum) // grid_size).astype(int)
    row = grid[frange[0][0]:frange[0][1] + 1]
    points_indices = flatten([i[frange[1][0]:frange[1][1] + 1] for i in row])

    return [points[point] for cell in points_indices for point in cell]

# find all points that are within a polygon
def points_in_poly(poly, points):
    points_in_polygon = []

    for p in points:
        if Point(p[0:2]).within(Polygon(poly)):
            points_in_polygon.append(p)

    return points_in_polygon

# flatten list of lists
def flatten(ar):
    return [i for j in ar for i in j]

# principal component analysis
def pca(nbs):
    return np.linalg.eig(np.cov(nbs - np.mean(nbs)))

### TREE GENERATOR FUNCTIONS ###
# generate tree crown from points
def crown_from_points(pts):
    if len(pts) > 3:
        hull = spatial.ConvexHull(np.stack(pts))

        return hull.simplices, np.array([[hull.points[f[0]], hull.points[f[1]], hull.points[f[2]]] for f in hull.simplices])

# generate tree trunk from tree crown
def trunk_from_crown(crown):
    crown_centroid = np.mean(np.mean(crown, axis=0), axis=0)
    max_radius = np.max(np.linalg.norm(crown_centroid - np.concatenate(crown)))

    return crown_centroid, max_radius / 5

# create dictionary that describes a tree (trunk/crown) from a set of points
def create_tree(pts):
    crown = crown_from_points(pts)

    if crown is not None:
        trunk = trunk_from_crown(crown[1])

        return {'crown':crown, 'trunk_top': trunk[0], 'trunk_width': trunk[1]}

# generate trees from point cloud using DBSCAN clustering and convex hulls
def generate_trees(tree_points, process_count):
    # compute clusters using sklearn's DBSCAN algorithm
    clus = DBSCAN(eps=eps, n_jobs=-1).fit(tree_points.astype(np.float32))
    labels = clus.labels_
    clusters_n = len(set(labels)) - (1 if -1 in labels else 0)  # got this from StackOverflow
    clusters = [tree_points[np.where(labels == n)] for n in range(clusters_n)]

    # compute tree crowns and trunks in parallel
    p = Pool(process_count)
    trees = p.map(create_tree, clusters)
    p.close()
    p.join()

    return trees

### OUTPUT FILE FUNCTIONS ###
def write_cityjson(buildings, trees):
    pass

if __name__ == '__main__':
    # configuration parameters
    footprint_file = './tudcampus.geojson'
    point_cloud_file = './ahn3_clipped_thinned.las'
    grid_size = np.array([4000, 4000])
    process_count = 6
    thinning_factor = 1

    # tree generator parameters
    eps = 1000
    top_trunk_ratio = 5

    # DATA INPUT
    # read input data
    print('Reading data')
    footprints = read_geojson(footprint_file)
    points_raw = read_laz(point_cloud_file)
    points = np.column_stack((points_raw.X, points_raw.Y, points_raw.Z, points_raw.classification))[::thinning_factor]

    # SPATIAL INDEX
    # construct list of lists to be used as grid for speeding up intersections
    print('Generating buildings')
    building_points = points[points[:, 3] == 6][:, :3]
    buildings_bbox = np.array([[building_points[:, 0].min(), building_points[:, 0].max()],
                               [building_points[:, 1].min(), building_points[:, 1].max()]])

    x_cells = (buildings_bbox[0][1] - buildings_bbox[0][0]) // grid_size[0]
    y_cells = (buildings_bbox[1][1] - buildings_bbox[1][0]) // grid_size[1]

    grid = [[[] for x in range(y_cells.astype(int) + 1)] for y in range(x_cells.astype(int) + 1)]
    points_shifted = (building_points[:, :2] - buildings_bbox[:, :1].transpose()).astype(int) // grid_size

    for i, p in enumerate(points_shifted):
        grid[p[0]][p[1]].append(i)

    # find points that potentially fall inside each building
    footprint_coords = [(np.concatenate(np.array(i['geometry']['coordinates'])) * 1000) for i in footprints['features']]
    candidate_points = [get_bbox_points(f, buildings_bbox[:, :1], building_points) for f in footprint_coords]

    # intersect point cloud with each building
    p = Pool(processes=process_count)
    points_in_polygons = p.starmap(points_in_poly, zip(footprint_coords, candidate_points))
    p.close()
    p.join()

    # compute median height of each building
    median_heights = [i[len(i) // 2][2] if len(i) else 0 for i in points_in_polygons]
    buildings = zip(footprints['features'], median_heights)

    print('Generating trees')
    tree_points = points[points[:, 3] == 1][:, :3]
    trees = generate_trees(tree_points, process_count)

    print('Writing CityJSON')
    write_cityjson(buildings, trees)

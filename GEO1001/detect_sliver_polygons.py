from shapely import geometry
from shapely.ops import cascaded_union
import matplotlib.pyplot as plt

polygon_1 = [(499.92, 66.53), (516.24, 86.35), (530.55, 142.87), (516.94, 183.09), (500.17, 144.08), (475.0, 160.0), (425.89, 110.11), (400.0, 60.0), (442.8, 69.33), (499.92, 66.53)]
polygon_2 = [(502.0, 63.53), (520.0, 180.0), (470.0, 155.0), (403.0, 58.0), (502.0, 63.53)]


def multipolygon_to_vertices(mp):
    return [list(i.exterior.coords) for i in mp]


def plot_poly_diff(a, b, diff_a, diff_b):
    plt.plot(*a.exterior.xy)
    plt.plot(*b.exterior.xy)

    for p in diff_a:
        plt.fill(*p.exterior.xy, 'b')

    for p in diff_b:
        plt.fill(*p.exterior.xy, 'r')

    plt.show()


def subtract_polygons(a, b):
    diff_a = list(a.difference(b))
    diff_b = list(b.difference(a))

    return diff_a, diff_b


poly_1, poly_2 = geometry.Polygon(polygon_1), geometry.Polygon(polygon_2)
diff = subtract_polygons(poly_1, poly_2)


plot_poly_diff(poly_1, poly_2, diff[0], diff[1])


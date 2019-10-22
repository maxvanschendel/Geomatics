# GEO1000 - Assignment 4
# Authors:
# Studentnumbers:

import math


class Point(object):
    def __init__(self, x, y):
        """Constructor"""
        self.x = x
        self.y = y

    def __str__(self):
        """Well Known Text of this point
        """
        return "POINT({} {})".format(self.x, self.y)

    def __hash__(self):
        """Allows a Point instance to be used 
        (as key) in a dictionary or in a set (i.e. hashed collections)."""
        return hash((self.x, self.y))

    def __eq__(self, other):
        """Compare Point instances for equivalence 
        (this object instance == other instance?).

        :param other: the point to compare with
        :type other: Point

        Returns True/False
        """
        # Your implementation here
        return (self.x, self.y) == (other.x, other.y)

    def distance(self, other):
        """Returns distance as float to the *other* point 
        (assuming Euclidean geometry)

        :param other: the point to compute the distance to
        :type other: Point
        """
        # Your implementation here

        return math.sqrt((self.x - other.x)**2 - (self.y - other.y)**2)


class Circle(object):
    def __init__(self, center, radius):
        """Constructor"""
        self.center = center
        self.radius = float(radius)

    def __str__(self):
        """Returns WKT str, discretizing the circle into straight
        line segments
        """
        N = 400  # the number of segments
        step = 2.0 * math.pi / N
        pts = []
        for i in range(N):
            pts.append(Point(self.center.x + math.cos(i * step) * self.radius,
                             self.center.y + math.sin(i * step) * self.radius))
        pts.append(pts[0])
        coordinates = ["{0} {1}".format(pt.x, pt.y) for pt in pts]
        coordinates = ", ".join(coordinates)
        return "POLYGON(({0}))".format(coordinates)

    def covers(self, pt):
        """Returns True when the circle covers point *pt*, 
        False otherwise

        Note that we consider points that are near to the boundary of the 
        circle also to be covered by the circle(arbitrary epsilon to use: 1e-8).
        """
        # Your implementation here
        epsilon = 10**-8
        return (pt.x - self.center[0])**2 + (pt.y - self.center[1])**2 <= self.radius**2. + epsilon

    def area(self):
        """Returns area as float of this circle
        """
        # Your implementation here
        return math.pi*self.radius**2

    def perimeter(self):
        """Returns perimeter as float of this circle
        """
        # Your implementation here
        return 2*math.pi*self.radius


class Triangle(object):
    def __init__(self, p0, p1, p2):
        """Constructor

        Arguments: p0, p1, p2 -- Point instances
        """
        self.p0, self.p1, self.p2 = p0, p1, p2

    def __str__(self):
        """String representation
        """
        points = ["{0.x} {0.y}".format(pt) for pt in (
            self.p0, self.p1, self.p2, self.p0)]
        return "POLYGON(({0}))".format(", ".join(points))

    def circumcircle(self):
        """Returns Circle instance that intersects the 3 points of the triangle.
        """
        # Your implementation here
        pass

    def area(self):
        """Area of this triangle, using Heron's formula."""
        # Your implementation here
        pass

    def perimeter(self):
        """Perimeter of this triangle (float)"""
        # Your implementation here
        pass


class DelaunayTriangulation(object):
    def __init__(self, points):
        """Constructor"""
        self.triangles = []
        self.points = points

    def triangulate(self):
        """Triangulates the given set of points.

        This method takes the set of points to be triangulated 
        (with at least 3 points) and for each 3-group of points instantiates 
        a triangle and checks whether the triangle conforms to Delaunay 
        criterion. If so, the triangle is added to the triangle list.

        To determine the 3-group of points, the group3 function is used.

        Returns None
        """
        # pre-condition: we should have at least 3 points
        assert len(self.points) > 2
        # Your implementation here
        pass

    def is_delaunay(self, tri):
        """Does a triangle *tri* conform to the Delaunay criterion?

        Algorithm:

        Are 3 points of the triangle collinear?
            No:
                Get circumcircle
                Count number of points inside circumcircle
                if number of points inside == 3:
                    Delaunay
                else:
                    not Delaunay
            Yes:
                not Delaunay

        Arguments:
            tri -- Triangle instance
        Returns:
            True/False
        """
        # Your implementation here
        pass

    def are_collinear(self, pa, pb, pc):
        """Orientation test to determine whether 3 points are collinear
        (on straight line).

        Note that we consider points that are nearly collinear also to be on 
        a straight line (arbitrary epsilon to use: 1e-8).

        Returns True / False
        """
        # Your implementation here
        pass

    def output_points(self, open_file_obj):
        """Outputs the points of the triangulation to an open file.
        """
        # Your implementation here
        pass

    def output_triangles(self, open_file_obj):
        """Outputs the triangles of the triangulation to an open file.
        """
        # Your implementation here
        pass

    def output_circumcircles(self, open_file_obj):
        """Outputs the circumcircles of the triangles of the triangulation
        to an open file
        """
        # Your implementation here
        pass


def group3(N):
    """Returns generator with 3-tuples with indices to form 3-groups
    of a list of length N.

    Total number of tuples that is generated: N! / (3! * (N-3)!)

    For N = 3: [(0, 1, 2)]
    For N = 4: [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
    For N = 5: [(0, 1, 2), (0, 1, 3), (0, 1, 4), (0, 2, 3), 
                (0, 2, 4), (0, 3, 4), (1, 2, 3), (1, 2, 4), 
                (1, 3, 4), (2, 3, 4)]

    Example use:

        >>> for item in group3(3):
        ...     print(item)
        ... 
        (0, 1, 2)

    """
    # See for more information about generators for example:
    # https://jeffknupp.com/blog/2013/04/07/improve-your-python-yield-and-generators-explained/
    for i in range(N - 2):
        for j in range(i+1, N - 1):
            for k in range(j+1, N):
                yield (i, j, k)


def make_random_points(n):
    """Makes n points distributed randomly in x,y between [0,1000]

    Note, no duplicate points will be created, but might result in slightly 
    less than the n number of points requested.
    """
    import random
    pts = list(set([Point(random.randint(0, 1000),
                          random.randint(0, 1000)) for i in range(n)]))
    return pts


def main(n):
    """Perform triangulation of n points and write the resulting geometries
    to text files.
    """
    pts = make_random_points(n)
    dt = DelaunayTriangulation(pts)
    dt.triangulate()
    # using the with statement, we do not need to close explicitly the file
    with open("points.wkt", "w") as fh:
        dt.output_points(fh)
    with open("triangles.wkt", "w") as fh:
        dt.output_triangles(fh)
    with open("circumcircles.wkt", "w") as fh:
        dt.output_circumcircles(fh)


def _test():
    # If you want, you can write tests in this function
    rand_points = make_random_points(100)
    assert(len(rand_points) == 100)


if __name__ == "__main__":
    main(5)

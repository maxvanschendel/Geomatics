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

        return (self.x, self.y) == (other.x, other.y)

    def distance(self, other):
        """Returns distance as float to the *other* point 
        (assuming Euclidean geometry)

        :param other: the point to compute the distance to
        :type other: Point
        """

        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)


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

        epsilon = 10**-8
        return (pt.x - self.center.x)**2 + (pt.y - self.center.y)**2 <= self.radius**2. + epsilon

    def area(self):
        """Returns area as float of this circle
        """

        return math.pi*self.radius**2

    def perimeter(self):
        """Returns perimeter as float of this circle
        """

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

        a = self.p0
        b = self.p1
        c = self.p2

        d = 2*(a.x*(b.y - c.y) + b.x*(c.y-a.y) + c.x*(a.y-b.y))
        ux = ((a.x**2 + a.y**2)*(b.y-c.y) + (b.x**2 + b.y**2)*(c.y-a.y) + (c.x**2 + c.y**2)*(a.y-b.y))/d
        uy = ((a.x**2 + a.y**2)*(c.x-b.x) + (b.x**2 + b.y**2)*(a.x-c.x) + (c.x**2 + c.y**2)*(b.x-a.x))/d

        center = Point(ux, uy)
        radius = a.distance(center)

        return Circle(center, radius)

    def area(self):
        """Area of this triangle, using Heron's formula."""

        a = self.p0.distance(self.p1)
        b = self.p1.distance(self.p2)
        c = self.p2.distance(self.p0)

        s = (a+b+c)/2

        return math.sqrt(s*(s-a)*(s-b)*(s-c))

    def perimeter(self):
        """Perimeter of this triangle (float)"""

        return self.p0.distance(self.p1) + self.p1.distance(self.p2) + self.p2.distance(self.p0)


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

        for i in group3(len(self.points)):
            tri = Triangle(self.points[i[0]], self.points[i[1]], self.points[i[2]])
            if self.is_delaunay(tri):
                self.triangles.append(tri)

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

        if self.are_collinear(tri.p0, tri.p1, tri.p2):
            return False
        else:
            cc = tri.circumcircle()
            if sum([cc.covers(i) for i in self.points]) == 3:
                return True
            else:
                return False

    def are_collinear(self, pa, pb, pc):
        """Orientation test to determine whether 3 points are collinear
        (on straight line).

        Note that we consider points that are nearly collinear also to be on 
        a straight line (arbitrary epsilon to use: 1e-8).

        Returns True / False
        """

        return (pa.x - pc.x)*(pb.y - pc.y) - (pb.x - pc.x)*(pa.y - pc.y) < 10**-8

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
    main(5000)
    print('a')


if __name__ == "__main__":
    main(5)
    print('a')

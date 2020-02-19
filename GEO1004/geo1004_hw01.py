# -- GEO1004.2020--hw01
# -- Max van Schendel
# -- 4384644


import numpy as np
import time
import itertools


class Vertex:
    def __init__(self, coords):
        self.x, self.y, self.z = coords[0], coords[1], coords[2]
        self.pos = np.asarray(coords)


class Face:
    def __init__(self, vxs):
        self.a, self.b, self.c = vxs[0], vxs[1], vxs[2]
        self.vxs = set(vxs)
        self.nbs = set()

    def flip(self):
        self.a, self.c = self.c, self.a

    def find_neighbours(self, fs):
        for f in fs:
            if len(self.vxs.intersection(f.vxs)) == 2 and len(self.nbs) <= 3:
                self.nbs.add(f)

    def normal(self):
        return np.cross(self.a.pos - self.b.pos, self.a.pos - self.c.pos)

    def centroid(self):
        return np.asarray(((self.a.x + self.b.x + self.c.x) / 3,
                          (self.a.y + self.b.y + self.c.y) / 3,
                          (self.a.z + self.b.z + self.c.z) / 3))

    def get_vertices(self):
        return [v.pos for v in self.vxs]

    def bbox(self):
        return compute_bbox(np.asarray(self.get_vertices()))

    def concavity(self, other_face):
        vertex_a = self.vxs.difference(other_face.vxs).pop()
        vertex_b = other_face.vxs.difference(self.vxs).pop()

        return np.dot((vertex_b.pos - vertex_a.pos), self.normal())


class Mesh:
    def __init__(self):
        self.faces = set()

    def bbox(self):
        return compute_bbox(np.concatenate([f.get_vertices() for f in self.faces]))

    def conform_normals(self):
        init_face = self.faces.pop()
        centroid = init_face.centroid()

        ray_origin = random_points_on_sphere(r=250, n=1)[0] + centroid
        ray = Ray((centroid - ray_origin), ray_origin)
        intersections = []

        for g in self.faces:
            inter = ray_intersect(ray.direction, ray.origin, g, epsilon=0.0000001)

            if inter is not None:
                dist = np.linalg.norm(inter - ray.origin)
                intersections.append((dist, g))

        # check if normal is correct, otherwise flips the face
        for i in enumerate(sorted(intersections, key=lambda x: x[0])):
            if i[0] % 2 == 0:
                if np.dot(ray.direction, i[1][1].normal()) > 0:
                    i[1][1].flip()
            else:
                if np.dot(ray.direction, i[1][1].normal()) < 0:
                    i[1][1].flip()

        processed = set()
        search_queue = {intersections[0][1]}

        while search_queue:
            face = search_queue.pop()
            for nb in face.nbs:
                if nb not in processed and nb not in search_queue:
                    search_queue.add(nb), processed.add(nb)

                    if face.concavity(nb) <= 0:
                        if np.dot(face.normal(), nb.normal()) <= 0:
                            nb.flip()
                    else:
                        if np.dot(face.normal(), nb.normal()) >= 0:
                            nb.flip()


class Ray:
    def __init__(self, direction, origin):
        self.direction = direction
        self.origin = origin


def compute_bbox(vxs):
    x_min, x_max = np.min(vxs[:, 0]), np.max(vxs[:, 0])
    y_min, y_max = np.min(vxs[:, 1]), np.max(vxs[:, 1])
    z_min, z_max = np.min(vxs[:, 2]), np.max(vxs[:, 2])

    return np.array((x_min, y_min, z_min)), np.array((x_max, y_max, z_max))


def parse_obj(fn):
    with open(fn) as obj_file:
        lines = [i.replace('\n', '').split(' ') for i in obj_file.readlines()]

    vxs, fs = [], []

    for i in lines:
        if i[0] == 'v':
            vxs.append([float(i) for i in i[1:]])
        elif i[0] == 'f':
            fs.append([int(i) - 1 for i in i[1:]])

    return vxs, fs


def write_obj(meshes, fn):
    pass


# use poly-based BFS to grow polygon soup into connected meshes
def group_triangles(unassigned_faces):
    meshes = set()
    while unassigned_faces:
        m = Mesh()
        search_queue = {unassigned_faces[0]}

        while search_queue:
            face = search_queue.pop()
            face.find_neighbours(unassigned_faces)

            for n in face.nbs:
                if n not in m.faces:
                    m.faces.add(n)
                    search_queue.add(n)
                    unassigned_faces.remove(n)

        meshes.add(m)

    return meshes


def random_points_on_sphere(r, n):
    p = np.random.normal(size=(n, 3))
    p /= np.linalg.norm(p, axis=1)[:, np.newaxis]

    return p * r


def ray_intersect(ray, ray_origin, tri, epsilon):
    edge1 = tri.b.pos - tri.a.pos
    edge2 = tri.c.pos - tri.a.pos

    h = np.cross(ray, edge2)
    a = np.dot(edge1, h)

    if -epsilon < a < epsilon:
        return None

    f = 1 / a
    s = ray_origin - tri.a.pos
    u = f * np.dot(s, h)

    if u < 0 or u > 1.0:
        return None

    q = np.cross(s, edge1)
    v = f * np.dot(ray, q)

    if not 0 < v < 1:
        return None

    t = f * np.dot(edge2, q)

    if t > epsilon:
        return ray_origin + ray * t
    else:
        return None


if __name__ == '__main__':

    # read geometry data from .obj file
    start_time = time.perf_counter()
    raw_geometry = parse_obj('./obj/bk_soup.obj')
    print(f'Read .obj file in {time.perf_counter() - start_time:.3f}s')

    # create objects from raw geometry
    start_time = time.perf_counter()
    vertices = [Vertex(i) for i in raw_geometry[0]]
    faces = [Face((vertices[i[0]], vertices[i[1]], vertices[i[2]])) for i in raw_geometry[1]]
    print(f'Created objects in {time.perf_counter() - start_time:.3f}s')

    # join meshes by growing seed faces using BFS
    start_time = time.perf_counter()
    joined_meshes = group_triangles(faces)
    print(f'Joined meshes in {time.perf_counter() - start_time:.3f}s')

    # fix meshes normals
    for m in joined_meshes:
        m.conform_normals()

    write_obj(joined_meshes, './output.obj')
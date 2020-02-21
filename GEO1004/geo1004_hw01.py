# -- GEO1004.2020--hw01
# -- Max van Schendel
# -- 4384644


import numpy as np
from sys import argv
from copy import deepcopy
import time


class Ray:
    def __init__(self, direction, origin):
        self.direction = direction
        self.origin = origin

    # Möller–Trumbore ray intersection
    def intersect(self, face, tolerance):
        h = np.cross(self.direction, face.edge2)
        a = np.dot(face.edge1, h)

        if -tolerance < a < tolerance:
            return None

        f = 1 / a
        s = self.origin - face.a.pos
        u = f * np.dot(s, h)

        if u < 0 or u > 1.0:
            return None

        q = np.cross(s, face.edge1)
        v = f * np.dot(self.direction, q)

        if v < 0 or u + v > 1.0:
            return None

        t = f * np.dot(face.edge2, q)

        if t > tolerance:
            return self.origin + self.direction * t
        else:
            return None

    # casts a ray through the mesh and returns all intersecting triangles
    def cast(self, mesh):
        intersections = []
        for g in mesh.faces:
            inter = self.intersect(g, 0.0001)
            if inter is not None:
                intersections.append((np.linalg.norm(inter - self.origin), g, inter))

        return sorted(intersections, key=lambda x: x[0])


class Vertex:
    def __init__(self, coords):
        self.pos = np.asarray(coords)


class Face:
    def __init__(self, vxs):
        self.a, self.b, self.c = vxs[0], vxs[1], vxs[2]
        self.edge1, self.edge2 = self.b.pos - self.a.pos, self.c.pos - self.a.pos
        self.vxs = set(vxs)
        self.nbs = set()

    # gets the next vertex in the winding order
    def next_vertex(self, vx):
        if vx == self.a:
            return self.b
        elif vx == self.b:
            return self.c
        elif vx == self.c:
            return self.a

    # flip face by flipping vertex order
    def flip(self):
        self.a, self.c = self.c, self.a

    # find at max 3 neighbours from set of faces
    def find_neighbours(self, fs):
        for f in fs:
            if len(f.nbs) < 3:
                if len(self.vxs.intersection(f.vxs)) == 2:
                    self.nbs.add(f)
                    f.nbs.add(self)
                if len(self.nbs) == 3:
                    break

    # face normal vector, vertices are in ccw order a>b>c
    def normal(self):
        return np.cross(self.a.pos - self.b.pos, self.a.pos - self.c.pos)

    # average of vertices
    def centroid(self):
        return (self.a.pos + self.b.pos + self.c.pos)/3


class Mesh:
    def __init__(self):
        self.faces = set()

    # compute diagonal of bounding box
    def bbox(self):
        # all vertices in mesh
        vxs = np.concatenate([[v.pos for v in f.vxs] for f in self.faces])

        return np.array((np.min(vxs[:, 0]), np.min(vxs[:, 1]), np.min(vxs[:, 2]))), \
            np.array((np.max(vxs[:, 0]), np.max(vxs[:, 1]), np.max(vxs[:, 2])))

    # create random ray from large sphere around mesh in direction of a face's centroid
    def random_external_ray(self, focus):
        # get bbox diagonal size
        bbox = self.bbox()
        radius = np.linalg.norm(bbox[0] - bbox[1])

        # generate random point on unit sphere
        p = np.random.normal(size=(1, 3))
        p /= np.linalg.norm(p, axis=1)[:, np.newaxis]

        # move sphere origin to centroid and calculate ray vector
        ray_origin = (p * radius)[0] + focus
        ray_direction = (focus - ray_origin)

        return Ray(ray_direction, ray_origin)

    # fixes mesh so that every face's normal points outwards
    def conform_normals(self):
        # cast random ray from environment at centroid of random face
        init_face = next(iter(self.faces))
        ray = self.random_external_ray(init_face.centroid())
        intersections = ray.cast(self)

        # check if normal orientation is correct relative to ray, otherwise flip face
        for i in enumerate(intersections):
            if i[0] % 2 == 0:
                if np.dot(ray.direction, i[1][1].normal()) >= 0:
                    i[1][1].flip()
            else:
                if np.dot(ray.direction, i[1][1].normal()) <= 0:
                    i[1][1].flip()

        # walk mesh, enforcing winding order for every face/neighbour pair
        self.walk(intersections[0][1], self.enforce_winding)

    # walk mesh faces in BFS order, applies function to every face/neighbour pair
    @staticmethod
    def walk(init, func):
        processed = set()
        search_queue = {init}

        # exhaustively traverse mesh
        while search_queue:
            face = search_queue.pop()
            for nb in face.nbs:
                if nb not in processed and nb not in search_queue:
                    search_queue.add(nb)
                    func(face, nb)

            processed.add(face)

    # checks if a face's neighbour's winding order is correct and modifies it if it isn't
    @staticmethod
    def enforce_winding(face, nb):
        for v in face.vxs.intersection(nb.vxs):
            if face.next_vertex(v) == nb.next_vertex(v):
                nb.flip()


class PolygonSoup:
    def __init__(self, faces):
        self.faces = faces

    # grow polygon soup into connected meshes
    def merge(self):
        unassigned_faces = deepcopy(self.faces)
        meshes = set()

        while unassigned_faces:
            m = Mesh()
            init_face = unassigned_faces[0]
            search_queue = {init_face}

            while search_queue:
                face = search_queue.pop()
                face.find_neighbours(unassigned_faces)

                for n in face.nbs:
                    if n not in m.faces:
                        m.faces.add(n)
                        search_queue.add(n)
                        unassigned_faces.remove(n)

            m.faces.add(init_face)
            meshes.add(m)

        return meshes


class ObjParser:
    @staticmethod
    def read(fn):
        lines = [i.replace('\n', '').split(' ') for i in open(fn).readlines()]
        vxs, fs = [], []

        for i in lines:
            if i[0] == 'v':
                vxs.append([float(i) for i in i[1:]])
            elif i[0] == 'f':
                fs.append([int(i) - 1 for i in i[1:]])

        return vxs, fs

    @staticmethod
    def write(output_meshes, out):
        all_verts = []

        for m in output_meshes:
            faces = list(m.faces)
            vertices = list(set().union(*[i.vxs for i in faces]))

            for i in vertices:
                all_verts.append(' '.join([format(x, '.2f') for x in i.pos]))

        with open(out, 'w') as obj_file:
            for v in all_verts:
                obj_file.write(f'v {v}\n')

            for m in enumerate(output_meshes):
                faces = list(m[1].faces)
                obj_file.write(f'o {m[0] + 1}\n')

                for f in faces:
                    a, b, c = ' '.join([format(x, '.2f') for x in f.a.pos]), \
                              ' '.join([format(x, '.2f') for x in f.b.pos]), \
                              ' '.join([format(x, '.2f') for x in f.c.pos])

                    a_index, b_index, c_index = all_verts.index(a), all_verts.index(b), all_verts.index(c)
                    obj_file.write(f'f {a_index + 1} {b_index + 1} {c_index + 1}\n')


if __name__ == '__main__':
    if len(argv) > 1:
        input_file, output_file = argv[1], argv[2]
    else:
        input_file, output_file = 'bk_soup.obj', 'output.obj'

    # read geometry data from .obj file
    raw_geometry = ObjParser.read(input_file)

    # create objects from raw geometry
    vertices = [Vertex(i) for i in raw_geometry[0]]


    polygon_soup = PolygonSoup([Face((vertices[i[0]], vertices[i[1]], vertices[i[2]])) for i in raw_geometry[1]])


    # join meshes by growing seed faces using BFS
    start_time = time.clock()
    meshes = polygon_soup.merge()
    print(start_time - time.clock())
    # fix meshes normals
    for m in meshes:
        m.conform_normals()

    # write obj file to disk
    ObjParser.write(meshes, output_file)

# -- GEO1004.2020--hw01
# -- Max van Schendel
# -- 4384644


import numpy as np
import time
import sys


class Ray:
    def __init__(self, direction, origin):
        self.direction = direction
        self.origin = origin


class Vertex:
    def __init__(self, coords):
        self.x, self.y, self.z = coords[0], coords[1], coords[2]
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
            if len(self.vxs.intersection(f.vxs)) == 2:
                self.nbs.add(f)
            if len(self.nbs) == 3:
                break

    # face normal vector, vertices are in ccw order a>b>c
    def normal(self):
        return np.cross(self.a.pos - self.b.pos, self.a.pos - self.c.pos)

    # average of vertices
    def centroid(self):
        return np.asarray(((self.a.x+self.b.x+self.c.x) / 3,
                           (self.a.y+self.b.y+self.c.y) / 3,
                           (self.a.z+self.b.z+self.c.z) / 3))

    # Möller–Trumbore intersection algorithm
    # Python port of C++ implementation on Wikipedia
    def ray_intersect(self, ray, tolerance):
        h = np.cross(ray.direction, self.edge2)
        a = np.dot(self.edge1, h)

        if -tolerance < a < tolerance:
            return None

        f = 1 / a
        s = ray.origin - self.a.pos
        u = f * np.dot(s, h)

        if u < 0 or u > 1.0:
            return None

        q = np.cross(s, self.edge1)
        v = f * np.dot(ray.direction, q)

        if v < 0 or u + v > 1.0:
            return None

        t = f * np.dot(self.edge2, q)

        if t > tolerance:
            return ray.origin + ray.direction * t
        else:
            return None


class Mesh:
    def __init__(self):
        self.faces = set()

    # compute diagonal of bounding box
    def bbox(self):
        # all vertices in mesh
        vxs = np.concatenate([[v.pos for v in f.vxs] for f in self.faces])
        x_min, x_max = np.min(vxs[:, 0]), np.max(vxs[:, 0])
        y_min, y_max = np.min(vxs[:, 1]), np.max(vxs[:, 1])
        z_min, z_max = np.min(vxs[:, 2]), np.max(vxs[:, 2])

        return np.array((x_min, y_min, z_min)), np.array((x_max, y_max, z_max))

    # create random ray from large sphere around mesh in direction of a face's centroid
    def random_external_ray(self, focus, scale):
        # get bbox diagonal size
        bbox = self.bbox()
        radius = np.linalg.norm(bbox[0] - bbox[1])

        # generate random point on unit sphere
        p = np.random.normal(size=(1, 3))
        p /= np.linalg.norm(p, axis=1)[:, np.newaxis]

        # move sphere origin to centroid and calculate ray vector
        ray_origin = (p * radius * scale)[0] + focus
        ray_direction = (focus - ray_origin)

        return Ray(ray_direction, ray_origin)

    # casts a ray through the mesh and returns all intersecting triangles
    def ray_cast(self, ray):
        intersections = []
        for g in self.faces:
            inter = g.ray_intersect(ray, 0.0001)

            if inter is not None:
                dist = np.linalg.norm(inter - ray.origin)
                intersections.append((dist, g, inter))

        return sorted(intersections, key=lambda x: x[0])

    # fixes mesh so that every face's normal points outwards
    def conform_normals(self, scale):
        # cast random ray from environment at centroid of random face
        init_face = next(iter(self.faces))
        ray = self.random_external_ray(init_face.centroid(), scale)
        intersections = self.ray_cast(ray)

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


class ObjParser:
    def __init__(self, fn):
        self.fn = fn

    # read .obj file
    def read(self):
        with open(self.fn) as obj_file:
            lines = [i.replace('\n', '').split(' ') for i in obj_file.readlines()]

        vxs, fs = [], []

        for i in lines:
            if i[0] == 'v':
                vxs.append([float(i) for i in i[1:]])
            elif i[0] == 'f':
                fs.append([int(i) - 1 for i in i[1:]])

        return vxs, fs

    # write .obj file
    def write(self, meshes, out):
        all_verts = []

        for m in meshes:
            faces = list(m.faces)
            vertices = list(set().union(*[i.vxs for i in faces]))

            for i in vertices:
                all_verts.append(' '.join([format(x, '.2f') for x in i.pos]))

        with open(out, 'w') as obj_file:
            for v in all_verts:
                obj_file.write(f'v {v}\n')

            for m in enumerate(meshes):
                faces = list(m[1].faces)
                obj_file.write(f'o {m[0]+1}\n')

                for f in faces:
                    a, b, c = ' '.join([format(x, '.2f') for x in f.a.pos]), \
                              ' '.join([format(x, '.2f') for x in f.b.pos]),\
                              ' '.join([format(x, '.2f') for x in f.c.pos])

                    a_index, b_index, c_index = all_verts.index(a), all_verts.index(b), all_verts.index(c)
                    obj_file.write(f'f {a_index + 1} {b_index + 1} {c_index + 1}\n')


# grow polygon soup into connected meshes
def merge_soup(unassigned_faces):
    meshes = set()
    fs = list(unassigned_faces)

    while unassigned_faces:
        m = Mesh()
        init_face = unassigned_faces[0]
        search_queue = {init_face}

        while search_queue:
            face = search_queue.pop()
            face.find_neighbours(fs)

            for n in face.nbs:
                if n not in m.faces:
                    m.faces.add(n)
                    search_queue.add(n)
                    unassigned_faces.remove(n)

        m.faces.add(init_face)
        meshes.add(m)

    return meshes


if __name__ == '__main__':
    if len(sys.argv) > 1:
        input_file, output_file = sys.argv[1], sys.argv[2]
    else:
        input_file, output_file = 'bk_soup.obj', 'output.obj'

    # read geometry data from .obj file
    start_time = time.perf_counter()

    parser = ObjParser(input_file)
    raw_geometry = parser.read()
    print(f'Read .obj file in {time.perf_counter() - start_time:.3f}s')

    # create objects from raw geometry
    start_time = time.perf_counter()
    vertices = [Vertex(i) for i in raw_geometry[0]]
    faces = [Face((vertices[i[0]], vertices[i[1]], vertices[i[2]])) for i in raw_geometry[1]]
    print(f'Created objects in {time.perf_counter() - start_time:.3f}s')

    # join meshes by growing seed faces using BFS
    start_time = time.perf_counter()
    meshes = merge_soup(faces)
    print(f'Joined meshes in {time.perf_counter() - start_time:.3f}s')

    # fix meshes normals
    start_time = time.perf_counter()
    for m in meshes:
        m.conform_normals(scale=2.5)
    print(f'Conformed normals in {time.perf_counter() - start_time:.3f}s')

    # write obj file to disk
    start_time = time.perf_counter()
    parser.write(meshes, output_file)
    print(f'Written .obj file to disk in {time.perf_counter() - start_time:.3f}s')

# -- GEO1004.2020--hw01
# -- Max van Schendel
# -- 4384644

import numpy as np

# CONSTRUCT DATA STRUCTURE BY GROWING FACE


class Vertex:
    def __init__(self, coords):
        self.x = coords[0]
        self.y = coords[1]
        self.z = coords[2]


class Face:
    def __init__(self, vxs):
        self.a, self.b, self.c = vxs[0], vxs[1], vxs[2]
        self.vxs = set(vxs)
        self.nb = []
        self.normal = []

    def find_neighbours(self, fs):
        return filter(lambda x: len(self.vxs.intersection(x.vxs)) == 2, fs)


class Mesh:
    def __init__(self):
        self.faces = set()

    def get_faces(self):
        return self.faces

    def count_faces(self):
        return len(self.faces)


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


def group_triangles(fs):
    meshes = set()

    while fs:
        m = Mesh()
        search_queue = set()
        search_queue.add(fs[0])

        while search_queue:
            face = search_queue.pop()
            face.nbs = face.find_neighbours(fs)

            for n in face.nbs:
                if n not in search_queue and n not in m.faces:
                    m.faces.add(n)
                    search_queue.add(n)
                    fs.remove(n)

        meshes.add(m)

    return meshes


def align_normals():
    pass


def write_obj():
    pass


if __name__ == '__main__':
    raw_geometry = parse_obj('./obj/bk_soup.obj')

    # reconstruct geometry from polygon soup
    vertices = [Vertex(i) for i in raw_geometry[0]]
    faces = [Face((vertices[i[0]], vertices[i[1]], vertices[i[2]])) for i in raw_geometry[1]]
    meshes = group_triangles(faces)

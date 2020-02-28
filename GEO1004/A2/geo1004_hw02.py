# -- GEO1004.2020--hw01
# -- Max van Schendel
# -- 4384644


import numpy as np
from multiprocessing import Pool
from sys import argv


# Ray object with direction and origin, can calculate intersection with face
class Ray:
    def __init__(self, direction, origin):
        self.direction = direction
        self.origin = origin

    # Möller–Trumbore ray intersection:
    def ray_triangle_intersect(self, face, tolerance):
        h = np.cross(self.direction, face.edge2)
        a = np.dot(face.edge1, h)

        if -tolerance < a < tolerance:
            return None

        f, s = 1 / a, self.origin - face.a.pos
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

    def ray_box_intersect(self, bbox):
        t = (bbox[0] - self.origin)/self.direction
        k = (bbox[1] - self.origin)/self.direction

        tmin = np.max([np.max([np.min([t[0], k[0]]), np.min([t[1], k[1]])]), np.min([t[2], k[2]])])
        tmax = np.min([np.min([np.max([t[0], k[0]]), np.max([t[1], k[1]])]), np.max([t[2], k[2]])])

        if tmin > tmax:
            return False
        else:
            return tmin

    # casts a ray through the scene and returns all intersecting triangles
    def cast(self, scene):
        intersections = []
        for mesh in scene:
            if self.ray_box_intersect(mesh.bbox):
                for g in mesh.faces:
                    if self.ray_box_intersect(g.bbox):
                        inter = self.ray_triangle_intersect(g, 0.0001)
                        if inter is not None:
                            intersections.append((np.linalg.norm(inter - self.origin), g, inter))

        return sorted(intersections, key=lambda x: x[0])


# just a point in 3D space
class Vertex:
    def __init__(self, coords):
        self.pos = np.asarray(coords)


# triangle consisting of 3 vertices, contains pointers to neighbouring faces
class Face:
    def __init__(self, vxs, parent=None):
        self.parent = parent
        self.a, self.b, self.c = vxs[0], vxs[1], vxs[2]

        self.vxs = np.array([v.pos for v in vxs])   # triangle as 3x3 matrix
        self.bbox = np.array([np.min(self.vxs, axis=0), np.max(self.vxs, axis=0)])  # min/max of each column


class Mesh:
    def __init__(self, mesh_id):
        self.id = mesh_id
        self.faces = set()

    def bbox(self):
        face_bboxes = np.concatenate([f.bbox for f in self.faces], axis=0)  # all vertices in mesh

        return np.array([np.min(face_bboxes[::2], axis=0), np.max(face_bboxes[1::2], axis=0)])


class Point:
    def __init__(self, pos, meta):
        self.pos = pos
        self.meta = meta


class PointCloud:
    def __init__(self, bbox, cell_size):
        self.bbox = bbox
        self.shape = (((self.bbox[1] - self.bbox[0]) // cell_size) + 1).astype(np.int8)
        self.points = {}


# reads and writes obj files
class ObjParser:
    @staticmethod
    def read(fn):
        vxs, meshes = [], []
        mesh = Mesh(None)

        for i in [i.replace('\n', '').split(' ') for i in open(fn).readlines()]:
            if i[0] == 'v':
                vxs.append(Vertex(np.asarray([float(i) for i in i[1:]])))

            elif i[0] == 'f':
                vx_ind = [int(i) - 1 for i in i[1:]]
                mesh.faces.add(Face(vxs=[vxs[vx_ind[0]], vxs[vx_ind[1]], vxs[vx_ind[2]]],
                                    parent=mesh))
            elif i[0] == 'o':
                mesh = Mesh(i[1])
                meshes.append(mesh)

            elif i[0] == 'usemtl':
                mesh.mat = i[1]

        # if file doesnt use mesh id
        if not len(meshes):
            meshes = [mesh]

        return meshes

    @staticmethod
    def write(scene, out):
        all_verts = []
        output_meshes = scene.meshes

        for m in output_meshes:
            faces = list(m.faces)
            vertices = list(set().union(*[(i.a, i.b, i.c) for i in faces]))

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


class Scene:
    def __init__(self):
        self.meshes = set()
        self.point_clouds = set()

    def mesh_bbox_union(self):
        bboxes = np.concatenate([m.bbox() for m in self.meshes])
        return np.array([np.min(bboxes[::2], axis=0), np.max(bboxes[1::2], axis=0)])


if __name__ == '__main__':
    if len(argv) > 1:
        input_file, output_file = argv[1], argv[2]
    else:
        input_file, output_file = '../obj/bag_bk.obj', 'output.obj'

    # read geometry data from .obj file and create necessary geometry objects
    scene = Scene()
    scene.meshes.add(ObjParser.read(input_file))
    scene.point_clouds.add(PointCloud(bbox=scene.mesh_bbox_union(), cell_size=10))

    # works but is very slow
    ObjParser.write(scene, 'out.obj')


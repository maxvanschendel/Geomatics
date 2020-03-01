# -- GEO1004.2020--hw01
# -- Max van Schendel
# -- 4384644


import numpy as np
from multiprocessing import Pool
from sys import argv
from timeit import default_timer as timer
import itertools


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
            return self.origin + (self.direction * t)

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
    def cast(self, scene, tolerance):
        intersections = []

        for mesh in filter(lambda x: self.ray_box_intersect(x.bbox), scene.meshes):
            for face in mesh.faces:
                inter = self.ray_triangle_intersect(face, tolerance)
                if inter is not None:
                    intersections.append((np.linalg.norm(inter - self.origin), face, inter))

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
        self.edge1, self.edge2 = self.b.pos - self.a.pos, self.c.pos - self.a.pos

        self.vxs = np.array([v.pos for v in vxs])   # triangle as 3x3 matrix
        self.bbox = np.array([np.min(self.vxs, axis=0), np.max(self.vxs, axis=0)])  # min/max of each column


class Mesh:
    def __init__(self, mesh_id):
        self.id = mesh_id
        self.faces = set()
        self.mat = None
        self.bbox = None

    def get_bbox(self):
        face_bboxes = np.concatenate([f.bbox for f in self.faces], axis=0)  # all vertices in mesh
        self.bbox = np.array([np.min(face_bboxes[::2], axis=0), np.max(face_bboxes[1::2], axis=0)]) # min/max of all vertices


class Point:
    def __init__(self, pos, meta):
        self.pos = pos
        self.meta = meta


class PointCloud:
    def __init__(self):
        self.points = {}

    def write_xyz(self, fn):
        with open(fn, 'w+') as file:
            for i in self.points:
                file.write(f'{i[0]} {i[1]} {i[2]}\n')


# reads and writes obj files
class ObjParser:
    @staticmethod
    def read(fn):
        vxs, meshes = [], set()
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
                meshes.add(mesh)

            elif i[0] == 'usemtl':
                mesh.mat = i[1]

        # if file doesnt use mesh id
        if not len(meshes):
            meshes = [mesh]

        [m.get_bbox() for m in meshes]

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
        self.meshes = None

    def mesh_bbox_union(self):
        bboxes = np.concatenate([m.bbox for m in self.meshes])
        return np.array([np.min(bboxes[::2], axis=0), np.max(bboxes[1::2], axis=0)])

    def voxelize(self, cell_size, ray_tolerance=0.00001):

        bbox = self.mesh_bbox_union()
        point_cloud = PointCloud()

        # get voxel grid shape and define direction of rays
        shape = (((bbox[1] - bbox[0]) // cell_size) + 1).astype(np.int8)

        # cast rays from bbox face with smallest area
        # min_face = np.argmin([abs(shape[0]*shape[1]), abs(shape[1]*shape[2]), abs(shape[0]*shape[2])])
        # direction = [0, 0, 0]
        # direction[min_face-1] = 1

        direction = np.array([0, 0, 1])
        ray_transform = np.array([cell_size / 2, cell_size / 2, 0.])

        # cast a ray from every cell in the bounding box's smallest face
        for i in np.arange(0, shape[0], np.sign(shape[0])):
            for j in np.arange(0, shape[1], np.sign(shape[1])):

                # convert voxel grid coordinates to world coordinates
                ray_origin = bbox[0] + \
                    np.array([i*cell_size*np.sign(shape[0]),
                              j*cell_size*np.sign(shape[1]),
                              -cell_size]) \
                    + ray_transform

                # cast ray and find intersections in order of distance
                ray = Ray(direction, ray_origin)
                intersections = ray.cast(self, ray_tolerance)

                # fill lines between every two subsequent intersections
                for z in range(0, len(intersections), 2):
                    dist = intersections[z+1][0] - intersections[z][0]

                    for d in np.arange(0, dist, cell_size):
                        point_cloud.points[(ray_origin[0],
                                            ray_origin[1],
                                            ray_origin[2] + intersections[z][0] + d)] \
                            = intersections[z][1].parent.mat

        return point_cloud


if __name__ == '__main__':
    if len(argv) > 1:
        input_file, output_file = argv[1], argv[2]
    else:
        input_file, output_file = '../obj/isolated_cubes.obj', 'output.obj'

    # read geometry data from .obj file and create necessary geometry objects
    scene = Scene()
    scene.meshes = ObjParser.read(input_file)

    start = timer()
    voxelized_scene = scene.voxelize(cell_size=0.2, ray_tolerance=0.001)
    end = timer()

    print(end-start)
    voxelized_scene.write_xyz('pc.xyz')

    # # works but is very slow
    # ObjParser.write(scene, 'out.obj')



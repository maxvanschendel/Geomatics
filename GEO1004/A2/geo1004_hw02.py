# -- GEO1004.2020--hw01
# -- Max van Schendel
# -- 4384644


import numpy as np
from multiprocessing import Pool
from sys import argv
from timeit import default_timer as timer
from copy import deepcopy
from itertools import product


# Ray object with direction and origin, can calculate intersection with face
class Ray:
    def __init__(self, direction, origin):
        self.direction = direction
        self.origin = origin

    # Möller–Trumbore ray intersection:
    def ray_triangle_intersect(self, face, tolerance):
        edge1, edge2 = face[1] - face[0], face[2] - face[0]

        h = np.cross(self.direction, edge2)
        a = np.dot(edge1, h)

        if -tolerance < a < tolerance:
            return None

        f, s = 1 / a, self.origin - face[0]
        u = f * np.dot(s, h)

        if u < 0 or u > 1.0:
            return None

        q = np.cross(s, edge1)
        v = f * np.dot(self.direction, q)

        if v < 0 or u + v > 1.0:
            return None

        t = f * np.dot(edge2, q)

        if t > tolerance:
            return t

    def ray_box_intersect(self, bbox):
        t = (bbox.extent[0] - self.origin)/self.direction
        k = (bbox.extent[1] - self.origin)/self.direction

        tmin = np.max([np.max([np.min([t[0], k[0]]), np.min([t[1], k[1]])]), np.min([t[2], k[2]])])
        tmax = np.min([np.min([np.max([t[0], k[0]]), np.max([t[1], k[1]])]), np.max([t[2], k[2]])])

        if tmin > tmax:
            return False
        else:
            return tmin

    # casts a ray through the scene and returns all intersecting triangles
    def cast(self, tolerance, faces):
        casts = [(self.ray_triangle_intersect(f.vxs, tolerance), f) for f in faces]
        intersections = [x for x in casts if x[0] is not None]

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
        self.bbox = Bbox(np.array([np.min(self.vxs, axis=0), np.max(self.vxs, axis=0)]))  # min/max of each column


class Mesh:
    def __init__(self, mesh_id):
        self.id = mesh_id
        self.faces = set()
        self.mat = None
        self.bbox = None

    def get_bbox(self):
        face_bboxes = np.concatenate([f.bbox.extent for f in self.faces], axis=0)  # all vertices in mesh
        self.bbox = Bbox(np.array([np.min(face_bboxes[::2], axis=0), np.max(face_bboxes[1::2], axis=0)]))


class Bbox:
    def __init__(self, extent):
        self.extent = extent

    # xy, yz, xz
    def min_area_plane(self):
        return np.argmin([abs(self.extent[0][0]*self.extent[1][0]),
                          abs(self.extent[0][1]*self.extent[1][1]),
                          abs(self.extent[0][2]*self.extent[1][2])])


class PointCloud:
    def __init__(self):
        self.points = {}

    def write_xyz(self, fn):
        with open(fn, 'w+') as file:
            for i in self.points:
                nomat = ''
                file.write(f'{i[0]:.2f} {i[1]:.2f} {i[2]:.2f} '
                           f'{self.points[i] if self.points[i] is not None else nomat}\n')


def voxelize_worker(args):
    args[2][args[3]] *= args[0]
    args[2][args[4]] *= args[1]

    ray = Ray(args[6], args[5] + args[2])

    return ray, ray.cast(args[7], args[8])


class Scene:
    def __init__(self):
        self.meshes = None

    def mesh_bbox_union(self):
        bboxes = np.concatenate([m.bbox.extent for m in self.meshes])
        return Bbox(np.array([np.min(bboxes[::2], axis=0), np.max(bboxes[1::2], axis=0)]))

    # parallel voxelizer
    def voxelize(self, cell_size, process_count, ray_tolerance=0.00001):

        bbox = self.mesh_bbox_union()
        point_cloud = PointCloud()

        # get voxel grid shape and define direction of rays
        shape = (((bbox.extent[1] - bbox.extent[0]) // cell_size) + 1).astype(np.int32)

        # cast rays from bbox face with smallest area
        min_face = bbox.min_area_plane()
        min_face = 0

        direction = np.array([0, 0, 0])
        step_size = np.array([cell_size / 2, cell_size / 2, cell_size / 2])

        step_size[min_face - 1] = 0
        ray_transform = step_size + bbox.extent[0]
        direction[min_face - 1] = 1

        if min_face == 0:
            i_dim, j_dim = shape[0], shape[1]
            i_ind, j_ind, k_ind = 0, 1, 2
            sign_mult = np.array([cell_size*np.sign(i_dim), cell_size*np.sign(j_dim), -cell_size])

        elif min_face == 1:
            i_dim, j_dim = shape[1], shape[2]
            i_ind, j_ind, k_ind = 1, 2, 0
            sign_mult = np.array([-cell_size, cell_size*np.sign(i_dim), cell_size*np.sign(j_dim)])

        else:
            i_dim, j_dim = shape[0], shape[2]
            i_ind, j_ind, k_ind = 0, 2, 1
            sign_mult = np.array([cell_size*np.sign(i_dim), -cell_size, cell_size*np.sign(j_dim)])

        print('- Constructing SEADS grid')
        seads_grid = [[[] for j in range(j_dim)] for i in range(i_dim)]

        for m in self.meshes:
            for f in m.faces:
                tri_bbox = np.floor((f.bbox.extent - bbox.extent[0]) / cell_size).astype(np.int32)
                for j in range(tri_bbox[0][j_ind], tri_bbox[1][j_ind] + 1):
                    for i in range(tri_bbox[0][i_ind], tri_bbox[1][i_ind] + 1):
                        seads_grid[i][j].append(f)

        print('- Preparing arguments')
        cells = product(np.arange(0, i_dim, np.sign(i_dim)), np.arange(0, j_dim, np.sign(j_dim)))
        arguments = [(i[0], i[1],
                      deepcopy(sign_mult),
                      i_ind, j_ind,
                      ray_transform,
                      direction,
                      ray_tolerance,
                      seads_grid[i[0]][i[1]]) for i in cells]

        print('- Casting rays')
        p = Pool(process_count)
        intersections = p.map(voxelize_worker, arguments, chunksize=int((i_dim * j_dim) / process_count))
        p.close()
        p.join()

        print('- Building point cloud')
        # convert ray intersections to point cloud
        for ray, inter in intersections:
            if len(inter) > 1 and len(inter) % 2 == 0:
                for z in range(0, len(inter), 2):
                    dist = inter[z+1][0] - inter[z][0]

                    # add voxel at every cell between the two intersections
                    for d in np.arange(0, dist, cell_size):
                        cell = [ray.origin[0], ray.origin[1], ray.origin[2]]
                        cell[k_ind] += int(round((inter[z][0] + d) / cell_size))*cell_size
                        point_cloud.points[tuple(cell)] = inter[z][1].parent.mat

            elif inter:
                # add single voxel at single intersection
                cell = [ray.origin[0], ray.origin[1], ray.origin[2]]
                cell[k_ind] += round(inter[0][0] / cell_size)*cell_size
                point_cloud.points[tuple(cell)] = inter[0][1].parent.mat

        return point_cloud


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
                mesh.faces.add(Face(vxs=[vxs[vx_ind[0]], vxs[vx_ind[1]], vxs[vx_ind[2]]], parent=mesh))
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


if __name__ == '__main__':

    if len(argv) > 1:
        input_file, output_file = argv[1], argv[2]
    else:
        input_file, output_file = '../obj/TUDelft_campus.obj', 'output.obj'

    # read geometry data from .obj file and create necessary geometry objects
    print('Reading .obj file')
    scene = Scene()
    scene.meshes = ObjParser.read(input_file)

    print('Voxelizing scene')
    start = timer()
    voxelized_scene = scene.voxelize(cell_size=1, process_count=6, ray_tolerance=0.0000001)
    print(timer() - start)

    voxelized_scene.write_xyz('pc.xyz')

    # # works but is very slow
    # ObjParser.write(scene, 'out.obj')



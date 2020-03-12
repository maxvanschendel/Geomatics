# -- GEO1004.2020--hw01
# -- Max van Schendel
# -- 4384644


import numpy as np
from multiprocessing import Pool
from sys import argv
from copy import deepcopy
from itertools import product


# Ray object with direction and origin, can calculate intersection with face
class Ray:
    def __init__(self, direction, origin):
        self.direction = direction
        self.origin = origin

    # Möller–Trumbore ray intersection:
    # Python implementation of C++ code on Wikipedia
    def ray_triangle_intersect(self, face, epsilon):
        edge1, edge2 = face[1] - face[0], face[2] - face[0]

        h = np.cross(self.direction, edge2)
        a = np.dot(edge1, h)

        if -epsilon < a < epsilon:
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

        if t > epsilon:
            return t

    def ray_box_intersect(self, bbox):
        t = (bbox.extent[0] - self.origin) / self.direction
        k = (bbox.extent[1] - self.origin) / self.direction

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
class Tri:
    def __init__(self, vxs, parent=None):
        self.parent = parent
        self.a, self.b, self.c = vxs[0], vxs[1], vxs[2]
        self.edge1, self.edge2 = self.b.pos - self.a.pos, self.c.pos - self.a.pos

        self.vxs = np.array([v.pos for v in vxs])
        self.bbox = Bbox(np.array([np.min(self.vxs, axis=0), np.max(self.vxs, axis=0)]))  # min/max of each column

    def get_vertices(self):
        return self.a, self.b, self.c


class Quad:
    def __init__(self, vxs, parent=None):
        self.parent = parent

        self.vxs = np.array([v.pos for v in vxs])
        self.a, self.b, self.c, self.d = vxs[0], vxs[1], vxs[2], vxs[3]

    def get_vertices(self):
        return self.a, self.b, self.c, self.d


class Mesh:
    def __init__(self, mesh_id=None, mat=None):
        self.id = mesh_id
        self.faces = set()
        self.mat = mat
        self.bbox = None

    def get_bbox(self):
        face_bboxes = np.concatenate([f.bbox.extent for f in self.faces], axis=0)  # all vertices in mesh
        self.bbox = Bbox(np.array([np.min(face_bboxes[::2], axis=0), np.max(face_bboxes[1::2], axis=0)]))


class Bbox:
    def __init__(self, extent):
        self.extent = extent


class Point:
    def __init__(self, pos, mat=None):
        self.nbs = [None for i in range(27)]
        self.pos = pos
        self.mat = mat


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


class Voxel:
    def __init__(self, dim, pos, vertices, point=None):
        self.dim = dim
        self.pos = pos
        self.point = point
        self.mat = point.mat
        self.nbs = point.nbs
        self.mesh = self.box_mesh(vertices)

    def box_mesh(self, vertices):
        mesh = Mesh(mat=self.mat)

        nb6 = [self.nbs[4], self.nbs[10], self.nbs[12], self.nbs[13], self.nbs[15], self.nbs[21]]
        nbs_connectivity = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1], [0, 0, 1], [0, 1, 0], [1, 0, 0]])

        offsets = list(product([-self.dim[0]/2, self.dim[0]/2], repeat=2))
        off_axes = ((1, 2), (0, 2), (0, 1), (1, 0), (2, 0), (2, 1))

        for n in enumerate(nb6):
            if not n[1]:
                quad_vertices = []
                face_centroid = self.pos + nbs_connectivity[n[0]].astype(float) / 2
                axis2, axis3 = off_axes[n[0]][0], off_axes[n[0]][1]

                for v in offsets:
                    vertex = deepcopy(face_centroid)
                    vertex[axis2] += v[0]
                    vertex[axis3] += v[1]

                    vtup = tuple(vertex)
                    if vtup not in vertices:
                        vx_object = Vertex(vertex)
                        quad_vertices.append(vx_object)
                        vertices[vtup] = vx_object

                    else:
                        quad_vertices.append(vertices[vtup])

                mesh.faces.add(Quad(quad_vertices))

            else:
                n[1].nbs[n[1].nbs.index(self.point)] = None

        return mesh


class Scene:
    def __init__(self, meshes=None, mtllib=None):
        self.meshes = meshes
        self.mtllib = mtllib

    def mesh_bbox_union(self):
        bboxes = np.concatenate([m.bbox.extent for m in self.meshes])
        return Bbox(np.array([np.min(bboxes[::2], axis=0), np.max(bboxes[1::2], axis=0)]))

    # parallel voxelizer
    def voxelize(self, cell_size, process_count, epsilon=0.00001):
        bbox = self.mesh_bbox_union()
        point_cloud = PointCloud()

        # get voxel grid shape and define direction of rays
        shape = (((bbox.extent[1] - bbox.extent[0]) // cell_size) + 1).astype(np.int32)

        # cast rays from x, y, z axes
        for axis in [0, 1, 2]:
            print(f'- Casting rays from axis {axis}')
            direction = np.array([0, 0, 0])
            step_size = np.array([cell_size / 2, cell_size / 2, cell_size / 2])

            step_size[axis - 1], direction[axis - 1]= 0, 1
            ray_transform = step_size + bbox.extent[0]

            if axis == 0:
                i_dim, j_dim = shape[0], shape[1]
                i_ind, j_ind, k_ind = 0, 1, 2
                sign_mult = np.array([cell_size * np.sign(i_dim), cell_size * np.sign(j_dim), -cell_size])

            elif axis == 1:
                i_dim, j_dim = shape[1], shape[2]
                i_ind, j_ind, k_ind = 1, 2, 0
                sign_mult = np.array([-cell_size, cell_size * np.sign(i_dim), cell_size * np.sign(j_dim)])

            else:
                i_dim, j_dim = shape[0], shape[2]
                i_ind, j_ind, k_ind = 0, 2, 1
                sign_mult = np.array([cell_size * np.sign(i_dim), -cell_size, cell_size * np.sign(j_dim)])

            # spatially enumerated auxilliary data structure
            seads_grid = [[[] for j in range(j_dim)] for i in range(i_dim)]

            for m in self.meshes:
                for f in m.faces:
                    tri_bbox = np.floor((f.bbox.extent - bbox.extent[0]) / cell_size).astype(np.int32)
                    for j in range(tri_bbox[0][j_ind], tri_bbox[1][j_ind] + 1):
                        for i in range(tri_bbox[0][i_ind], tri_bbox[1][i_ind] + 1):
                            seads_grid[i][j].append(f)

            # prepare arguments for parellel processes
            cells = product(np.arange(0, i_dim, np.sign(i_dim)), np.arange(0, j_dim, np.sign(j_dim)))
            arguments = [(i[0], i[1],
                          deepcopy(sign_mult),
                          i_ind, j_ind,
                          ray_transform,
                          direction,
                          epsilon,
                          seads_grid[i[0]][i[1]]) for i in cells]

            p = Pool(process_count)
            intersections = p.map(voxelize_worker, arguments, chunksize=int((i_dim * j_dim) / process_count))
            p.close()
            p.join()

            # convert ray intersections to point cloud
            for ray, inter in intersections:
                # add voxel at every cell between the two intersections
                if len(inter) > 1 and len(inter) % 2 == 0:
                    for z in range(0, len(inter), 2):
                        dist = inter[z + 1][0] - inter[z][0]

                        for d in np.arange(0, dist, cell_size):
                            cell = [ray.origin[0] // cell_size, ray.origin[1] // cell_size, ray.origin[2] // cell_size]
                            cell[k_ind] += (inter[z][0] + d) // cell_size

                            point_cloud.points[tuple(cell)] = Point(mat=inter[z][1].parent.mat, pos=tuple(cell))

                # add single voxel at single intersection
                elif inter:
                    cell = [int(ray.origin[0]/cell_size), int(ray.origin[1]/cell_size), int(ray.origin[2]/cell_size)]
                    cell[k_ind] += int(round(inter[0][0] / cell_size))

                    point_cloud.points[tuple(cell)] = Point(mat=inter[0][1].parent.mat, pos=tuple(cell))

        # build 26-connectivity
        print('- Building 26-connectivity')
        neighbours = [i for i in product([-1, 0, 1], repeat=3) if i != (0, 0, 0)]

        for p in point_cloud.points:
            for i in enumerate(neighbours):
                cur_point = point_cloud.points[p]
                if not cur_point.nbs[i[0]]:
                    nb = point_cloud.points.get(tuple(map(lambda i, j: i + j, p, i[1])))
                    cur_point.nbs[i[0]] = nb

        # create voxel mesh from point cloud
        print('- Creating voxel mesh')
        meshes = set()
        vertices = {}
        for x in point_cloud.points:
            meshes.add(Voxel(dim=(1, 1, 1), vertices=vertices, pos=x, point=point_cloud.points[x]).mesh)

        return Scene(meshes, self.mtllib)


# reads and writes obj files to and
class ObjParser:
    @staticmethod
    def read(fn):
        vxs, meshes = [], set()
        mesh = Mesh(None)
        mtllib = None

        for i in [i.replace('\n', '').split(' ') for i in open(fn).readlines()]:
            if i[0] == 'v':
                vxs.append(Vertex(np.asarray([float(i) for i in i[1:]])))
            elif i[0] == 'f':
                # extract vertices froms tring
                vx_ind = [int(i.split('/')[0]) - 1 for i in i[1:]]

                # simple triangle face
                if len(vx_ind) == 3:
                    mesh.faces.add(Tri(vxs=[vxs[vx_ind[0]], vxs[vx_ind[1]], vxs[vx_ind[2]]], parent=mesh))

                # split quad into two tris
                elif len(vx_ind) == 4:
                    mesh.faces.add(Tri(vxs=[vxs[vx_ind[0]], vxs[vx_ind[1]], vxs[vx_ind[2]]], parent=mesh))
                    mesh.faces.add(Tri(vxs=[vxs[vx_ind[0]], vxs[vx_ind[2]], vxs[vx_ind[3]]], parent=mesh))

            elif i[0] == 'o':
                mesh = Mesh(i[1])
                meshes.add(mesh)

            elif i[0] == 'usemtl':
                mesh.mat = i[1]

            elif i[0] == 'mtllib':
                mtllib = i[1]

        # if file doesnt use mesh id
        if not len(meshes):
            meshes = [mesh]

        meshes = {m for m in meshes if len(m.faces)}
        [m.get_bbox() for m in meshes]

        return Scene(meshes, mtllib)

    @staticmethod
    def write(scene, out):
        all_verts = []
        output_meshes = scene.meshes
        added_verts = {}

        # write to obj file
        with open(out, 'w') as obj_file:
            if scene.mtllib:
                obj_file.write(f'mtllib {scene.mtllib}\n')

            # write vertex strings
            vxs_i, vcur = {}, 0
            for m in output_meshes:
                faces = list(m.faces)
                vertices = list(set().union(*[i.get_vertices() for i in faces]))

                for i in vertices:
                    if i not in added_verts:
                        all_verts.append(' '.join([format(x, '.2f') for x in i.pos]))
                        added_verts[i], vxs_i[i] = vcur, vcur
                        vcur += 1
                    else:
                        vxs_i[i] = added_verts[i]

            for vstring in all_verts:
                obj_file.write(f'v {vstring}\n')

            # write face strings
            for m in enumerate(output_meshes):
                faces = list(m[1].faces)
                if faces:
                    obj_file.write(f'o {m[0] + 1}\n')
                    if m[1].mat:
                        obj_file.write(f'usemtl {m[1].mat}\n')

                    for f in faces:
                        if type(f) == Tri:
                            fstring = f'{vxs_i[f.a] + 1} {vxs_i[f.b] + 1} {vxs_i[f.c] + 1}'

                        elif type(f) == Quad:
                            fstring = f'{vxs_i[f.a] + 1} {vxs_i[f.b]+ 1} {vxs_i[f.d] + 1} {vxs_i[f.c] + 1}'

                        obj_file.write('f ' + fstring + '\n')


if __name__ == '__main__':
    if len(argv) > 1:
        input_file, output_file, cell_size = argv[1], argv[2], argv[3]
    else:
        input_file, output_file, cell_size = '../obj/TUDelft_campus.obj', 'output.obj', 1

    # read geometry data from .obj file and create necessary geometry objects
    print('Reading .obj file')
    input_geo = ObjParser.read(input_file)

    print('Voxelizing scene')
    voxels = input_geo.voxelize(cell_size=cell_size, process_count=6, epsilon=0.0000001)

    print('Writing obj')
    ObjParser.write(voxels, 'out.obj')

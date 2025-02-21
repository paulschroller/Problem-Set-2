import numpy as np
import matplotlib.pyplot as plt
from random import randint
import math
from scipy.spatial import Delaunay


data = np.loadtxt("./mesh.dat", skiprows=1)

#convex hull
plt.plot(data[:,0], data[:,1], linewidth=0,marker='o',color='blue', label='data')



def jarvismarch(array):
    basepoint = array[0]
    basepointindex = 0
    outputhull = []
    for i in range(len(array)):
        if array[i][0] < basepoint[0]:
            basepoint = array[i]
            basepointindex = i
        elif array[i][0] == basepoint[0]:
            if array[i][1] > basepoint[1]:
                basepoint = array[i]
                basepointindex=i
    originalbasepoint = array[basepointindex]
    outputhull.append(originalbasepoint)
    while (True):
        nextpoint = array[0]
        for i in range(1,len(array)):
            if array[i][0] != basepoint[0] or array[i][1] != basepoint[1]:
                if (nextpoint[0]-basepoint[0])*(array[i][1]-basepoint[1]) - (nextpoint[1]-basepoint[1])*(array[i][0]-basepoint[0]) >0:
                    nextpoint = array[i]
        basepoint = nextpoint
        if nextpoint[0] == originalbasepoint[0] and nextpoint[1] == originalbasepoint[1]:
            break
        else:
            outputhull.append(basepoint)
    outputhull.append(originalbasepoint)
    return outputhull

def xandy(array, offset):
    arrayx = []
    arrayy = []
    for i in range(len(array)):
        arrayx.append(array[i][0] + offset)
        arrayy.append(array[i][1])
    return arrayx, arrayy

hullx, hully = xandy(jarvismarch(data),0)

plt.plot(hullx, hully,color='red')
plt.savefig("./plots/problem3a.png")
plt.show()
plt.clf


points_2d = data[:, :2]

# Lift to 3D using z = x^2 + y^2
x, y = points_2d[:, 0], points_2d[:, 1]
z = x**2 + y**2
points_3d = np.column_stack((x, y, z))  # Create (x, y, z) points

#delaunay triangulation
tri = Delaunay(points_2d)

#part c)
def inducedmetric(x,y):
    return [[1 + 4*x**2, 4*x*y],[4*x*y, 1+4*y**2]]

#part d)
def compute_face_normals(tri, points_3d):
    normals = []
    for simplex in tri.simplices:
        p1, p2, p3 = points_3d[simplex]
        v1, v2 = p2 - p1, p3 - p1
        normal = np.cross(v1, v2)
        normal /= np.linalg.norm(normal)  # Normalize the normal
        normals.append(normal)
    return np.array(normals)

face_normals = compute_face_normals(tri, points_3d)

#part e)
def compute_vertex_normals(tri, face_normals, num_points):
    vertex_normals = np.zeros((num_points, 3))
    count = np.zeros(num_points)

    for i, simplex in enumerate(tri.simplices):
        for vertex in simplex:
            vertex_normals[vertex] += face_normals[i]
            count[vertex] += 1

    vertex_normals /= count[:, None]  # Normalize by count
    vertex_normals /= np.linalg.norm(vertex_normals, axis=1, keepdims=True)  # Normalize vectors
    return vertex_normals

vertex_normals = compute_vertex_normals(tri, face_normals, len(points_3d))

# Plot face normals on the lifted mesh
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], triangles=tri.simplices, alpha=0.5, edgecolor='gray')

# Plot surface normals
for simplex, normal in zip(tri.simplices, face_normals):
    center = np.mean(points_3d[simplex], axis=0)  # Compute triangle center
    ax.quiver(center[0], center[1], center[2], normal[0], normal[1], normal[2], color='red', length=0.2, normalize=True)

ax.set_title("Surface Normals")
plt.savefig("./plots/problem3esurfacenormals")
plt.show()
plt.clf

# Plot vertex normals
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], triangles=tri.simplices, alpha=0.5, edgecolor='gray')

# Plot vertex normals
for i in range(len(points_3d)):
    ax.quiver(points_3d[i, 0], points_3d[i, 1], points_3d[i, 2], vertex_normals[i, 0], vertex_normals[i, 1], vertex_normals[i, 2], color='blue', length=0.2, normalize=True)

ax.set_title("Vertex Normals")
plt.savefig("./plots/problem3evertexnormals")
plt.show()
plt.clf

#part f)
def secondfundamentalform(x,y):
    return [[2/((4*x**2+4*y**2+1)**0.5), 0], [0, 2/((4*x**2+4*y**2+1)**0.5)]]
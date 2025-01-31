import numpy as np
import matplotlib.pyplot as plt
from random import randint
import math
from scipy.spatial import Delaunay


#Part a)
data = np.loadtxt("/root/Desktop/host/ProblemSet2/mesh.dat", skiprows=1)
#plt.plot(data[:,0], data[:,1], linewidth=0,marker='o',color='blue', label='data')



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

#plt.plot(hullx, hully,color='red')
tri = Delaunay(data)
trisimplices = tri.simplices
#plt.triplot(data[:,0], data[:,1], trisimplices)
#plt.xlabel("X")
#plt.ylabel("Y")
#plt.show

#part b)

liftedpoints = []
for i in range(len(data)):
    nextpoint = [data[i][0], data[i][1], (data[i][0]**2+data[i][1]**2)]
    liftedpoints.append(nextpoint)

def trianglearea(point1, point2, point3):
    vector1 = [point2[0]-point1[0], point2[1]-point1[1], point2[2]-point1[2]]
    vector2 = [point3[0]-point1[0], point3[1]-point1[1], point3[2]-point1[2]]
    mag1 = ((vector1[0]**2)+(vector1[1]**2)+(vector1[2]**2))**0.5
    mag2 = ((vector2[0]**2)+(vector2[1]**2)+(vector2[2]**2))**0.5
    dotproduct = vector1[0]*vector2[0]+vector1[1]*vector2[1]+vector1[2]*vector2[2]
    costheta = dotproduct/(mag1*mag2)
    sintheta = (1-costheta**2)**0.5
    return 0.5*mag1*mag2*sintheta


triangleareas = []
for i in range(len(trisimplices)):
    point1 = [data[trisimplices[i][0]][0], data[trisimplices[i][0]][1], 0]
    point2 = [data[trisimplices[i][1]][0], data[trisimplices[i][1]][1], 0]
    point3 = [data[trisimplices[i][2]][0], data[trisimplices[i][2]][1], 0]
    triangleareas.append(trianglearea(point1, point2, point3))


liftedareas = []
for i in range(len(trisimplices)):
    point1 = liftedpoints[trisimplices[i][0]]
    point2 = liftedpoints[trisimplices[i][1]]
    point3 = liftedpoints[trisimplices[i][2]]
    liftedareas.append(trianglearea(point1, point2, point3))

trianglecenters = []
for i in range(len(trisimplices)):
    point1 = data[trisimplices[i][0]]
    point2 = data[trisimplices[i][1]]
    point3 = data[trisimplices[i][2]]
    centerpoint = [(point1[0]+point2[0]+point3[0])/3, (point1[1]+point2[1]+point3[1])/3]
    trianglecenters.append(centerpoint)

arearatio = []
for i in range(len(triangleareas)):
    arearatio.append(liftedareas[i]/triangleareas[i])

print(arearatio)

centersx, centersy = xandy(trianglecenters,0)
plt.scatter(centersx, centersy, c=arearatio, cmap='viridis', edgecolor='k', vmin = 0, vmax = 40)
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math


#a, b, c will be x, y, z for cartesian, r, theta, z for cylindrical, r, theta, phi for spherical where theta is the azimuthal angle
#frombasis and tobasis will be written as cart, sph, or cyl
def switchbasis(a, b, c, frombasis, tobasis):
    result = [a,b,c]
    if frombasis == "cart" and tobasis== "cart":
        result = [a,b,c]
    if frombasis == "cart" and tobasis == "sph":
        result = [(a**2 + b**2 + c**2)**0.5, math.atan(((a**2 + b**2)**0.5)/c), math.atan(b/a)]
    if frombasis == "cart" and tobasis == "cyl":
        result = [(a**2 + b**2)**0.5, math.atan(b/a), c]
    if frombasis == "sph" and tobasis == "cart":
        result = [a*math.sin(b)*math.cos(c), a*math.sin(b)*math.sin(c), a*math.cos(b)]
    if frombasis == "sph" and tobasis == "sph":
        result = [a,b,c]
    if frombasis == "sph" and tobasis == "cyl":
        result = [a*math.sin(b), c, a*math.cos(b)]
    if frombasis == "cyl" and tobasis == "cart":
        result = [a*math.cos(b), a*math.sin(b), c]
    if frombasis == "cyl" and tobasis == "sph":
        result = [(a**2 + c**2)**0.5, np.atan(a/c), b]
    if frombasis == "cyl" and tobasis == "cyl":
        result = [a,b,c]
    return np.array(result)


#For part b
def compute_basis_vectors(theta, phi):
    e_r = np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])
    e_theta = np.array([np.cos(theta)*np.cos(phi), np.cos(theta)*np.sin(phi), -np.sin(theta)])
    e_phi = np.array([-np.sin(phi), np.cos(phi), 0])
    return e_r, e_theta, e_phi


theta_vals = np.linspace(0, np.pi, 10)
phi_vals = np.linspace(0, 2*np.pi, 20)

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')

u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
x = np.sin(v) * np.cos(u)
y = np.sin(v) * np.sin(u)
z = np.cos(v)
ax.plot_surface(x, y, z, color='c', alpha=0.3, edgecolor='k')

for theta in theta_vals:
    for phi in phi_vals:
        point = switchbasis(1, theta, phi, "sph", "cart")
        e_r, e_theta, e_phi = compute_basis_vectors(theta, phi)
        scale = 0.1
        ax.quiver(*point, *e_r * scale, color='r', linewidth=2, label='e_r' if (theta == theta_vals[0] and phi == phi_vals[0]) else "")
        ax.quiver(*point, *e_theta * scale, color='g', linewidth=2, label='e_theta' if (theta == theta_vals[0] and phi == phi_vals[0]) else "")
        ax.quiver(*point, *e_phi * scale, color='b', linewidth=2, label='e_phi' if (theta == theta_vals[0] and phi == phi_vals[0]) else "")

plt.savefig("./plots/problem1b.png")
plt.show()
plt.clf

#Part c) - You can't plot in spherical basis because pyplot works in cartesian, so you have to convert to cartesian first.



#Part d)
def f(x, y):
    return x**2 + y**2 #sample function

x_vals = np.linspace(-2, 2, 20) #range for x values on plot
y_vals = np.linspace(-2, 2, 20) #range for y values on plot
X, Y = np.meshgrid(x_vals, y_vals)
Z = f(X, Y)

dZ_dx, dZ_dy = np.gradient(Z, x_vals, y_vals)

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X, Y, Z, color='c', alpha=0.3, edgecolor='k')

for i in range(0, len(x_vals), 4):
    for j in range(0, len(y_vals), 4):
        x, y, z = X[i, j], Y[i, j], Z[i, j]
        dzdx, dzdy = dZ_dx[i, j], dZ_dy[i, j]

        normal = np.array([-dzdx, -dzdy, 1])
        normal /= np.linalg.norm(normal)

        t1 = np.array([1, 0, dzdx])
        t1 /= np.linalg.norm(t1)

        t2 = np.array([0, 1, dzdy])
        t2 /= np.linalg.norm(t2)

        scale = 0.3

        ax.quiver(x, y, z, *(normal * scale), color='r', label="Normal" if i == 0 and j == 0 else "")
        ax.quiver(x, y, z, *(t1 * scale), color='g', label="Tangent 1" if i == 0 and j == 0 else "")
        ax.quiver(x, y, z, *(t2 * scale), color='b', label="Tangent 2" if i == 0 and j == 0 else "")

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title("Surface with Local Coordinate Frames")

handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys())
plt.savefig("./plots/problem1d.png")
plt.show()
plt.clf


#Part e)
alphavar = 0.2
betavar = 0.1

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')

#plotting the sphere
u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
x = np.sin(v) * np.cos(u)
y = np.sin(v) * np.sin(u)
z = np.cos(v)
ax.plot_surface(x, y, z, color='c', alpha=0.3, edgecolor='k')

basepoints = []
thetavalues = np.linspace(2, 5, 10)
for i in range(len(thetavalues)):
    newpoint = [1,thetavalues[i]*np.pi/10,0]
    basepoints.append(newpoint)
for i in range(len(basepoints)):
    e_r, e_theta, e_phi = compute_basis_vectors(basepoints[i][1], basepoints[i][2])
    vector = np.array([alphavar * e_theta[0] + betavar * e_phi[0], alphavar * e_theta[1] + betavar * e_phi[1], alphavar * e_theta[2] + betavar*e_phi[2]])
    basisconverted = switchbasis(basepoints[i][0],basepoints[i][1],basepoints[i][2], "sph", "cart")
    ax.quiver(*basisconverted, *vector, color='r', linewidth=2)


ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title("Parallel Transport phi=0")
plt.savefig("./plots/problem1e.png")
plt.show()
plt.clf


#part f)
alphavar = 0.2
betavar = 0.1
theta0 = 0.3

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')

#plotting the sphere
u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
x = np.sin(v) * np.cos(u)
y = np.sin(v) * np.sin(u)
z = np.cos(v)
ax.plot_surface(x, y, z, color='c', alpha=0.3, edgecolor='k')

basepoints = []
phivalues = np.linspace(0, 2, 20)
for i in range(len(phivalues)):
    newpoint = [1,theta0,phivalues[i]*np.pi]
    basepoints.append(newpoint)
for i in range(len(basepoints)):
    e_r, e_theta, e_phi = compute_basis_vectors(basepoints[i][1], basepoints[i][2])
    vector = np.array([alphavar * e_theta[0] + betavar * e_phi[0], alphavar * e_theta[1] + betavar * e_phi[1], alphavar * e_theta[2] + betavar*e_phi[2]])
    basisconverted = switchbasis(basepoints[i][0],basepoints[i][1],basepoints[i][2], "sph", "cart")
    ax.quiver(*basisconverted, *vector, color='r', linewidth=2)


ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title("Parallel Transport at theta_0 from phi=0 to phi=2pi")
plt.savefig("./plots/problem1f.png")
plt.show()
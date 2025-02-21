import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

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

def compute_basis_vectors(theta, phi):
    e_r = np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])
    e_theta = np.array([np.cos(theta)*np.cos(phi), np.cos(theta)*np.sin(phi), -np.sin(theta)])
    e_phi = np.array([-np.sin(phi), np.cos(phi), 0])
    return e_r, e_theta, e_phi


#part a)
def stereographic_projection(x, y, z):
    """Perform stereographic projection from sphere to plane."""
    return x / (1 - z), y / (1 - z)


# Generate sphere coordinates
theta = np.linspace(0, 2*np.pi, 100)
phi = np.linspace(0, np.pi, 50)
Theta, Phi = np.meshgrid(theta, phi)

# Convert to Cartesian coordinates
X = np.cos(Theta) * np.sin(Phi)
Y = np.sin(Theta) * np.sin(Phi)
Z = np.cos(Phi)

# Define two intersecting curves (latitude and longitude)
lat_phi = np.pi / 4  # Latitude at 45 degrees
long_theta = np.pi / 3  # Longitude at 60 degrees

lat_x = np.cos(theta) * np.sin(lat_phi)
lat_y = np.sin(theta) * np.sin(lat_phi)
lat_z = np.full_like(theta, np.cos(lat_phi))

long_x = np.cos(long_theta) * np.sin(phi)
long_y = np.sin(long_theta) * np.sin(phi)
long_z = np.cos(phi)

# Compute tangent vectors before projection
lat_x_interp = np.interp(phi, theta, lat_x)
lat_y_interp = np.interp(phi, theta, lat_y)
lat_z_interp = np.interp(phi, theta, lat_z)

dlat_x = np.gradient(lat_x_interp)
dlat_y = np.gradient(lat_y_interp)
dlat_z = np.gradient(lat_z_interp)

dlong_x = np.gradient(long_x)
dlong_y = np.gradient(long_y)
dlong_z = np.gradient(long_z)

# Compute angles before projection
dot_product = dlat_x * dlong_x + dlat_y * dlong_y + dlat_z * dlong_z
mag_lat = np.sqrt(dlat_x**2 + dlat_y**2 + dlat_z**2)
mag_long = np.sqrt(dlong_x**2 + dlong_y**2 + dlong_z**2)
angle_before = np.arccos(dot_product / (mag_lat * mag_long))

# Apply stereographic projection
lat_x_proj, lat_y_proj = stereographic_projection(lat_x_interp, lat_y_interp, lat_z_interp)
long_x_proj, long_y_proj = stereographic_projection(long_x, long_y, long_z)

# Compute tangent vectors after projection
dlat_x_proj = np.gradient(lat_x_proj)
dlat_y_proj = np.gradient(lat_y_proj)
dlong_x_proj = np.gradient(long_x_proj)
dlong_y_proj = np.gradient(long_y_proj)

# Compute angles after projection
dot_product_proj = dlat_x_proj * dlong_x_proj + dlat_y_proj * dlong_y_proj
mag_lat_proj = np.sqrt(dlat_x_proj**2 + dlat_y_proj**2)
mag_long_proj = np.sqrt(dlong_x_proj**2 + dlong_y_proj**2)
angle_after = np.arccos(dot_product_proj / (mag_lat_proj * mag_long_proj))

# Plot original sphere with curves
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(121, projection='3d')
ax.plot_surface(X, Y, Z, color='c', alpha=0.3)
ax.plot(lat_x_interp, lat_y_interp, lat_z_interp, 'r', label='Latitude')
ax.plot(long_x, long_y, long_z, 'b', label='Longitude')
ax.set_title("Unit Sphere with Curves")
ax.legend()

# Plot stereographic projection
ax2 = fig.add_subplot(122)
ax2.plot(lat_x_proj, lat_y_proj, 'r', label='Projected Latitude')
ax2.plot(long_x_proj, long_y_proj, 'b', label='Projected Longitude')
ax2.set_title("Stereographic Projection")
ax2.legend()
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.grid()

plt.savefig("./plots/problem2a.png")
plt.show()
plt.clf

# Print angle comparison
print("Angle before projection:", np.degrees(angle_before))
print("Angle after projection:", np.degrees(angle_after))
print("Difference between angles before and after projection", np.degrees(angle_before) - np.degrees(angle_after))


#part b)

# Generate sphere coordinates
theta = np.linspace(0, 2*np.pi, 100)
phi = np.linspace(0, np.pi, 50)
Theta, Phi = np.meshgrid(theta, phi)

# Convert to Cartesian coordinates
X = np.cos(Theta) * np.sin(Phi)
Y = np.sin(Theta) * np.sin(Phi)
Z = np.cos(Phi)

#make 2 great circles
circlex = np.cos(theta)
circley = np.sin(theta)
theta0 = np.pi/5
theta1 = np.pi/7
greatcirclex = circlex*np.cos(theta0)
greatcircley = circley
greatcirclez = circlex*np.sin(theta0)
greatcirclex2 = circlex
greatcircley2 = circley*np.cos(theta1)
greatcirclez2 = circley*np.sin(theta1)

#stereographic projection
greatcircleprojx, greatcircleprojy = stereographic_projection(greatcirclex, greatcircley, greatcirclez)
greatcircleprojx2, greatcircleprojy2 = stereographic_projection(greatcirclex2, greatcircley2, greatcirclez2)

# Plot original sphere with curves
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(121, projection='3d')
ax.plot_surface(X, Y, Z, color='c', alpha=0.3)
ax.plot(greatcirclex, greatcircley, greatcirclez, 'r')
ax.plot(greatcirclex2, greatcircley2, greatcirclez2, 'b')

# Plot stereographic projection
ax2 = fig.add_subplot(122)
ax2.plot(greatcircleprojx, greatcircleprojy, 'r', label='Projected Great Circle 1')
ax2.plot(greatcircleprojx2, greatcircleprojy2, 'b', label='Projected Great Circle 2')
ax2.set_title("Stereographic Projection")
ax2.legend()
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.grid()

plt.savefig("./plots/problem2b")
plt.show()
plt.clf
#so great circles appear as circles after stereographic projection


#part c)

fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122)
alphavar = 0.2
betavar = 0.1
theta0 = 0.7
# Generate sphere coordinates
theta = np.linspace(0, 2*np.pi, 100)
phi = np.linspace(0, np.pi, 50)
Theta, Phi = np.meshgrid(theta, phi)

# Convert to Cartesian coordinates
X = np.cos(Theta) * np.sin(Phi)
Y = np.sin(Theta) * np.sin(Phi)
Z = np.cos(Phi)

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
    vectorproj = stereographic_projection(*vector)
    basisproj = stereographic_projection(*basisconverted)
    ax2.quiver(*basisproj, *vectorproj, angles='xy', scale_units='xy', color='r', linewidth=2)

# Plot original sphere with curves
ax.plot_surface(X, Y, Z, color='c', alpha=0.3)
plt.savefig("./plots/problem2c")


#part d)

vectorbase = [3**(-0.5),3**(-0.5),3**(-0.5)]
vector1end = [-2, 5, 3]
vector2end = [1, 3, 4]
dotprodbefore = 0
for i in range(len(vector1end)):
    dotprodbefore = dotprodbefore + (vector1end[i]-vectorbase[i])*(vector2end[i] - vectorbase[i])
vectorbaseproj = stereographic_projection(*vectorbase)
vector1endproj = stereographic_projection(*vector1end)
vector2endproj = stereographic_projection(*vector2end)
dotprodafter = 0
for i in range(len(vector1endproj)):
    dotprodafter = dotprodafter + (vector1endproj[i] - vectorbaseproj[i])*(vector2endproj[i] - vectorbaseproj[i])
print("Dot product before stereographic projection: ", dotprodbefore)
print("Dot product after stereographic projection ", dotprodafter)
if (dotprodafter - dotprodbefore)**2 > 0.001:
    print("Dot product is not preserved by stereographic projection")

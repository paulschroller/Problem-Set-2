import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

plt.show()

# Print angle comparison
print("Angle before projection:", np.degrees(angle_before))
print("Angle after projection:", np.degrees(angle_after))
print("Difference between angles before and after projection", np.degrees(angle_before) - np.degrees(angle_after))
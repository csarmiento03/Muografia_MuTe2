# scan_clear_sky_fov_P1.py

import numpy as np
import pandas as pd
from MuTeLogic import MuographySimulation
import rasterio
from math import atan2, radians, degrees, sin, cos

# --------- Cargador SRTM ---------
class SRTMDataLoader:
    def __init__(self, file_paths):
        self.data = {}
        for path in file_paths:
            with rasterio.open(path) as dataset:
                elevation_data = dataset.read(1)
                extent = dataset.bounds
                self.data[path] = {
                    "elevation": elevation_data,
                    "bounds": extent,
                    "nodata": dataset.nodata
                }

    def get_altitude(self, latitude, longitude):
        for file_data in self.data.values():
            bounds = file_data["bounds"]
            if bounds.left <= longitude <= bounds.right and bounds.bottom <= latitude <= bounds.top:
                elevation_data = file_data["elevation"]
                lon = np.linspace(bounds.left, bounds.right, elevation_data.shape[1])
                lat = np.linspace(bounds.top, bounds.bottom, elevation_data.shape[0])
                lon_idx = (np.abs(lon - longitude)).argmin()
                lat_idx = (np.abs(lat - latitude)).argmin()
                altitude = elevation_data[lat_idx, lon_idx]
                return altitude if altitude != file_data["nodata"] else np.nan
        return np.nan

# --------- Cargar topografía ---------
region = [4.466944, 4.500833, -75.404720, -75.372694, "Cerro Machín"]
srtm_files = ["N04W075.hgt", "N04W076.hgt"]
srtm_loader = SRTMDataLoader(srtm_files)

sim = MuographySimulation(region, points=200, srtm1_data=srtm_loader)
sim.load_elevation_data()

# --------- Punto P1 ---------
P1_lat = 4.492298
P1_lon = -75.381092
P1_alt = srtm_loader.get_altitude(P1_lat, P1_lon)

# --------- Buscar el punto más alto manualmente ---------
latitudes = np.linspace(region[0], region[1], 200)
longitudes = np.linspace(region[2], region[3], 200)
max_elev = -np.inf
max_lat, max_lon = None, None

for lat in latitudes:
    for lon in longitudes:
        elev = srtm_loader.get_altitude(lat, lon)
        if elev is not None and not np.isnan(elev):
            if elev > max_elev:
                max_elev = elev
                max_lat, max_lon = lat, lon

# --------- Calcular azimut ---------
lat1, lon1 = radians(P1_lat), radians(P1_lon)
lat2, lon2 = radians(max_lat), radians(max_lon)
delta_lon = lon2 - lon1

x = sin(delta_lon) * cos(lat2)
y = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(delta_lon)
azimuth_rad = atan2(x, y)
azimuth_deg = (degrees(azimuth_rad) + 360) % 360

# --------- Definir ángulos restringidos ---------
fov = 140  # grados
phi_vals = np.arange(azimuth_deg - fov/2, azimuth_deg + fov/2 + 0.5, 0.5) % 360
theta_vals = np.radians(np.arange(60.0, 120.5, 0.5))
phi_vals_rad = np.radians(phi_vals)

# --------- Simulación ---------
num_points = 140
max_distance = 2000  # metros
deg_per_m = 1 / 111320  # grados por metro aprox (latitud)
valid_angles = []

print(f"🔭 Escaneando desde P1 hacia azimut {azimuth_deg:.2f}° (domo en {max_lat:.5f}, {max_lon:.5f})...")

for theta in theta_vals:
    for phi in phi_vals_rad:
        dx = np.sin(theta) * np.cos(phi)
        dy = np.sin(theta) * np.sin(phi)
        dz = np.cos(theta)  # hacia arriba

        t = np.linspace(0, 1, num_points)
        x = P1_lon + dx * max_distance * t * deg_per_m / np.cos(np.radians(P1_lat))
        y = P1_lat + dy * max_distance * t * deg_per_m
        z = P1_alt + dz * max_distance * t

        z_topo = np.array([
            srtm_loader.get_altitude(lat, lon) for lat, lon in zip(y, x)
        ])

        if np.all(z >= z_topo):
            valid_angles.append((np.degrees(theta), np.degrees(phi)))

# --------- Guardar resultados ---------
df = pd.DataFrame(valid_angles, columns=["theta_deg", "phi_deg"])
df.to_csv("AngulosNoBloqueados.csv", index=False)
print(f"✅ Ángulos dentro del FOV sin obstrucción guardados: {len(df)}")



# --------- Visualización 2D de distribución angular ---------
import matplotlib.pyplot as plt

# Crear grilla angular para graficar
theta_unique = np.sort(df["theta_deg"].unique())
phi_unique = np.sort(df["phi_deg"].unique())
theta_grid, phi_grid = np.meshgrid(theta_unique, phi_unique)

# Crear máscara binaria
valid_pairs = set((round(row["theta_deg"], 2), round(row["phi_deg"], 2)) for _, row in df.iterrows())
mask = np.zeros_like(theta_grid, dtype=int)
for i in range(phi_grid.shape[0]):
    for j in range(theta_grid.shape[1]):
        if (round(theta_grid[i, j], 2), round(phi_grid[i, j], 2)) in valid_pairs:
            mask[i, j] = 1

# Graficar
fig, ax = plt.subplots(figsize=(10, 6))
c = ax.pcolormesh(phi_grid, theta_grid, mask, cmap="viridis", shading="auto")
ax.invert_yaxis()
ax.set_xlabel("Azimuth angle φ (deg)")
ax.set_ylabel("Zenith angle θ (deg)")

ax.set_title("Clear-Sky Angular Distribution from P1 (FOV)")
fig.colorbar(c, ax=ax, label="1 = clear sky")
plt.tight_layout()
plt.show()


# --------- Visualización 3D topografía + cono sólido (ajustado a la malla) ---------
from mpl_toolkits.mplot3d import Axes3D

# Crear grilla de topografía
lat_grid, lon_grid = np.meshgrid(latitudes, longitudes, indexing="ij")
z_topo = np.array([[srtm_loader.get_altitude(lat, lon) for lon in longitudes] for lat in latitudes])

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(lon_grid, lat_grid, z_topo, cmap="terrain", alpha=0.9, linewidth=0)

# Dibujar líneas del cono dentro del dominio de la malla
theta_max = np.radians(90)
phi_span = np.radians(np.linspace(azimuth_deg - fov/2, azimuth_deg + fov/2, 30))
r_max = 0.005  # grados latitud (~550 m), ajustado al tamaño de malla

for phi in phi_span:
    dx = np.sin(theta_max) * np.cos(phi)
    dy = np.sin(theta_max) * np.sin(phi)
    dz = np.cos(theta_max)

    lat_tip = P1_lat + dy * r_max
    lon_tip = P1_lon + dx * r_max / np.cos(np.radians(P1_lat))
    alt_tip = P1_alt + dz * 2000  # hasta 2 km de altura

    ax.plot([P1_lon, lon_tip], [P1_lat, lat_tip], [P1_alt, alt_tip], color="red", alpha=0.6)

ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_zlabel("Elevation (m)")
ax.set_title("Topography + Field of View Cone from P1")
plt.tight_layout()
plt.show()

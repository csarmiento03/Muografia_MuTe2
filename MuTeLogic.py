############################################
# MuTeLogic.py – utilities for muography   #
# ¡Versión corregida y unificada!          #
############################################
"""
Módulo con utilidades para simulación de muografía:
  • SRTMDataLoader: carga mosaicos SRTM (.hgt)
  • MuographySimulation: manejo de topografía, proyecciones y
    cálculo de recorridos dentro del terreno.

Todos los métodos adicionales (set_observation_points, path_length,
plot_with_projection, etc.) han sido integrados dentro de la clase
MuographySimulation para poder usarse como atributos de instancia.
"""

from __future__ import annotations

import os
import math
from pathlib import Path
from typing import Tuple, List

import numpy as np
import matplotlib.pyplot as plt
import rasterio
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 – necesario para proyección 3D

###############################################################################
#  CARGA DE DATOS SRTM                                                       #
###############################################################################


class SRTMDataLoader:
    """Carga uno o varios archivos .hgt (SRTM) y permite consultar altitud."""

    def __init__(self, file_paths: List[str]):
        self.data = {}
        for path in file_paths:
            with rasterio.open(path) as ds:
                self.data[path] = {
                    "elevation": ds.read(1),
                    "bounds": ds.bounds,
                    "nodata": ds.nodata,
                }

    # ---------------------------------------------------------------------
    def get_altitude(self, latitude: float, longitude: float) -> float:
        """Devuelve la altitud (m) interpolada más cercana en el mosaico."""
        for d in self.data.values():
            b = d["bounds"]
            if b.left <= longitude <= b.right and b.bottom <= latitude <= b.top:
                elev = d["elevation"]
                # ejes en rasterio: (fila ↔ lat ↘, columna ↔ lon ↗)
                lon_vec = np.linspace(b.left, b.right, elev.shape[1])
                lat_vec = np.linspace(b.top, b.bottom, elev.shape[0])
                j = np.abs(lon_vec - longitude).argmin()
                i = np.abs(lat_vec - latitude).argmin()
                alt = elev[i, j]
                return float("nan") if alt == d["nodata"] else float(alt)
        return float("nan")


###############################################################################
#  SIMULACIÓN DE MUOGRAFÍA                                                   #
###############################################################################


class MuographySimulation:
    """Gestión de topografía y trayectorias para muografía."""

    # ------------------------------------------------------------------
    def __init__(
        self,
        region: Tuple[float, float, float, float, str],
        points: int,
        srtm1_data: SRTMDataLoader,
        cmap: str = "terrain",
    ):
        self.lat1, self.lat2, self.lon1, self.lon2, self.name = region
        self.points = points
        self.srtm1_data = srtm1_data
        self.cmap = cmap
        # atributos que se poblarán más adelante
        self.X = self.Y = self.elevation = None  # type: ignore
        self.obsPX = self.obsPY = self.obsPZ = None  # observador (lon, lat, alt)

    # ------------------------------------------------------------------
    #  TOPOGRAFÍA
    # ------------------------------------------------------------------
    def load_elevation_data(self) -> None:
        """Construye las mallas X, Y y la matriz de elevaciones."""
        lats = np.linspace(self.lat1, self.lat2, self.points)
        lons = np.linspace(self.lon1, self.lon2, self.points)
        self.X, self.Y = np.meshgrid(lons, lats)
        self.elevation = np.array(
            [
                [self.srtm1_data.get_altitude(lat, lon) for lon in lons]
                for lat in lats
            ]
        )

    def save_elevation_data(self, file_name: str = "elevation_data.txt") -> None:
        np.savetxt(file_name, self.elevation, fmt="%.2f")
        print(f"✅ Datos de elevación guardados en '{file_name}'")

    # ------------------------------------------------------------------
    #  PLOTTERS 2D / 3D
    # ------------------------------------------------------------------
    def _make_3d_axes(self):
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection="3d")
        ax.set_xlabel("Longitud")
        ax.set_ylabel("Latitud")
        ax.set_zlabel("Elevación (m)")
        ax.view_init(elev=45, azim=135)
        ax.grid(False)
        return fig, ax

    def plot_topography_mesh(self, path: str = "topography_mesh.png") -> None:
        fig, ax = self._make_3d_axes()
        ax.set_title(f"Malla 3D – {self.name}")
        ax.plot_wireframe(self.X, self.Y, self.elevation, color="black", lw=0.4)
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"✅ Guardado: {path}")

    def plot_topography_surface(self, path: str = "topography_surface.png") -> None:
        fig, ax = self._make_3d_axes()
        ax.set_title(f"Superficie 3D – {self.name}")
        surf = ax.plot_surface(
            self.X, self.Y, self.elevation, cmap=self.cmap, linewidth=0, antialiased=False
        )
        fig.colorbar(surf, shrink=0.5, aspect=10, label="m s.n.m.")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"✅ Guardado: {path}")

    # ------------------------------------------------------------------
    #  CONFIGURACIÓN DEL DETECTOR / DIRECCIONES
    # ------------------------------------------------------------------
    def set_observation_points(
        self, obs_point: Tuple[float, float], ref_point: Tuple[float, float]
    ):
        """Define telescopio (obs) y punto de referencia en el relieve."""
        self.obsPY, self.obsPX = obs_point
        self.RefPY, self.RefPX = ref_point
        self.obsPZ = self.srtm1_data.get_altitude(*obs_point)
        self.RefPZ = self.srtm1_data.get_altitude(*ref_point)

    def plot_with_projection(self, path: str = "projection_view.png") -> None:
        fig, ax = self._make_3d_axes()
        ax.set_title(f"Vista desde observador – {self.name}")
        ax.plot_surface(
            self.X,
            self.Y,
            self.elevation,
            cmap=self.cmap,
            linewidth=0,
            antialiased=False,
            alpha=0.6,
        )
        # observador y referencia
        ax.scatter(self.obsPX, self.obsPY, self.obsPZ, c="red", s=80, label="Observador")
        ax.scatter(self.RefPX, self.RefPY, self.RefPZ, c="blue", s=80, label="Referencia")
        ax.plot(
            [self.obsPX, self.RefPX],
            [self.obsPY, self.RefPY],
            [self.obsPZ, self.RefPZ],
            c="k",
            ls="--",
            lw=1.5,
        )
        ax.legend()
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"✅ Guardado: {path}")

    # ------------------------------------------------------------------
    #  TOOLS: distancia máxima dentro del volcán
    # ------------------------------------------------------------------
    def _meters_to_dlat_dlon(self, dx: float, dy: float) -> Tuple[float, float]:
        """Convierte desplazamientos en metros → grados lat/lon."""
        d_lat = dy / 111_320.0
        d_lon = dx / (111_320.0 * math.cos(math.radians(self.obsPY)))
        return d_lat, d_lon

    def path_length(
        self,
        theta_deg: float,
        phi_deg: float,
        *,
        max_distance_m: float = 25_000,
        step_m: float = 25.0,
    ) -> float:
        """Longitud (cm) dentro de la topografía para (θ, φ)."""
        θ = math.radians(theta_deg)
        φ = math.radians(phi_deg)

        # dirección (unitaria) *entrante* en ENU (m)
        de = -math.sin(φ) * math.sin(θ)
        dn = -math.cos(φ) * math.sin(θ)
        du = -math.cos(θ)
        norm = math.hypot(math.hypot(de, dn), du)
        de, dn, du = de / norm, dn / norm, du / norm

        # recorrido paso a paso con detección de múltiples entradas/salidas
        dist_in = 0.0
        x_m = y_m = z_m = 0.0

        # posición inicial y evaluación de g = h_line - h_terrain
        dlat, dlon = self._meters_to_dlat_dlon(x_m, y_m)
        lat = self.obsPY + dlat
        lon = self.obsPX + dlon
        if not (self.lat1 <= lat <= self.lat2 and self.lon1 <= lon <= self.lon2):
            return 0.0  # fuera de la malla desde el inicio

        h_terrain_prev = self.srtm1_data.get_altitude(lat, lon)
        h_line_prev = self.obsPZ + z_m
        g_prev = h_line_prev - h_terrain_prev  # >0 arriba, <0 dentro

        n_steps = int(max_distance_m // step_m)
        for _ in range(n_steps):
            x_m += de * step_m
            y_m += dn * step_m
            z_m += du * step_m

            dlat, dlon = self._meters_to_dlat_dlon(x_m, y_m)
            lat = self.obsPY + dlat
            lon = self.obsPX + dlon
            if not (self.lat1 <= lat <= self.lat2 and self.lon1 <= lon <= self.lon2):
                break  # fuera de la malla

            h_terrain = self.srtm1_data.get_altitude(lat, lon)
            h_line = self.obsPZ + z_m
            g = h_line - h_terrain

            if g_prev < 0 and g < 0:
                # continúa dentro
                dist_in += step_m
            elif g_prev >= 0 and g < 0:
                # entrada: solo la fracción final del paso está dentro
                alpha = g_prev / (g_prev - g)
                dist_in += (1 - alpha) * step_m
            elif g_prev < 0 and g >= 0:
                # salida: solo la fracción inicial del paso estaba dentro
                alpha = g_prev / (g_prev - g)
                dist_in += alpha * step_m
            # fuera a fuera: nada

            g_prev = g  # actualizar para siguiente iteración

        return dist_in * 100.0  # cm


###############################################################################
#  MÓDULO PRINCIPAL (pruebas rápidas)                                        #
###############################################################################

if __name__ == "__main__":
    # Ejemplo mínimo de uso
    region = (4.466944, 4.500833, -75.404720, -75.372694, "Cerro Machín")
    srtm = SRTMDataLoader(["N04W075.hgt", "N04W076.hgt"])
    sim = MuographySimulation(region, points=200, srtm1_data=srtm)
    sim.load_elevation_data()
    sim.set_observation_points(obs_point=(4.48, -75.39), ref_point=(4.496, -75.388))
    L_cm = sim.path_length(30, 120)
    print(f"Recorrido dentro del terreno: {L_cm/100:.1f} m")

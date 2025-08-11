# srtm_loader.py – utilidades para cargar y consultar SRTM
# -------------------------------------------------------------
# ⬇️ Copia este archivo junto a MuTeLogic.py y MuTeFastSim.py
#    o instálalo como paquete local para que el resto del código
#    pueda hacer:  `from srtm_loader import SRTMDataLoader`
# -------------------------------------------------------------
from pathlib import Path
from typing import List, Dict
import numpy as np
import rasterio

class SRTMDataLoader:
    """Carga uno o varios mosaicos SRTM (.hgt) y permite consultar la
    altitud en cualquier punto dentro de sus límites.

    Ejemplo
    -------
    >>> loader = SRTMDataLoader(["N04W075.hgt", "N04W076.hgt"])
    >>> z = loader.get_altitude(4.49, -75.39)  # lat, lon
    """

    def __init__(self, file_paths: List[str]):
        self.tiles: Dict[str, Dict] = {}
        self._load_tiles([Path(p) for p in file_paths])

    # ------------------------------------------------------------------
    # 🔒 Métodos internos
    # ------------------------------------------------------------------
    def _load_tiles(self, paths: List[Path]):
        """Lee todos los archivos .hgt y guarda elevación + bounds."""
        for p in paths:
            if not p.exists():
                raise FileNotFoundError(p)
            with rasterio.open(p) as ds:
                elev = ds.read(1)
                self.tiles[str(p)] = {
                    "elevation": elev,
                    "bounds": ds.bounds,     # (left, bottom, right, top)
                    "nodata": ds.nodata if ds.nodata is not None else -32768,
                }

    # ------------------------------------------------------------------
    # 🧩  API pública
    # ------------------------------------------------------------------
    def get_altitude(self, latitude: float, longitude: float) -> float:
        """Devuelve la altitud (m) en lat/lon.  np.nan si no hay datos."""
        for t in self.tiles.values():
            b = t["bounds"]
            if b.left <= longitude <= b.right and b.bottom <= latitude <= b.top:
                elev = t["elevation"]
                # crear mallas de lat/lon correspondientes al array
                lon_vec = np.linspace(b.left,  b.right,  elev.shape[1])
                lat_vec = np.linspace(b.top,   b.bottom, elev.shape[0])
                ix = np.abs(lon_vec - longitude).argmin()
                iy = np.abs(lat_vec - latitude ).argmin()
                h = elev[iy, ix]
                return np.nan if h == t["nodata"] else float(h)
        return np.nan  # fuera de los mosaicos cargados

    # alias para mantener compatibilidad con algunos scripts
    __call__ = get_altitude


# -------------------------
# Ejecución de prueba rápida
# -------------------------
if __name__ == "__main__":
    files = ["N04W075.hgt", "N04W076.hgt"]
    srtm = SRTMDataLoader(files)
    print(srtm.get_altitude(4.4865, -75.3890))

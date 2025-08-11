#!/usr/bin/env python3
"""
Longitudes.py

Clasifica muones/antimuones de un .shw en libres/bloqueados,
calcula longitudes dentro de la topografía, y:
  • grafica (φ,θ) con eje Y invertido
  • guarda un .shw solo con los bloqueados
  • guarda un .shw con todos los muones (ID 0005/0006) y su longitud

Se actualiza θ y φ usando los componentes del momento (px, py, pz), sin tocar el resto del flujo.
"""
import argparse
from pathlib import Path
from MuTeLogic import MuographySimulation
from srtm_loader import SRTMDataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import math

# --- Cálculo de ángulos a partir de px, py, pz ---

def compute_theta(px, py, pz):
    p_mag = math.sqrt(px*px + py*py + pz*pz)
    if p_mag == 0.0:
        return None
    cos_th = max(-1.0, min(1.0, pz / p_mag))
    return math.degrees(math.acos(cos_th))


def compute_phi(px, py):
    phi = math.degrees(math.atan2(py, px))
    return phi if phi >= 0.0 else phi + 360.0

# Mantener entrada nombrada para consistencia
Entry = namedtuple("Entry", ["raw", "theta", "phi", "is_blocked", "quant_key"] )

# Función para cuantizar ángulos

def quantize(angle, step):
    return round(angle / step) * step

# Función para cálculo paralelo de longitudes por clave

def compute_path_lengths(sim, blocked_keys, use_parallel=False, max_workers=4):
    path_length_map = {}
    def worker(key):
        theta_q, phi_q = key
        return key, sim.path_length(theta_q, phi_q)
    if use_parallel:
        with ThreadPoolExecutor(max_workers=max_workers) as exe:
            for fut in as_completed({exe.submit(worker, k): k for k in blocked_keys}):
                key, L = fut.result()
                path_length_map[key] = L
    else:
        for key in blocked_keys:
            _, L = worker(key)
            path_length_map[key] = L
    return path_length_map

# Función principal

def main():
    parser = argparse.ArgumentParser(description="Procesa archivo .shw y clasifica muones bloqueados")
    parser.add_argument("-i", "--input", type=Path, default=Path("bga-2212-01_043200.shw"),
                        help=".shw de entrada")
    parser.add_argument("-c", "--clear-sky", type=Path, default=Path("AngulosNoBloqueados.csv"),
                        help="CSV de clear sky")
    parser.add_argument("--step-deg", type=float, default=0.5,
                        help="Resolución angular para cuantizar")
    parser.add_argument("--parallel", action="store_true",
                        help="Calcular path_length en paralelo")
    parser.add_argument("--blocked-out", type=Path, default=Path("muones_bloqueados_P1.shw"),
                        help="Salida: solo bloqueados")
    parser.add_argument("--all-out", type=Path, default=Path("muones_con_longitud.shw"),
                        help="Salida: todos los muones con longitud")
    args = parser.parse_args()

    # Configuración fija
    region_points = [4.466944, 4.500833, -75.404720, -75.372694, "Cerro Machín"]
    obs_point = (4.48653, -75.38895)
    ref_point = (4.48653, -75.38895)
    srtm_files = ["N04W075.hgt", "N04W076.hgt"]
    srtm_data = SRTMDataLoader(srtm_files)

    sim = MuographySimulation(region_points,
                              points=200,
                              srtm1_data=srtm_data,
                              cmap="terrain")
    sim.load_elevation_data()
    sim.set_observation_points(obs_point, ref_point)

    # Clear sky
    df_clear = pd.read_csv(args.clear_sky)
    theta_col = "theta_deg"
    phi_col = "phi_deg"
    clear_set = {(quantize(t, args.step_deg), quantize(p, args.step_deg))
                 for t, p in zip(df_clear[theta_col], df_clear[phi_col])}
    phi_min, phi_max = df_clear[phi_col].min(), df_clear[phi_col].max()

    # Recolectar entradas y keys bloqueados
    entries = []
    blocked_keys_set = set()
    allowed_ids = ("0005", "0006")

    with args.input.open() as fh:
        for line in fh:
            if not line.startswith(allowed_ids):
                continue
            sp = line.split()
            if len(sp) < 4:
                continue
            # Calcular ángulos desde momentos
            px, py, pz = map(float, sp[1:4])
            theta = compute_theta(px, py, pz)
            phi   = compute_phi(px, py)
            if theta is None or theta <= 60.0:
                continue
            if not (phi_min <= phi <= phi_max):
                continue
            quant_key = (quantize(theta, args.step_deg),
                         quantize(phi, args.step_deg))
            is_blocked = quant_key not in clear_set
            if is_blocked:
                blocked_keys_set.add(quant_key)
            entries.append(Entry(raw=line.rstrip("\n"),
                                 theta=theta,
                                 phi=phi,
                                 is_blocked=is_blocked,
                                 quant_key=quant_key))

    theta_all = np.array([e.theta for e in entries])
    phi_all   = np.array([e.phi   for e in entries])
    blocked_mask = np.array([e.is_blocked for e in entries])

    # Calcular longitudes para ángulos bloqueados únicos
    path_length_map = compute_path_lengths(sim,
                                          blocked_keys_set,
                                          use_parallel=args.parallel,
                                          max_workers=8 if args.parallel else 1)

    # Escritura de salidas
    blocked_length_by_line = {e.raw: path_length_map[e.quant_key]
                              for e in entries if e.is_blocked}

    with args.input.open() as fin, \
         args.blocked_out.open("w") as f_blocked, \
         args.all_out.open("w") as f_all:
        for line in fin:
            sp = line.strip().split()
            if not sp or sp[0] not in allowed_ids:
                continue
            raw = line.rstrip("\n")
            L = blocked_length_by_line.get(raw, 0.0)
            f_all.write(f"{raw}  {L:10.2f}\n")
            if raw in blocked_length_by_line:
                f_blocked.write(f"{raw}\n")

    # Estadísticas
    total = len(theta_all)
    blocked = blocked_mask.sum()
    free = total - blocked
    print(f"Eventos dentro del FOV : {total:,}")
    print(f"   • Libres            : {free:,}")
    print(f"   • Bloqueados        : {blocked:,}")
    print(f"📝 Muones bloqueados escritos en {args.blocked_out}")
    print(f"📝 Muones con longitud escritos en {args.all_out}")

    # Gráfica
    plt.figure(figsize=(9, 6))
    plt.scatter(phi_all[~blocked_mask], theta_all[~blocked_mask],
                s=6, label="Clear sky")
    plt.scatter(phi_all[blocked_mask], theta_all[blocked_mask],
                s=12, marker="x", label="Blocked (topografía)")
    plt.xlabel("Azimuth φ (°)")
    plt.ylabel("Zenith θ (°)")
    plt.title("Muones dentro del FOV – libres vs bloqueados")
    plt.xlim(phi_min, phi_max)
    plt.ylim(60, 90)
    plt.gca().invert_yaxis()
    plt.grid(alpha=0.3, ls="--")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

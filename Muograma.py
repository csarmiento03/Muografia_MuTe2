#!/usr/bin/env python3
"""
Pipeline combinado: a partir del CSV filtrado con 'survives' o 'passes' produce:
  1. Scatter θ–φ de los muones dentro del FOV: sobreviven (verde) vs mueren (rojo).
  2. Histograma 2D θ–φ de los que sobreviven.
Salida: una figura PNG con ambas subplots y estadísticas impresas.
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

def wrap_phi(phi):
    """Normaliza φ a [0,360)."""
    return np.mod(phi, 360.0)

def in_fov(theta, phi, theta_min, theta_max, phi_min, phi_max):
    """Máscara lógica de estar en el campo de visión, con manejo de wrap-around en φ."""
    th_mask = (theta >= theta_min) & (theta <= theta_max)
    if phi_min is None or phi_max is None:
        ph_mask = np.ones_like(phi, dtype=bool)
    else:
        if phi_min <= phi_max:
            ph_mask = (phi >= phi_min) & (phi <= phi_max)
        else:
            # wrap-around, p.ej phi_min=350, phi_max=10
            ph_mask = (phi >= phi_min) | (phi <= phi_max)
    return th_mask & ph_mask

def make_2d_hist(theta_vals, phi_vals, theta_bins, phi_bins):
    H, theta_edges, phi_edges = np.histogram2d(theta_vals, phi_vals, bins=[theta_bins, phi_bins])
    return H, theta_edges, phi_edges

def main():
    parser = argparse.ArgumentParser(description="Scatter + histograma 2D de supervivencia de muones en θ–φ.")
    parser.add_argument("-i", "--input", required=True, help="CSV con columnas 'prm_theta','prm_phi' y 'survives' o 'passes'.")
    parser.add_argument("-o", "--output-prefix", default="survival", help="Prefijo para la imagen de salida.")
    parser.add_argument("--theta-min", type=float, default=60.0, help="Theta mínimo en grados (zenith).")
    parser.add_argument("--theta-max", type=float, default=90.0, help="Theta máximo en grados (zenith).")
    parser.add_argument("--phi-min", type=float, default=None, help="Phi mínimo en grados (azimuth). Si no se da, se usa el rango del data.")
    parser.add_argument("--phi-max", type=float, default=None, help="Phi máximo en grados (azimuth). Si no se da, se usa el rango del data.")
    parser.add_argument("--scatter-size-survive", type=float, default=6, help="Tamaño de punto para sobrevivientes.")
    parser.add_argument("--scatter-size-died", type=float, default=12, help="Tamaño / marcador para los que murieron.")
    parser.add_argument("--theta-bins", type=int, default=50, help="Bins para histograma en θ.")
    parser.add_argument("--phi-bins", type=int, default=50, help="Bins para histograma en φ.")
    parser.add_argument("--log-hist", action="store_true", help="Grafica el histograma 2D en escala logarítmica.")
    parser.add_argument("--show", action="store_true", help="Mostrar la figura además de guardarla.")
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    # determinar columna de supervivencia
    if "survives" in df.columns:
        mask_surv = df["survives"].astype(bool)
    elif "passes" in df.columns:
        mask_surv = df["passes"].astype(bool)
    else:
        print("Error: no se encontró 'survives' ni 'passes' en el CSV.", file=sys.stderr)
        sys.exit(1)

    if "prm_theta" not in df.columns or "prm_phi" not in df.columns:
        print("Error: faltan columnas 'prm_theta' o 'prm_phi'.", file=sys.stderr)
        sys.exit(1)

    theta = df["prm_theta"].astype(float).to_numpy()
    phi = wrap_phi(df["prm_phi"].astype(float).to_numpy())

    survived_mask = mask_surv
    died_mask = ~mask_surv

    # aplicar FOV
    # si no dan phi_min/max, usar rango observados
    phi_min = args.phi_min if args.phi_min is not None else np.min(phi)
    phi_max = args.phi_max if args.phi_max is not None else np.max(phi)
    fov_mask = in_fov(theta, phi, args.theta_min, args.theta_max, phi_min, phi_max)

    surv_in = survived_mask & fov_mask
    died_in = died_mask & fov_mask

    # estadísticas
    total = len(theta)
    total_in_fov = np.count_nonzero(fov_mask)
    n_surv = np.count_nonzero(surv_in)
    n_died = np.count_nonzero(died_in)
    print(f"Total muones: {total:,}")
    print(f"Eventos en FOV: {total_in_fov:,}")
    print(f"  • Sobreviven en FOV: {n_surv:,}")
    print(f"  • Mueren en FOV:     {n_died:,}")

    # preparar figura
    fig, axs = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1, 1.2]})

    # Scatter plot: sobrevivientes vs muertos
    ax = axs[0]
    ax.scatter(
        phi[surv_in], theta[surv_in],
        s=args.scatter_size_survive, c="limegreen", label="Survive", alpha=0.7
    )
    ax.scatter(
        phi[died_in], theta[died_in],
        s=args.scatter_size_died, c="crimson", marker="x", label="Died"
    )
    ax.set_xlabel("Azimuth φ (°)")
    ax.set_ylabel("Zenith θ (°)")
    ax.set_title("Muon survival in θ–φ (FOV filtered)")
    ax.set_xlim(phi_min, phi_max)
    ax.set_ylim(args.theta_min, args.theta_max)  # invertir para que 0° arriba
    ax.invert_yaxis()  # refuerzo
    ax.legend(frameon=True)
    ax.grid(alpha=0.3, linestyle="--")

    # Histograma 2D de los sobrevivientes dentro del FOV
    theta_surv_vals = theta[surv_in]
    phi_surv_vals = phi[surv_in]
    if len(theta_surv_vals) == 0:
        print("Advertencia: no hay sobrevivientes dentro del FOV para el histograma 2D.", file=sys.stderr)
        H = np.zeros((args.theta_bins, args.phi_bins))
        theta_edges = np.linspace(args.theta_min, args.theta_max, args.theta_bins + 1)
        phi_edges = np.linspace(phi_min, phi_max, args.phi_bins + 1)
    else:
        H, theta_edges, phi_edges = make_2d_hist(theta_surv_vals, phi_surv_vals, args.theta_bins, args.phi_bins)

    ax2 = axs[1]
    # preparar datos para pcolormesh: phi en x, theta en y
    PHI, THETA = np.meshgrid(phi_edges, theta_edges)
    if args.log_hist:
        plot_H = np.log10(H + 1.0)
        cb_label = "log10(count + 1)"
    else:
        plot_H = H
        cb_label = "count"
    pcm = ax2.pcolormesh(PHI, THETA, plot_H, shading="auto")
    cbar = fig.colorbar(pcm, ax=ax2, pad=0.02)
    cbar.set_label(cb_label)
    ax2.set_xlabel("Azimuth φ (°)")
    ax2.set_ylabel("Zenith θ (°)")
    ax2.set_title("2D histogram of survivors (θ–φ)")
    ax2.set_ylim(args.theta_min, args.theta_max)  # invertir
    ax2.invert_yaxis()
    ax2.set_xlim(phi_min, phi_max)
    ax2.grid(alpha=0.2, linestyle=":")

    plt.tight_layout()
    out_png = f"{args.output_prefix}_survival_scatter_hist2d.png"
    fig.savefig(out_png, dpi=200)
    print(f"Figura guardada en {out_png}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()

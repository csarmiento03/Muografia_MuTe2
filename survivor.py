#!/usr/bin/env python3
"""
Versión paralela corregida y optimizada de survivor3/survivor4:
 - Procesamiento por chunks en paralelo usando procesos.
 - Vectorización interna (CSDA + decaimiento).
 - Filtrado temprano de muones por ID.
 - Uso correcto de índices tras usar `usecols`.
 - Escritura centralizada (sin condiciones de carrera).
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import math
import sys
import re
from concurrent.futures import ProcessPoolExecutor, as_completed

# Constantes físicas
MUON_MASS_MEV = 105.6583755  # masa en reposo del muón en MeV
C = 299792458.0  # velocidad de la luz en m/s
TAU0 = 2.1969811e-6  # vida media propia en segundos

# Globals en workers
GLOBAL_CSDA_DF = None
GLOBAL_ARGS = None
GLOBAL_PARTICLE_IDS_ALLOWED = None


def worker_init(csda_df, args_namespace, particle_ids_allowed):
    global GLOBAL_CSDA_DF, GLOBAL_ARGS, GLOBAL_PARTICLE_IDS_ALLOWED
    GLOBAL_CSDA_DF = csda_df
    GLOBAL_ARGS = args_namespace
    GLOBAL_PARTICLE_IDS_ALLOWED = particle_ids_allowed


def load_csda_table(path):
    raw_rows = []
    with open(path, "r") as f:
        for line in f:
            if re.match(r'^\s*\d+\.?\d*E[+\-]\d+', line, re.IGNORECASE):
                raw_rows.append(line)
                break
        for line in f:
            raw_rows.append(line)

    parsed = []
    for line in raw_rows:
        parts = line.strip().split()
        if len(parts) < 11:
            continue
        try:
            row = list(map(float, parts[:11]))
            parsed.append(row)
        except ValueError:
            continue

    if not parsed:
        raise RuntimeError(f"No se pudo parsear ninguna fila válida de {path}")

    cols = [
        "T_MeV", "p_MeV_c", "Ionization",
        "brems", "pair", "photonuc",
        "Radloss", "dEdx", "CSDA_range_g_cm2",
        "dltterm", "dEdx_R"
    ]
    df = pd.DataFrame(parsed, columns=cols)
    df = df.sort_values("T_MeV").reset_index(drop=True)
    return df


def csda_range_from_energy_array(T_MeV_array, df):
    t = df["T_MeV"].values
    r = df["CSDA_range_g_cm2"].values
    low = np.full_like(T_MeV_array, 0.0, dtype=float)
    high = np.full_like(T_MeV_array, float(r[-1]), dtype=float)
    res = np.interp(T_MeV_array, t, r)
    res = np.where(T_MeV_array <= t[0], low, res)
    res = np.where(T_MeV_array >= t[-1], high, res)
    return res


def survival_probability_array(T_MeV_array, length_m_array):
    E = T_MeV_array + MUON_MASS_MEV
    p_sq = E * E - MUON_MASS_MEV * MUON_MASS_MEV
    valid = p_sq > 0
    p = np.sqrt(np.clip(p_sq, 0.0, None))
    beta = np.zeros_like(E)
    gamma = np.zeros_like(E)
    beta[valid] = p[valid] / E[valid]
    gamma[valid] = E[valid] / MUON_MASS_MEV

    t_lab = np.zeros_like(E)
    tau_lab = np.zeros_like(E)
    with np.errstate(divide='ignore', invalid='ignore'):
        t_lab[valid] = length_m_array[valid] / (beta[valid] * C)
        tau_lab[valid] = gamma[valid] * TAU0
    prob = np.zeros_like(E)
    good = valid & (beta > 0) & (tau_lab > 0)
    prob[good] = np.exp(- t_lab[good] / tau_lab[good])
    return prob


def parse_corsika_id_series(series):
    digits = series.astype(str).str.extract(r'(\d+)', expand=False)
    return pd.to_numeric(digits, errors="coerce").astype("Int64")


def process_chunk_worker(args_tuple):
    chunk_index, df_chunk = args_tuple
    global GLOBAL_CSDA_DF, GLOBAL_ARGS, GLOBAL_PARTICLE_IDS_ALLOWED

    # Se esperan exactamente 8 columnas después de usecols: [0,1,2,3,9,10,11,12] del original
    if df_chunk.shape[1] < 8:
        print(f"[chunk {chunk_index}] salto: columnas insuficientes ({df_chunk.shape[1]} < 8)", file=sys.stderr)
        return chunk_index, None, None, 0, 0

    # Indices relativos en el chunk reducido
    IDX_CORSIKA = 0  # CorsikaId
    IDX_PX = 1
    IDX_PY = 2
    IDX_PZ = 3
    IDX_PRM_ENERGY = 4  # energía original de entrada
    IDX_THETA = 5
    IDX_PHI = 6
    IDX_LENGTH = 7

    # Filtrar por IDs de muones de interés
    corsika_ids = parse_corsika_id_series(df_chunk.iloc[:, IDX_CORSIKA])
    mask_muon = corsika_ids.isin(GLOBAL_PARTICLE_IDS_ALLOWED)
    if not mask_muon.any():
        return chunk_index, None, None, 0, 0

    df_muons = df_chunk.loc[mask_muon].copy()
    corsika_ids = parse_corsika_id_series(df_muons.iloc[:, IDX_CORSIKA])

    # Cálculo de momento y energía
    try:
        px = df_muons.iloc[:, IDX_PX].astype(float)
        py = df_muons.iloc[:, IDX_PY].astype(float)
        pz = df_muons.iloc[:, IDX_PZ].astype(float)
    except Exception as e:
        raise RuntimeError(f"[chunk {chunk_index}] Error leyendo momentos: {e}")
    energies_calc = np.sqrt(px ** 2 + py ** 2 + pz ** 2)

    # Ángulos y longitud
    try:
        theta_raw = df_muons.iloc[:, IDX_THETA].astype(float)
        phi_raw = df_muons.iloc[:, IDX_PHI].astype(float)
        lengths_raw = df_muons.iloc[:, IDX_LENGTH].astype(float)
    except Exception as e:
        raise RuntimeError(f"[chunk {chunk_index}] Error extrayendo columnas clave: {e}")

    # Conversión de energía según unidad especificada
    energies_MeV = energies_calc * (1e3 if GLOBAL_ARGS.energy_unit == "GeV" else 1.0)
    energies_GeV = energies_MeV / 1e3

    # Longitud y densidad de columna
    if GLOBAL_ARGS.length_unit == "cm":
        length_cm = lengths_raw
        length_m = lengths_raw / 100.0
        column_density = length_cm * GLOBAL_ARGS.density
    else:
        length_m = lengths_raw
        length_cm = lengths_raw * 100.0
        column_density = length_m * 100.0 * GLOBAL_ARGS.density

    # Rango CSDA y probabilidad de supervivencia
    csda_ranges = csda_range_from_energy_array(energies_MeV, GLOBAL_CSDA_DF)
    energy_sufficient = csda_ranges >= column_density
    survival_probs = survival_probability_array(energies_MeV, length_m)

    if GLOBAL_ARGS.min_survival > 0:
        survives = energy_sufficient & (survival_probs >= GLOBAL_ARGS.min_survival)
    else:
        survives = energy_sufficient.copy()

    out_df = pd.DataFrame({
        "CorsikaId": corsika_ids,
        "is_muon_id": True,
        "prm_energy_input": df_muons.iloc[:, IDX_PRM_ENERGY],
        "energy_MeV": energies_MeV,
        "energy_GeV": energies_GeV,
        "prm_theta": theta_raw,
        "prm_phi": phi_raw,
        "length_cm": length_cm,
        "length_m": length_m,
        "column_density_g_cm2": column_density,
        "csda_range_g_cm2": csda_ranges,
        "energy_sufficient": energy_sufficient,
        "survival_probability": survival_probs,
        "survives": survives,
        "passes": survives,
    })

    simple_cols = [
        "CorsikaId",
        "energy_GeV",
        "energy_MeV",
        "length_cm",
        "length_m",
        "survival_probability",
        "energy_sufficient",
        "survives",
        "prm_theta",
        "prm_phi",
    ]
    simplified = out_df[simple_cols].copy()

    total = len(out_df)
    n_surv = int(out_df["survives"].sum())

    return chunk_index, out_df, simplified, total, n_surv


def main():
    parser = argparse.ArgumentParser(
        description="Versión paralela y corregida: filtra muones y calcula supervivencia usando CSDA y decaimiento."
    )
    parser.add_argument("-i", "--input", required=True, help="Archivo .shw de entrada.")
    parser.add_argument("-r", "--rock-data", required=True, help="Tabla CSDA (por ejemplo data_rock.dat).")
    parser.add_argument("--density", "-d", type=float, default=2.65, help="Densidad de roca en g/cm^3.")
    parser.add_argument("--energy-unit", choices=["MeV", "GeV"], default="GeV", help="Unidad de energía.")
    parser.add_argument("--length-unit", choices=["cm", "m"], default="cm", help="Unidad de longitud en el archivo.")
    parser.add_argument("--particle-ids", type=str, default="5,6", help="IDs separados por coma (ej. 5,6).")
    parser.add_argument("--min-survival", type=float, default=0.0, help="Umbral mínimo de supervivencia.")
    parser.add_argument("-o", "--output", default="muon_passes.csv", help="CSV completo de salida.")
    parser.add_argument("--chunksize", type=int, default=500_000, help="Número de filas por chunk.")
    parser.add_argument("--workers", type=int, default=4, help="Cantidad de procesos paralelos.")
    args = parser.parse_args()

    # Cargar tabla CSDA
    try:
        csda_df = load_csda_table(args.rock_data)
    except Exception as e:
        print(f"Error cargando tabla CSDA: {e}", file=sys.stderr)
        sys.exit(1)

    # IDs permitidos
    particle_ids_allowed = set(int(x) for x in args.particle_ids.split(",") if x.strip())

    input_path = Path(args.input)
    output_full = Path(args.output)
    output_simple = Path(f"simplified_{args.output}")

    # Eliminar archivos previos si existen
    if output_full.exists():
        output_full.unlink()
    if output_simple.exists():
        output_simple.unlink()

    usecols = [0, 1, 2, 3, 9, 10, 11, 12]  # columnas de interés del .shw

    total_processed = 0
    total_survived = 0
    first_write_full = True
    first_write_simple = True

    try:
        reader = pd.read_csv(
            input_path,
            sep=r'\s+',
            engine='python',
            header=None,
            comment="#",
            usecols=usecols,
            dtype=str,
            chunksize=args.chunksize
        )
    except Exception as e:
        print(f"Error abriendo archivo de entrada: {e}", file=sys.stderr)
        sys.exit(1)

    with ProcessPoolExecutor(max_workers=args.workers,
                             initializer=worker_init,
                             initargs=(csda_df, args, particle_ids_allowed)) as executor:

        future_to_chunk = {}
        for chunk_idx, df_chunk in enumerate(reader):
            future = executor.submit(process_chunk_worker, (chunk_idx, df_chunk))
            future_to_chunk[future] = chunk_idx

        for future in as_completed(future_to_chunk):
            try:
                chunk_index, out_df, simplified_df, total_chunk, n_surv_chunk = future.result()
            except Exception as e:
                print(f"[worker error] {e}", file=sys.stderr)
                continue

            if out_df is None:
                continue

            # Escritura segura (solo en hilo principal)
            if first_write_full:
                out_df.to_csv(output_full, index=False, mode="w")
                first_write_full = False
            else:
                out_df.to_csv(output_full, index=False, mode="a", header=False)

            if first_write_simple:
                simplified_df.to_csv(output_simple, index=False, mode="w")
                first_write_simple = False
            else:
                simplified_df.to_csv(output_simple, index=False, mode="a", header=False)

            total_processed += total_chunk
            total_survived += n_surv_chunk

    # Resumen final
    total_muons = total_processed
    n_surv = total_survived
    n_died = total_muons - n_surv
    frac_surv = 100.0 * n_surv / total_muons if total_muons > 0 else 0.0
    frac_died = 100.0 * n_died / total_muons if total_muons > 0 else 0.0

    print(f"Procesados (muones IDs {','.join(str(x) for x in sorted(particle_ids_allowed))}): {total_muons}")
    print(f"Sobreviven: {n_surv} ({frac_surv:.1f}%)")
    print(f"Murieron: {n_died} ({frac_died:.1f}%)")
    print(f"CSV completo en: {output_full}")
    print(f"CSV simplificado para graficar en: {output_simple}")


if __name__ == "__main__":
    main()

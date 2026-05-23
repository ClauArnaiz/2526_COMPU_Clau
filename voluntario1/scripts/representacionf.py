import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

archivos = {
    16: "/home/claudia/Escritorio/voluntario1/datos_ising_N16.csv",
    32: "/home/claudia/Escritorio/voluntario1/datos_ising_N32.csv",
    64: "/home/claudia/Escritorio/voluntario1/datos_ising_N64.csv",
    128: "/home/claudia/Escritorio/voluntario1/datos_ising_N128.csv"
}

temperaturas_objetivo = [1.5, 2.3, 3.5]

for N, archivo in archivos.items():
    df = pd.read_csv(archivo)

    columnas_f = [col for col in df.columns if col.startswith("f_")]
    distancias = np.array([int(col.split("_")[1]) for col in columnas_f])

    plt.figure(figsize=(7, 5))

    for T_obj in temperaturas_objetivo:
        idx = (df["T"] - T_obj).abs().idxmin()
        T_real = df.loc[idx, "T"]

        f_vals = df.loc[idx, columnas_f].values

        plt.plot(distancias, f_vals, marker="o", label=f"T = {T_real:.2f}")

    plt.xlabel("Distancia i")
    plt.ylabel("f(i)")
    plt.title(f"Función de correlación para N = {N}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
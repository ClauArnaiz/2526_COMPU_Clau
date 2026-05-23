import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

archivos = {
    16: "/home/claudia/Escritorio/voluntario1/scripts/datos_ising_N16.csv",
    32: "/home/claudia/Escritorio/voluntario1/scripts/datos_ising_N32.csv",
    64: "/home/claudia/Escritorio/voluntario1/scripts/datos_ising_N64.csv",
    128: "/home/claudia/Escritorio/voluntario1/scripts/datos_ising_N128.csv"
}

Tc = 2.269
resultados_beta = []

for N, archivo in archivos.items():
    df = pd.read_csv(archivo)

    # Cogemos solo temperaturas por debajo de Tc
    df_sub = df[df["T"] < Tc].copy()

    # Evitamos puntos demasiado lejos o magnetizaciones no positivas
    df_sub = df_sub[df_sub["M"] > 0]

    x = np.log(Tc - df_sub["T"].values)
    y = np.log(df_sub["M"].values)

    # Ajuste lineal: ln(M) = beta ln(Tc-T) + cte
    beta, cte = np.polyfit(x, y, 1)

    resultados_beta.append({
        "N": N,
        "beta": beta
    })

    plt.figure(figsize=(6,4))
    plt.scatter(x, y, label="Datos")
    plt.plot(x, beta*x + cte, label=f"Ajuste: beta = {beta:.3f}")
    plt.xlabel(r"$\ln(T_c - T)$")
    plt.ylabel(r"$\ln(m)$")
    plt.title(f"Estimación de beta para N = {N}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

tabla_beta = pd.DataFrame(resultados_beta)
tabla_beta
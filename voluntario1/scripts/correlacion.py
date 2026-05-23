import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Configuración de rutas de archivos
archivos = {
    16: "/home/claudia/Escritorio/voluntario1/datos_ising_N16.csv",
    32: "/home/claudia/Escritorio/voluntario1/datos_ising_N32.csv",
    64: "/home/claudia/Escritorio/voluntario1/datos_ising_N64.csv",
    128: "/home/claudia/Escritorio/voluntario1/datos_ising_N128.csv"
}

# Temperaturas a analizar (evitamos T=1.5 porque no decae debido al orden ferromagnético)
temperaturas_objetivo = [2.39, 3.50]  

resultados_xi = []

for N, archivo in archivos.items():
    df = pd.read_csv(archivo)

    # Identificar y ordenar las columnas de la función de correlación f_i
    columnas_f = [col for col in df.columns if col.startswith("f_")]
    columnas_f = sorted(columnas_f, key=lambda x: int(x.split("_")[1]))

    distancias = np.array([int(col.split("_")[1]) for col in columnas_f])

    for T_obj in temperaturas_objetivo:
        # Buscar la temperatura real simulada más cercana a la objetivo
        idx = (df["T"] - T_obj).abs().idxmin()
        T_real = df.loc[idx, "T"]

        f_vals = df.loc[idx, columnas_f].values.astype(float)

        # MÁSCARA BASE: Evitamos i=0 (f(0)=1 siempre) y valores indeterminados (f <= 0)
        mask = (distancias > 0) & (f_vals > 0)

        # MÁSCARA GEOMÉTRICA: Máximo hasta N/2 para evitar el rebote de las condiciones periódicas
        mask = mask & (distancias <= N/2)

        # MÁSCARA TÉRMICA (CORRECCIÓN CRÍTICA): 
        # A altas temperaturas, el orden físico desaparece rápido. Filtramos el ruido de fondo
        # limitando el ajuste solo a los primeros pasos de decaimiento real (i <= 3).
        if T_obj > 3.0:
            mask = mask & (distancias <= 3)

        # Extracción de puntos válidos para la regresión
        x = distancias[mask]
        y = np.log(f_vals[mask])

        # Ajuste lineal: ln f(i) = a + b*i
        b, a = np.polyfit(x, y, 1)

        # La longitud de correlación es el inverso negativo de la pendiente
        xi = -1 / b

        resultados_xi.append({
            "N": N,
            "T": T_real,
            "pendiente": b,
            "xi": xi
        })

        # Generación de la gráfica del ajuste lineal
        plt.figure(figsize=(6, 4))
        plt.scatter(x, y, color='tab:blue', zorder=3, label="Datos Simulación")
        plt.plot(x, a + b*x, color='tab:orange', linestyle='--', label=f"Ajuste lineal ($\\xi$ = {xi:.2f})")
        plt.xlabel("Distancia $i$")
        plt.ylabel("$\ln f(i)$")
        plt.title(f"Ajuste de Correlación para $N$={N}, $T$={T_real:.2f}")
        plt.grid(True, linestyle=':')
        plt.legend()
        plt.tight_layout()
        plt.show()

# Creación y muestra de la tabla resumen de resultados
tabla_xi = pd.DataFrame(resultados_xi)
print("\n=== TABLA DE RESULTADOS DE LONGITUD DE CORRELACIÓN ===")
print(tabla_xi.to_string(index=False))
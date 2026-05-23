import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =====================================================================
# 1. CARGA DE DATOS Y ESTIMACIÓN DE Tc(N) MEDIANTE AJUSTE PARABÓLICO
# =====================================================================
archivos = {
    16: "/home/claudia/Escritorio/voluntario1/scripts/datos_ising_N16.csv",
    32: "/home/claudia/Escritorio/voluntario1/scripts/datos_ising_N32.csv",
    64: "/home/claudia/Escritorio/voluntario1/scripts/datos_ising_N64.csv",
    128: "/home/claudia/Escritorio/voluntario1/scripts/datos_ising_N128.csv"
}

resultados = []

for N, archivo in archivos.items():
    df = pd.read_csv(archivo)
    
    # Encontrar el índice del máximo absoluto directo
    idx_max = df["C"].idxmax()
    T_max_directo = df.loc[idx_max, "T"]
    C_max_directo = df.loc[idx_max, "C"]
    
    # Ajuste parabólico local usando el máximo y sus dos vecinos contiguos
    if idx_max > 0 and idx_max < len(df) - 1:
        T_fit = df.loc[idx_max-1:idx_max+1, "T"].values
        C_fit = df.loc[idx_max-1:idx_max+1, "C"].values
        
        # Ajuste de polinomio de grado 2: C = a*T^2 + b*T + c
        a, b, c = np.polyfit(T_fit, C_fit, 2)
        
        # El vértice de la parábola nos da el máximo interpolado
        Tc_parabolico = -b / (2*a)
        Cmax_parabolico = a*Tc_parabolico**2 + b*Tc_parabolico + c
    else:
        Tc_parabolico = T_max_directo
        Cmax_parabolico = C_max_directo
        
    resultados.append({
        "N": N,
        "Tc directo": T_max_directo,
        "Cmax directo": C_max_directo,
        "Tc parabólico": Tc_parabolico,
        "Cmax parabólico": Cmax_parabolico
    })

# Convertir a DataFrame y mostrar la tabla de picos calculados
tabla_Tc = pd.DataFrame(resultados)
print("=== TABLA DE VALORES CRÍTICOS POR TAMAÑO ===")
print(tabla_Tc.to_string(index=False))
print("\n" + "="*45 + "\n")

# =====================================================================
# 2. EXTRAPOLACIÓN AL LÍMITE TERMODINÁMICO (N → ∞)
# =====================================================================
N_vals = tabla_Tc["N"].values
Tc_vals = tabla_Tc["Tc parabólico"].values

# Definimos la variable x como 1/N para la extrapolación lineal
x = 1 / N_vals

# Ajuste lineal: Tc(N) = a * (1/N) + Tc_inf
coef = np.polyfit(x, Tc_vals, 1)
a_slope, Tc_inf = coef

# Generar puntos para pintar la recta de ajuste desde 1/N = 0 (infinito)
x_fit = np.linspace(0, max(x) * 1.1, 100)
Tc_fit = a_slope * x_fit + Tc_inf

# Visualización gráfica de la extrapolación
plt.figure(figsize=(7, 5))
plt.scatter(x, Tc_vals, color="tab:blue", s=60, zorder=3, label="Datos simulados ($T_c$ parabólico)")
plt.plot(x_fit, Tc_fit, color="tab:orange", linestyle="-", label=f"Ajuste lineal: $T_c(\\infty) = {Tc_inf:.4f}$")

# Línea de referencia teórica de Onsager
plt.axhline(2.269, color="black", linestyle="--", label="Exacto de Onsager: $T_c \\approx 2.269$")

# Configuración del gráfico
plt.xlabel("$1/N$ (Inverso del tamaño lineal)")
plt.ylabel("$T_c(N)$")
plt.title("Extrapolación de la Temperatura Crítica ($N \\rightarrow \\infty$)")
plt.xlim(0, max(x) * 1.1)
plt.grid(True, linestyle=":")
plt.legend(loc="upper left")
plt.tight_layout()

# Guardar y mostrar imagen
plt.savefig("extrapolacion_Tc.png", dpi=300)
plt.show()

print(f"🎯 Temperatura crítica extrapolada para el sistema infinito Tc(∞) = {Tc_inf:.4f}")
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

# =====================================================
# PARÁMETROS DEL EXPERIMENTO 
# =====================================================
N_values = [16, 32, 64, 128]
T_values = np.linspace(1.5, 3.5, 10) # 10 temperaturas en el intervalo [1.5, 3.5]

PMC_TERM = 20000
PMC_TOTAL = 1020000 # Total de pasos Monte Carlo (incluyendo termalización)
MEDIDA_CADA = 100 ## Medir cada 100 PMC

# Número REAL de medidas después de termalizar
TOTAL_MEDIDAS = (PMC_TOTAL - PMC_TERM) // MEDIDA_CADA

# Temperatura crítica exacta de Onsager
Tc_onsager = 2.269

# =====================================================
# FUNCIONES FÍSICAS
# =====================================================
def paso_monte_carlo(red, T, N):
    """
    Un paso Monte Carlo usando checkerboard update para acelerar el algoritmo de Metropolis.
    """
    for offset in [0, 1]:
        vecinos = (np.roll(red, 1, axis=0)+ np.roll(red, -1, axis=0)+ np.roll(red, 1, axis=1)+ np.roll(red, -1, axis=1))
        delta_E = 2 * red * vecinos
        p_aceptacion = np.exp(-np.clip(delta_E, 0, None) / T)
        aleatorio = np.random.rand(N, N)
        voltear = (delta_E <= 0) | (aleatorio < p_aceptacion)
        i, j = np.indices((N, N))
        mascara = (i + j) % 2 == offset
        red[voltear & mascara] *= -1
    return red

def calcular_energia_total(red):
    """
    Energía total del sistema:
    E = -1/2 Σ s_i s_j
    """
    vecinos = (np.roll(red, 1, axis=0)+ np.roll(red, -1, axis=0)+ np.roll(red, 1, axis=1)+ np.roll(red, -1, axis=1))
    E = -0.5 * np.sum(red * vecinos)
    return E

def calcular_correlacion(red, i_max=30):
    """
    f(i) = < s(n,m) s(n+i,m) >
    """
    f = np.zeros(i_max)
    for i in range(i_max):
        f[i] = np.mean(red * np.roll(red, i, axis=0))
    return f

# =====================================================
# DICCIONARIOS PARA GUARDAR RESULTADOS DE PLOTS
# =====================================================
resultados_M = {}
resultados_E = {}
resultados_C = {}

print("======================================")
print("INICIANDO SIMULACIÓN MODELO DE ISING")
print("======================================")

# =====================================================
# BUCLE PRINCIPAL
# =====================================================
for N in N_values:

    print(f"\n========== Simulación para N = {N} ==========")
    resultados_N = []
    lista_M = []
    lista_E = []
    lista_C = []

    for T in tqdm(T_values, desc=f"N={N}"):
        # ==================================
        # 1. CONFIGURACIÓN INICIAL ORDENADA
        # ==================================
        red = np.ones((N, N), dtype=int)

        # ==================================
        # 2. TERMALIZACIÓN
        # ==================================
        for _ in range(PMC_TERM):
            red = paso_monte_carlo(red, T, N)

        # ==================================
        # 3. MEDIDAS
        # ==================================
        medidas_M = np.zeros(TOTAL_MEDIDAS)
        medidas_E = np.zeros(TOTAL_MEDIDAS)

        f_i_acumulada = np.zeros(30)

        for m_idx in range(TOTAL_MEDIDAS):

            # Esperar 100 pMC entre medidas
            for _ in range(MEDIDA_CADA):
                red = paso_monte_carlo(red, T, N)

            # Magnetización
            M = np.abs(np.sum(red)) / (N**2)

            # Energía total
            E = calcular_energia_total(red)
            medidas_M[m_idx] = M
            medidas_E[m_idx] = E

            # Correlación
            f_i_acumulada += calcular_correlacion(red, i_max=30)

        # ==================================
        # 4. PROMEDIOS Y ERRORES
        # ==================================
        # ---- Magnetización promedio ----
        M_prom = np.mean(medidas_M)
        M_err = np.std(medidas_M) / np.sqrt(TOTAL_MEDIDAS)

        # ---- Energía media por enlace ----
        E_prom = np.mean(medidas_E) / (2 * N**2)
        E_err = (np.std(medidas_E / (2 * N**2)) / np.sqrt(TOTAL_MEDIDAS))

        # ---- Calor específico ----
        # C = ( <E²> - <E>² ) / (N² T)
        E2_prom = np.mean(medidas_E**2)
        E_prom_total = np.mean(medidas_E)
        C = (E2_prom - E_prom_total**2) / ((N**2) * T)

        # ---- Correlación ----
        F_i_prom = f_i_acumulada / TOTAL_MEDIDAS

        # ==================================
        # 5. GUARDAR RESULTADOS
        # ==================================
        fila = [T, M_prom, M_err, E_prom, E_err, C] + F_i_prom.tolist()
        resultados_N.append(fila)
        lista_M.append(M_prom)
        lista_E.append(E_prom)
        lista_C.append(C)

    # ==================================
    # GUARDAR CSV
    # ==================================
    columnas = ( ['T', 'M', 'M_err', 'E', 'E_err', 'C']+ [f'f_{i}' for i in range(30)])

    df = pd.DataFrame(resultados_N, columns=columnas)

    nombre_csv = f"datos_ising_N{N}.csv"

    df.to_csv(nombre_csv, index=False)

    print(f"✅ Guardado: {nombre_csv}")

    # Guardar para plots globales

    resultados_M[N] = lista_M
    resultados_E[N] = lista_E
    resultados_C[N] = lista_C

# =====================================================
# PLOTS FINALES
# =====================================================
print("\n======================================")
print("GENERANDO GRÁFICAS")
print("======================================")

fig, axs = plt.subplots(1, 3, figsize=(18, 5))

# =====================================================
# MAGNETIZACIÓN
# =====================================================
for N in N_values:
    axs[0].plot(T_values, resultados_M[N], marker='o', label=f'N={N}')

axs[0].axvline(x=Tc_onsager, color='k', linestyle='--', label=r'$T_c$ Onsager')

axs[0].set_title('Magnetización Promedio')
axs[0].set_xlabel('Temperatura T')
axs[0].set_ylabel(r'$m_N$')
axs[0].legend()
axs[0].grid(True)

# =====================================================
# ENERGÍA
# =====================================================
for N in N_values:
    axs[1].plot(T_values, resultados_E[N], marker='o', label=f'N={N}')

axs[1].axvline(x=Tc_onsager, color='k', linestyle='--')
axs[1].set_title('Energía Media por enlace')
axs[1].set_xlabel('Temperatura T')
axs[1].set_ylabel(r'$e_N$')
axs[1].grid(True)

# =====================================================
# CALOR ESPECÍFICO
# =====================================================
for N in N_values: 
    axs[2].plot(T_values, resultados_C[N], marker='o', label=f'N={N}')

axs[2].axvline(x=Tc_onsager, color='k', linestyle='--')
axs[2].set_title('Calor Específico')
axs[2].set_xlabel('Temperatura T')
axs[2].set_ylabel(r'$c_N$')
axs[2].legend()
axs[2].grid(True)

# =====================================================
# GUARDAR FIGURA
# =====================================================
plt.tight_layout()

plt.savefig("resultados_ising.png", dpi=300)

plt.show()

print("\n======================================")
print("SIMULACIÓN FINALIZADA")
print("======================================")
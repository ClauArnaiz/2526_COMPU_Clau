import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time  
from tqdm import tqdm

# =====================================================
# PARÁMETROS DEL EXPERIMENTO 
# =====================================================
N_values = [16, 32, 64, 128]
T_values = np.linspace(1.5, 3.5, 10) # 10 temperaturas en el intervalo [1.5, 3.5]

PMC_TERM = 200000
PMC_TOTAL = 1020000# Total de pasos Monte Carlo (incluyendo termalización)
MEDIDA_CADA = 100 # Medir cada 100 PMC

# Número REAL de medidas después de termalizar
TOTAL_MEDIDAS = (PMC_TOTAL - PMC_TERM) // MEDIDA_CADA
 
# Temperatura crítica exacta de Onsager
Tc_onsager = 2.269

# =====================================================
# FUNCIONES FÍSICAS OPTIMIZADAS
# =====================================================

def paso_monte_carlo_optimo(red, T, mascara_blanca, mascara_negra):
    """Versión optimizada pasándole las máscaras ya creadas"""
    # subred blanca
    vecinos = (np.roll(red, 1, axis=0) + np.roll(red, -1, axis=0) +
               np.roll(red, 1, axis=1) + np.roll(red, -1, axis=1))
    delta_E = 2 * red * vecinos
    p_aceptacion = np.exp(-np.clip(delta_E, 0, None) / T)
    aleatorio = np.random.rand(*red.shape)
    voltear = (delta_E <= 0) | (aleatorio < p_aceptacion)
    red[voltear & mascara_blanca] *= -1
    
    # subred negra
    vecinos = (np.roll(red, 1, axis=0) + np.roll(red, -1, axis=0) +
               np.roll(red, 1, axis=1) + np.roll(red, -1, axis=1))
    delta_E = 2 * red * vecinos
    p_aceptacion = np.exp(-np.clip(delta_E, 0, None) / T)
    aleatorio = np.random.rand(*red.shape)
    voltear = (delta_E <= 0) | (aleatorio < p_aceptacion)
    red[voltear & mascara_negra] *= -1
    
    return red

def calcular_energia_total(red):
    """Energía total del sistema: E = -1/2 Σ s_i s_j"""
    vecinos = (np.roll(red, 1, axis=0) + np.roll(red, -1, axis=0) + 
               np.roll(red, 1, axis=1) + np.roll(red, -1, axis=1))
    E = -0.5 * np.sum(red * vecinos)
    return E

def calcular_correlacion(red, i_max=30):
    """f(i) = < s(n,m) s(n+i,m) >"""
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
resultados_M_err = {}  
resultados_E_err = {}  
resultados_C_err = {}  

print("======================================")
print("INICIANDO SIMULACIÓN MODELO DE ISING")
print("======================================")

# =====================================================
# BUCLE PRINCIPAL CON CRONÓMETRO Y OPTIMIZACIÓN
# =====================================================
tiempos_ejecucion = []  

for N in N_values:
    t_inicio = time.time()  

    print(f"\n========== Simulación para N = {N} ==========")
    resultados_N = []
    lista_M = []
    lista_E = []
    lista_C = []
    lista_M_err = []  # Inicialización corregida
    lista_E_err = []  # Inicialización corregida
    lista_C_err = []  # Error del calor específico

    # Creación eficiente de máscaras estáticas para este tamaño N
    i, j = np.indices((N, N))
    mascara_blanca = (i + j) % 2 == 0
    mascara_negra = (i + j) % 2 == 1

    for T in tqdm(T_values, desc=f"N={N}"):
        # 1. CONFIGURACIÓN INICIAL ORDENADA
        red = np.ones((N, N), dtype=int)

        # 2. TERMALIZACIÓN
        for _ in range(PMC_TERM):
            red = paso_monte_carlo_optimo(red, T, mascara_blanca, mascara_negra)

        # 3. MEDIDAS
        medidas_M = np.zeros(TOTAL_MEDIDAS)
        medidas_E = np.zeros(TOTAL_MEDIDAS)
        f_i_acumulada = np.zeros(30)

        for m_idx in range(TOTAL_MEDIDAS):
            for _ in range(MEDIDA_CADA):
                red = paso_monte_carlo_optimo(red, T, mascara_blanca, mascara_negra)

            M = np.abs(np.sum(red)) / (N**2)
            E = calcular_energia_total(red)
            
            medidas_M[m_idx] = M
            medidas_E[m_idx] = E
            f_i_acumulada += calcular_correlacion(red, i_max=30)

        # 4. PROMEDIOS Y ERRORES
        M_prom = np.mean(medidas_M)
        M_err = np.std(medidas_M) / np.sqrt(TOTAL_MEDIDAS)

        E_prom = np.mean(medidas_E) / (2 * N**2)
        E_err = np.std(medidas_E / (2 * N**2)) / np.sqrt(TOTAL_MEDIDAS)

        # Calor específico corregido con T**2
        E2_prom = np.mean(medidas_E**2)
        E_prom_total = np.mean(medidas_E)
        C = (E2_prom - E_prom_total**2) / ((N**2) * (T**2))

        # Error del calor específico por bootstrap
        # Se re-muestrean las energías medidas y se recalcula C muchas veces.
        # La desviación típica de esos valores se usa como incertidumbre.
        n_boot = 300
        C_boot = np.zeros(n_boot)

        for b in range(n_boot):
            muestra = np.random.choice(medidas_E, size=len(medidas_E), replace=True)
            C_boot[b] = (np.mean(muestra**2) - np.mean(muestra)**2) / ((N**2) * (T**2))

        C_err = np.std(C_boot)

        F_i_prom = f_i_acumulada / TOTAL_MEDIDAS

        # 5. GUARDAR RESULTADOS EN LAS LISTAS DE LA ITERACIÓN
        fila = [T, M_prom, M_err, E_prom, E_err, C, C_err] + F_i_prom.tolist()
        resultados_N.append(fila)
        lista_M.append(M_prom)
        lista_E.append(E_prom)
        lista_C.append(C)
        lista_M_err.append(M_err)  # Guardado corregido
        lista_E_err.append(E_err)  # Guardado corregido
        lista_C_err.append(C_err)  # Guardado del error del calor específico

    # GUARDAR CSV
    columnas = ['T', 'M', 'M_err', 'E', 'E_err', 'C', 'C_err'] + [f'f_{idx}' for idx in range(30)]
    df = pd.DataFrame(resultados_N, columns=columnas)
    nombre_csv = f"datos_ising_N{N}.csv"
    df.to_csv(nombre_csv, index=False)
    print(f"Guardado correctamente: {nombre_csv}")

    # Guardar en memoria para los gráficos
    resultados_M[N] = lista_M
    resultados_E[N] = lista_E
    resultados_C[N] = lista_C
    resultados_M_err[N] = lista_M_err  
    resultados_E_err[N] = lista_E_err  
    resultados_C_err[N] = lista_C_err  

    t_fin = time.time()
    tiempo_total_N = t_fin - t_inicio
    tiempos_ejecucion.append(tiempo_total_N)  
    print(f" Tiempo total para N={N}: {tiempo_total_N:.2f} segundos")

# =====================================================
# PLOT 1: COSTE COMPUTACIONAL EN ESCALA LINEAL
# =====================================================
print("\nGenerando gráfica de tiempos de ejecución...")

plt.figure(figsize=(6, 4))
plt.plot(N_values, tiempos_ejecucion, 's--', color='purple', label='Tiempo medido')
plt.title('Coste computacional: tiempo de ejecución frente a N')
plt.xlabel('Tamaño de red N')
plt.ylabel('Tiempo total de ejecución (s)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("tiempo_vs_N_lineal.png", dpi=300)
plt.show()

# =====================================================
# PLOT 2: COSTE COMPUTACIONAL EN ESCALA LOG-LOG
# =====================================================
N_array = np.array(N_values)
t_array = np.array(tiempos_ejecucion)

coef = np.polyfit(np.log(N_array), np.log(t_array), 1)
alpha = coef[0]
A = np.exp(coef[1])

t_ajuste = A * N_array**alpha

plt.figure(figsize=(6, 4))
plt.loglog(N_array, t_array, 'o', color='purple', label='Tiempo medido')
plt.loglog(N_array, t_ajuste, '--', color='black',
           label=rf'Ajuste $t \sim N^{{{alpha:.2f}}}$')
plt.title('Coste computacional en escala log-log')
plt.xlabel('Tamaño de red N')
plt.ylabel('Tiempo total de ejecución (s)')
plt.grid(True, which='both')
plt.legend()
plt.tight_layout()
plt.savefig("tiempo_vs_N_loglog.png", dpi=300)
plt.show()

print(f"Exponente estimado del coste computacional: alpha = {alpha:.2f}")

# =====================================================
# PLOT 3: OBSERVABLES TERMODINÁMICOS CON ERRORES
# =====================================================
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

# PANEL 1: MAGNETIZACIÓN
for N in N_values:
    m_prom = np.array(resultados_M[N])
    m_err = np.array(resultados_M_err[N])

    axs[0].errorbar(
        T_values, m_prom, yerr=m_err,
        marker='o', capsize=3, linewidth=1.5,
        label=f'N={N}'
    )

axs[0].axvline(x=Tc_onsager, color='k', linestyle='--', label=r'$T_c$ Onsager')
axs[0].set_title('Magnetización Promedio')
axs[0].set_xlabel('Temperatura T')
axs[0].set_ylabel(r'$m_N$')
axs[0].legend()
axs[0].grid(True)


# PANEL 2: ENERGÍA
for N in N_values:
    e_prom = np.array(resultados_E[N])
    e_err = np.array(resultados_E_err[N])

    axs[1].errorbar(
        T_values, e_prom, yerr=e_err,
        marker='o', capsize=3, linewidth=1.5,
        label=f'N={N}'
    )

axs[1].axvline(x=Tc_onsager, color='k', linestyle='--')
axs[1].set_title('Energía Media por enlace')
axs[1].set_xlabel('Temperatura T')
axs[1].set_ylabel(r'$e_N$')
axs[1].grid(True)


# PANEL 3: CALOR ESPECÍFICO
for N in N_values:
    c_prom = np.array(resultados_C[N])
    c_err = np.array(resultados_C_err[N])

    axs[2].errorbar(
        T_values, c_prom, yerr=c_err,
        marker='o', capsize=3, linewidth=1.5,
        label=f'N={N}'
    )

axs[2].axvline(x=Tc_onsager, color='k', linestyle='--')
axs[2].set_title('Calor Específico')
axs[2].set_xlabel('Temperatura T')
axs[2].set_ylabel(r'$c_N$')
axs[2].legend()
axs[2].grid(True)

plt.tight_layout()
plt.savefig("resultados_ising_barras_error.png", dpi=300)
plt.show()


print("\n======================================")
print("SIMULACIÓN FINALIZADA CON ÉXITO")
print("======================================")
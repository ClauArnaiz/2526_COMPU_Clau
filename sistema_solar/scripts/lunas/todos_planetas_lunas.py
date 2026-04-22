import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random

# ==============================================================================
# 1. DATOS COMPLETOS + ENJAMBRE DE LUNAS (Ajustado para velocidad)
# ==============================================================================
M_SOLAR = 1988500.0 
AU_KM = 149.6       

# Datos: [Nombre, Masa (10^24kg), Perihelio (10^6km), Excentricidad]
datos = [
    ["Sol",      M_SOLAR, 0.0,    0.0],
    ["Mercurio", 0.330,   46.0,   0.205],
    ["Venus",    4.87,    107.5,  0.007],
    ["Tierra",   5.97,    147.1,  0.017],
    ["Marte",    0.642,   206.6,  0.094],
    ["Jupiter",  1898,    740.5,  0.049],
    ["Saturno",  568,     1352.6, 0.057],
    ["Urano",    86.8,    2741.3, 0.046],
    ["Neptuno",  102,     4444.5, 0.011]
]

# Añadimos 20 lunas caóticas (suficientes para ver el caos sin colgar el PC)
n_lunas = 20
for i in range(n_lunas):
    d_relativa = random.uniform(0.5, 2.0) # Distancia respecto a Jupiter en UA
    # Masa grande (100) para que se note la influencia gravitatoria
    datos.append([f"L_{i}", 100.0, 740.5 + (d_relativa * AU_KM), 0.1])

n = len(datos)
m = np.array([p[1] / M_SOLAR for p in datos])
r = np.zeros((n, 2))
v = np.zeros((n, 2))

for i, p in enumerate(datos):
    if i == 0: continue
    r_ua = p[2] / AU_KM
    r[i, 0] = r_ua
    # Añadimos una pequeña perturbación aleatoria en la velocidad para generar caos
    v[i, 1] = np.sqrt((1 + p[3]) / r_ua) + random.uniform(-0.03, 0.03)

# Parámetros de simulación optimizados
h = 0.05       # Paso de tiempo (grande para ver movimiento rápido)
pasos = 2000   # Menos pasos para que el cálculo sea instantáneo
historico_r = []

# ==============================================================================
# 2. MOTOR FÍSICO OPTIMIZADO (Evita el error de antes)
# ==============================================================================
def calcular_aceleracion(pos, masas):
    num_cuerpos = len(masas)
    acc = np.zeros_like(pos)
    eps = 0.05 # "Softening parameter" para evitar fuerzas infinitas en choques
    
    for i in range(num_cuerpos):
        for j in range(i + 1, num_cuerpos):
            diff = pos[j] - pos[i]
            dist_sq = np.sum(diff**2) + eps**2
            dist_inv3 = 1.0 / (dist_sq * np.sqrt(dist_sq))
            
            f_grav = diff * dist_inv3
            acc[i] += masas[j] * f_grav
            acc[j] -= masas[i] * f_grav
    return acc

# --- Ejecución del algoritmo de Verlet ---
print(f"Calculando trayectoria de {n} cuerpos... Espera un momento.")
a_actual = calcular_aceleracion(r, m)

for t_step in range(pasos):
    # Guardamos 1 de cada 5 posiciones para que la animación vaya fluida
    if t_step % 5 == 0: 
        historico_r.append(r.copy())
    
    # Algoritmo de Verlet
    r_nuevo = r + h * v + 0.5 * h**2 * a_actual
    a_nueva = calcular_aceleracion(r_nuevo, m)
    v = v + 0.5 * h * (a_actual + a_nueva)
    
    r = r_nuevo
    a_actual = a_nueva

# ==============================================================================
# 3. ANIMACIÓN DEL CAOS TOTAL
# ==============================================================================
fig, ax = plt.subplots(figsize=(9,9))
ax.set_facecolor('black')
fig.patch.set_facecolor('black')

# Zoom para ver todo el sistema (hasta Neptuno)
ax.set_xlim(-40, 40)
ax.set_ylim(-40, 40)
ax.set_aspect('equal')

# Colores para planetas y lunas
colores_planetas = ['yellow', 'grey', 'orange', 'blue', 'red', 'brown', 'tan', 'cyan', 'royalblue']
colores = colores_planetas + ['white'] * n_lunas

puntos = [ax.plot([], [], 'o', color=colores[i], ms=5 if i < 9 else 2)[0] for i in range(n)]

def animar(i):
    for j in range(n):
        pos = historico_r[i][j]
        puntos[j].set_data([pos[0]], [pos[1]])
    return puntos

ani = FuncAnimation(fig, animar, frames=len(historico_r), interval=20, blit=True)

# Estética de los ejes en blanco
ax.tick_params(colors='white')
for spine in ax.spines.values(): spine.set_color('white')
plt.title("EXPERIMENTO: Sistema Solar + Enjambre Caótico", color='white')

plt.show()
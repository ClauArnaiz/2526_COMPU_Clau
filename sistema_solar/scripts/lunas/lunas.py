import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random

# ==============================================================================
# 1. CONFIGURACIÓN Y DATOS (Incluyendo el Enjambre de Lunas)
# ==============================================================================
M_SOLAR = 1988500.0 
AU_KM = 149.6       

# [Nombre, Masa (10^24kg), Perihelio (10^6km), Excentricidad]
datos = [
    ["Sol",      M_SOLAR, 0.0,    0.0],
    ["Mercurio", 0.330,   46.0,   0.205],
    ["Venus",    4.87,    107.5,  0.007],
    ["Tierra",   5.97,    147.1,  0.017],
    ["Marte",    0.642,   206.6,  0.094],
    ["Jupiter",  1898,    740.5,  0.049]
]

# --- AQUÍ ROMPEMOS EL SISTEMA: AÑADIMOS 50 LUNAS ---
n_lunas = 50
masa_luna_caotica = 50.0 # Masas grandes para que "tiren" de Júpiter
radio_enjambre = 0.8     # UA alrededor de Júpiter

for i in range(n_lunas):
    # Distancia aleatoria y ángulo aleatorio alrededor de Júpiter
    d_relativa = random.uniform(0.1, radio_enjambre)
    angulo = random.uniform(0, 2 * np.pi)
    dist_total = 740.5 + (d_relativa * AU_KM) # Perihelio aproximado
    datos.append([f"L_{i}", masa_luna_caotica, dist_total, 0.01])

n = len(datos)
m = np.array([p[1] / M_SOLAR for p in datos])
r = np.zeros((n, 2))
v = np.zeros((n, 2))

for i, p in enumerate(datos):
    if i == 0: continue
    r_ua = p[2] / AU_KM
    r[i, 0] = r_ua
    # Velocidad orbital con un poco de ruido aleatorio para causar caos
    v_base = np.sqrt((1 + p[3]) / r_ua)
    v[i, 1] = v_base + random.uniform(-0.05, 0.05) 

# Parámetros de simulación
h = 0.02 # Paso un poco más grande para ver el caos rápido
pasos = 5000
historico_r = []

# ==============================================================================
# 2. MOTOR FÍSICO (VERLET)
# ==============================================================================
def calcular_aceleracion(pos, masas):
    acc = np.zeros_like(pos)
    for i in range(n):
        for j in range(i + 1, n):
            diff = pos[j] - pos[i]
            dist_sq = np.sum(diff**2)
            dist = np.sqrt(dist_sq)
            f = masas[j] * diff / (dist**3)
            acc[i] += f
            acc[j] -= (masas[i]/masas[j]) * f
    return acc

a_actual = calcular_aceleracion(r, m)

print("Calculando el desastre gravitatorio...")
for t_step in range(pasos):
    if t_step % 10 == 0: historico_r.append(r.copy())
    
    # Verlet
    r_nuevo = r + h * v + 0.5 * h**2 * a_actual
    a_nueva = calcular_aceleracion(r_nuevo, m)
    v = v + 0.5 * h * (a_actual + a_nueva)
    r, a_actual = r_nuevo, a_nueva

# ==============================================================================
# 3. ANIMACIÓN DEL CAOS
# ==============================================================================
fig, ax = plt.subplots(figsize=(8,8))
ax.set_facecolor('black')
# Hacemos zoom en la zona de Júpiter para ver las lunas pelearse
ax.set_xlim(-8, 8)
ax.set_ylim(-8, 8)

colores = ['yellow', 'grey', 'orange', 'blue', 'red', 'brown'] + ['white']*n_lunas
tamanos = [100, 20, 30, 30, 25, 80] + [5]*n_lunas

puntos = [ax.plot([], [], 'o', color=colores[i], ms=(tamanos[i]/10))[0] for i in range(n)]

def animar(i):
    for j in range(n):
        pos = historico_r[i][j]
        puntos[j].set_data([pos[0]], [pos[1]])
    return puntos

ani = FuncAnimation(fig, animar, frames=len(historico_r), interval=20, blit=True)
plt.title("Experimento: Inestabilidad con 50 Lunas en Júpiter")
plt.show()